"""calibrate.py — one-time calibration + tolerance-envelope builder (CLI).

Subcommands / Подкоманды:
  calibrate      Checkerboard -> mm_per_pixel + camera matrix + distortion.
  envelope       Build mean +/- k*sigma envelope from known-good parts, with a
                 held-out validation split and a reported false-reject rate.
  repeatability  Measure one static part N times; report per-feature sigma
                 (gauge R&R sanity check).

/ Калибровка по шахматной доске, построение допуска с валидацией на отложенной
  выборке и проверка повторяемости (gauge R&R).
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from camera import Camera
from inference import extract_geometry
from utils import (
    SIDE_LOWER,
    SIDE_TWO,
    SIDE_UPPER,
    STATUS_OK,
    calibrate_camera_from_images,
    feature_sidedness,
    load_config,
    matrix_to_list,
    save_config,
)


# --------------------------------------------------------------------------- #
# Image loading helpers / Помощники загрузки изображений
# --------------------------------------------------------------------------- #
def _load_images(directory: str) -> List[np.ndarray]:
    """Load all readable images from a directory, sorted by name.

    / Загружает все изображения из каталога, отсортированные по имени.
    """
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    paths: List[str] = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(directory, pat)))
    images: List[np.ndarray] = []
    for path in sorted(paths):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            images.append(img)
    return images


# --------------------------------------------------------------------------- #
# Calibration / Калибровка
# --------------------------------------------------------------------------- #
def run_calibration(cfg: Dict[str, Any], images_dir: str) -> Dict[str, Any]:
    """Calibrate from checkerboard images and write results into ``cfg``.

    / Калибрует по изображениям доски и записывает результаты в cfg.
    """
    chk = cfg["calibration"]["checkerboard"]
    pattern_size = (int(chk["cols"]), int(chk["rows"]))
    square_mm = float(chk["square_size_mm"])

    images = _load_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No checkerboard images in '{images_dir}'.")

    result = calibrate_camera_from_images(images, pattern_size, square_mm)

    cfg["calibration"]["mm_per_pixel"] = result.mm_per_pixel
    cfg["calibration"]["camera_matrix"] = matrix_to_list(result.camera_matrix)
    cfg["calibration"]["dist_coeffs"] = matrix_to_list(result.dist_coeffs)
    cfg["calibration"]["image_size"] = list(result.image_size)
    cfg["calibration"]["rms_reproj_error_px"] = result.rms_reproj_error_px

    print(
        f"[calibrate] views={result.num_views}  "
        f"mm/px={result.mm_per_pixel:.5f}  "
        f"RMS reproj={result.rms_reproj_error_px:.3f}px"
    )
    return cfg


# --------------------------------------------------------------------------- #
# Envelope building / Построение допуска
# --------------------------------------------------------------------------- #
def _measure_many(
    frames: Sequence[np.ndarray], cfg: Dict[str, Any]
) -> Tuple[Dict[str, List[float]], int]:
    """Measure a batch of frames; collect per-feature value lists.

    Returns (values_by_feature, num_failed_extractions). Frames that do not
    yield an OK status are counted but excluded from the statistics.
    / Измеряет партию кадров; собирает значения по признакам.
    """
    values: Dict[str, List[float]] = {}
    failed = 0
    for frame in frames:
        geo = extract_geometry(frame, cfg)
        if geo.get("status") != STATUS_OK:
            failed += 1
            continue
        merged = dict(geo["features"])
        merged.update(geo.get("roundness", {}))
        for name, val in merged.items():
            if isinstance(val, (int, float)) and np.isfinite(val):
                values.setdefault(name, []).append(float(val))
    return values, failed


def build_envelope(
    train_frames: Sequence[np.ndarray],
    holdout_frames: Sequence[np.ndarray],
    cfg: Dict[str, Any],
    k_sigma: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a mean +/- k*sigma envelope on TRAIN and validate on HOLD-OUT.

    Critically, the envelope statistics come only from ``train_frames`` and the
    false-reject rate is reported on the separate ``holdout_frames``. Computing
    both on the same parts would guarantee an in-sample FRR of ~0 and hide a
    threshold that fails to generalise — the classic calibration-bias trap.
    / Допуск строится ТОЛЬКО на обучающей выборке, а доля ложных отбраковок
      считается на ОТЛОЖЕННОЙ. Иначе FRR искусственно нулевой (ловушка смещения).
    """
    k = float(k_sigma if k_sigma is not None else cfg["envelope"].get("k_sigma", 3.0))

    train_values, train_failed = _measure_many(train_frames, cfg)
    if not train_values:
        raise RuntimeError("No good-part measurements succeeded on the train set.")

    # Generous sigma floor: a coincidentally-tight train set (or a metric that
    # is nearly constant, e.g. base squareness) must not yield an absurdly
    # narrow band. The floor is a fraction of the feature mean, honouring the
    # brief's "generous envelope sigma rather than a single hard nominal".
    # / Щедрый нижний порог sigma: доля от среднего, чтобы полоса не была узкой.
    min_sigma_frac = float(cfg["envelope"].get("min_sigma_frac_of_mean", 0.0))

    features_env: Dict[str, Any] = {}
    for name, vals in train_values.items():
        arr = np.asarray(vals, dtype=np.float64)
        mean = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        sigma = max(sigma, abs(mean) * min_sigma_frac, 1e-4)
        side = feature_sidedness(name)
        # One-sided form tolerances leave the "good" direction unbounded.
        # / Односторонние допуски оставляют «хорошую» сторону без границы.
        lower = round(mean - k * sigma, 5) if side in (SIDE_TWO, SIDE_LOWER) else None
        upper = round(mean + k * sigma, 5) if side in (SIDE_TWO, SIDE_UPPER) else None
        features_env[name] = {
            "mean": round(mean, 5),
            "sigma": round(sigma, 5),
            "side": side,
            "lower": lower,
            "upper": upper,
            "n": int(len(arr)),
        }

    cfg["envelope"]["k_sigma"] = k
    cfg["envelope"]["features"] = features_env

    # --- Held-out validation / Валидация на отложенной выборке --- #
    from inference import classify  # local import to avoid cycle at module load

    holdout_values, holdout_failed = _measure_many(holdout_frames, cfg)
    n_holdout = len(holdout_frames)
    rejects = 0
    per_frame_measured = _measure_per_frame(holdout_frames, cfg)
    for feats in per_frame_measured:
        if feats is None:
            rejects += 1  # unmeasurable good part counts as a (false) reject
            continue
        if classify(feats, cfg["envelope"])["verdict"] != "PASS":
            rejects += 1
    frr = (rejects / n_holdout) if n_holdout else None
    cfg["envelope"]["held_out_false_reject_rate"] = (
        round(frr, 4) if frr is not None else None
    )

    print(f"[envelope] k={k}  features={len(features_env)}  "
          f"train_n={sum(v['n'] for v in features_env.values()) // max(1, len(features_env))}")
    # Report any spec tolerances that override the learned band.
    # / Показываем допуски по чертежу, перекрывающие обученную полосу.
    for name, ov in cfg["envelope"].get("manual_bounds", {}).items():
        parts = ", ".join(f"{side}={val}" for side, val in ov.items())
        print(f"[envelope] manual bound (spec): {name} {parts}")
    if frr is not None:
        print(f"[envelope] held-out false-reject rate = {frr:.3f} "
              f"({rejects}/{n_holdout} good parts rejected)")
        if frr >= 0.5:
            print("[envelope] WARNING: FRR >= 0.5 — threshold does NOT generalise; "
                  "increase k or check illumination/repeatability.")
    if train_failed or holdout_failed:
        print(f"[envelope] note: extraction failed on "
              f"{train_failed} train + {holdout_failed} holdout frame(s).")
    return cfg


def _measure_per_frame(
    frames: Sequence[np.ndarray], cfg: Dict[str, Any]
) -> List[Optional[Dict[str, float]]]:
    """Per-frame feature dicts (or None on non-OK extraction), preserving order.

    / Признаки по кадрам (или None при неуспехе), с сохранением порядка.
    """
    out: List[Optional[Dict[str, float]]] = []
    for frame in frames:
        geo = extract_geometry(frame, cfg)
        if geo.get("status") != STATUS_OK:
            out.append(None)
        else:
            out.append(dict(geo["features"]))
    return out


# --------------------------------------------------------------------------- #
# Repeatability / gauge R&R / Повторяемость
# --------------------------------------------------------------------------- #
def run_repeatability(
    frames: Sequence[np.ndarray], cfg: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    """Measure the same static part N times; report per-feature sigma.

    A gauge R&R sanity check: this measurement sigma must be small relative to
    the tolerance band width, otherwise the gauge cannot discriminate good from
    bad parts regardless of how the envelope is set.
    / Многократное измерение одной статичной детали; sigma должна быть мала
      относительно ширины допуска, иначе прибор не различает годное/брак.
    """
    values, failed = _measure_many(frames, cfg)
    report: Dict[str, Dict[str, float]] = {}
    env_feats = cfg.get("envelope", {}).get("features", {})
    for name, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        entry = {
            "mean": round(float(np.mean(arr)), 5),
            "sigma": round(sigma, 6),
            "n": int(len(arr)),
        }
        band = env_feats.get(name)
        if band is not None and band.get("lower") is not None and band.get("upper") is not None:
            width = float(band["upper"]) - float(band["lower"])
            if width > 0:
                # Fraction of the tolerance band consumed by measurement noise.
                # / Доля полосы допуска, «съедаемая» шумом измерения.
                entry["sigma_over_band"] = round(sigma / width, 4)
        elif band is not None:
            # One-sided band: compare noise to the single-sided margin (k*sigma).
            # / Односторонняя полоса: сравнение шума с односторонним запасом.
            half = float(band.get("sigma", 0.0)) * float(cfg["envelope"].get("k_sigma", 3.0))
            if half > 0:
                entry["sigma_over_margin"] = round(sigma / half, 4)
        report[name] = entry

    cfg["envelope"]["measurement_repeatability"] = {
        n: r["sigma"] for n, r in report.items()
    }
    print(f"[repeatability] n={len(frames)}  failed_extractions={failed}")
    for name, r in sorted(report.items()):
        if "sigma_over_band" in r:
            extra = f"  sigma/band={r['sigma_over_band']:.3f}"
        elif "sigma_over_margin" in r:
            extra = f"  sigma/margin={r['sigma_over_margin']:.3f}"
        else:
            extra = ""
        print(f"  {name:32s} mean={r['mean']:.4f} sigma={r['sigma']:.5f}{extra}")
    return report


# --------------------------------------------------------------------------- #
# Frame acquisition for envelope/repeatability / Захват кадров
# --------------------------------------------------------------------------- #
def _grab_n(cfg: Dict[str, Any], n: int) -> List[np.ndarray]:
    """Grab N averaged frames from the configured camera/fallback source.

    / Захватывает N усреднённых кадров с камеры/резервного источника.
    """
    frames: List[np.ndarray] = []
    with Camera(cfg["camera"]) as cam:
        for _ in range(n):
            frames.append(cam.grab_averaged())
    return frames


# --------------------------------------------------------------------------- #
# CLI / Интерфейс командной строки
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calibration & envelope builder.")
    p.add_argument("--config", default="config.json", help="Path to config.json")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("calibrate", help="Checkerboard camera calibration.")
    c.add_argument("--images", required=True, help="Dir of checkerboard images.")

    e = sub.add_parser("envelope", help="Build tolerance envelope from good parts.")
    e.add_argument("--train", help="Dir of good-part images (train).")
    e.add_argument("--holdout", help="Dir of good-part images (held-out).")
    e.add_argument("--live", type=int, metavar="N",
                   help="Grab N frames live; split 70/30 train/holdout.")
    e.add_argument("--k", type=float, default=None, help="Sigma multiplier (default from config).")

    r = sub.add_parser("repeatability", help="Gauge R&R on one static part.")
    r.add_argument("--images", help="Dir of repeat images of one static part.")
    r.add_argument("--live", type=int, metavar="N", help="Grab N live frames of one part.")

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "calibrate":
        cfg = run_calibration(cfg, args.images)
        save_config(cfg, args.config)
        print(f"[calibrate] saved -> {args.config}")
        return 0

    if args.command == "envelope":
        if args.live:
            frames = _grab_n(cfg, args.live)
            split = int(len(frames) * 0.7)
            train, holdout = frames[:split], frames[split:]
        else:
            if not (args.train and args.holdout):
                raise SystemExit("Provide --train and --holdout dirs, or --live N.")
            train = _load_images(args.train)
            holdout = _load_images(args.holdout)
        if not train or not holdout:
            raise SystemExit("Empty train or holdout set.")
        cfg = build_envelope(train, holdout, cfg, k_sigma=args.k)
        save_config(cfg, args.config)
        print(f"[envelope] saved -> {args.config}")
        return 0

    if args.command == "repeatability":
        frames = _grab_n(cfg, args.live) if args.live else _load_images(args.images or "")
        if not frames:
            raise SystemExit("No frames for repeatability check.")
        run_repeatability(frames, cfg)
        save_config(cfg, args.config)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
