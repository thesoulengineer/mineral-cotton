"""demo.py — inline demonstration of the calibrate -> envelope -> classify flow.

Generates synthetic sample images (a checkerboard set, a batch of good parts,
and a few defective parts), then runs the full pipeline end-to-end so the
system is verifiable without plant hardware. Also serves as a smoke test.
/ Демонстрация полного цикла на синтетических образцах: калибровка, построение
  допуска, классификация. Работает без заводского оборудования.
"""

from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np

from calibrate import build_envelope, run_calibration, run_repeatability
from camera import render_synthetic_part
from inference import inspect
from utils import default_config, save_config


SAMPLES_DIR = "samples"


# --------------------------------------------------------------------------- #
# Synthetic checkerboard / Синтетическая шахматная доска
# --------------------------------------------------------------------------- #
def render_checkerboard(
    cols: int, rows: int, square_px: int, warp: float = 0.0, seed: int = 0,
    image_size: tuple = (1200, 1200),
) -> np.ndarray:
    """Render a checkerboard with (cols x rows) *inner* corners on the camera frame.

    The board is drawn onto a fixed ``image_size`` canvas — the SAME resolution
    as production part frames, because one physical camera images both the
    calibration target and the parts. A small perspective warp per view gives
    cv2.calibrateCamera the parallax it needs.
    / Доска рисуется на холсте того же размера, что и рабочие кадры (одна камера).
    """
    board_cols, board_rows = cols + 1, rows + 1
    bh = board_rows * square_px
    bw = board_cols * square_px
    board = np.full((bh, bw), 255, np.uint8)
    for r in range(board_rows):
        for c in range(board_cols):
            if (r + c) % 2 == 0:
                board[r * square_px:(r + 1) * square_px,
                      c * square_px:(c + 1) * square_px] = 0
    # Center the board on the full camera frame / центрируем доску на кадре.
    H, W = image_size
    img = np.full((H, W), 255, np.uint8)
    oy, ox = (H - bh) // 2, (W - bw) // 2
    img[oy:oy + bh, ox:ox + bw] = board
    if warp > 0:
        rng = np.random.default_rng(seed)
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        jitter = rng.uniform(-warp, warp, size=(4, 2)).astype(np.float32) * min(H, W)
        M = cv2.getPerspectiveTransform(src, src + jitter)
        img = cv2.warpPerspective(img, M, (W, H), borderValue=255)
    return img


# --------------------------------------------------------------------------- #
# Sample generation / Генерация образцов
# --------------------------------------------------------------------------- #
def generate_samples() -> None:
    """Write checkerboard, good-part and defect images under samples/.

    / Записывает изображения доски, годных деталей и брака в samples/.
    """
    os.makedirs(os.path.join(SAMPLES_DIR, "checkerboard"), exist_ok=True)
    os.makedirs(os.path.join(SAMPLES_DIR, "good_train"), exist_ok=True)
    os.makedirs(os.path.join(SAMPLES_DIR, "good_holdout"), exist_ok=True)
    os.makedirs(os.path.join(SAMPLES_DIR, "repeat"), exist_ok=True)
    os.makedirs(os.path.join(SAMPLES_DIR, "defects"), exist_ok=True)

    # Checkerboard views (9x6 inner corners, 5 mm squares -> config), rendered at
    # the same 1200x1200 resolution as part frames. 50 px/square => 0.10 mm/px.
    for i in range(10):
        board = render_checkerboard(9, 6, square_px=50, warp=0.012 * (1 + i % 4), seed=i)
        cv2.imwrite(os.path.join(SAMPLES_DIR, "checkerboard", f"cb_{i:02d}.png"), board)

    rng = np.random.default_rng(42)

    def good_part(seed: int) -> np.ndarray:
        # Nominal part (75/60/25 mm circles, 100 mm base at 0.10 mm/px) with
        # small, realistic part-to-part variation.
        # / Номинальная деталь (75/60/25 мм) с небольшой реалистичной вариацией.
        return render_synthetic_part(
            outer_d_px=750.0 + rng.normal(0, 3.0),
            pocket_d_px=600.0 + rng.normal(0, 2.5),
            hole_d_px=250.0 + rng.normal(0, 2.0),
            center_jitter_px=(rng.normal(0, 1.5), rng.normal(0, 1.5)),
            ellipticity=1.0 + rng.normal(0, 0.004),
            seed=seed,
        )

    for i in range(40):
        cv2.imwrite(os.path.join(SAMPLES_DIR, "good_train", f"good_{i:02d}.png"), good_part(1000 + i))
    for i in range(20):
        cv2.imwrite(os.path.join(SAMPLES_DIR, "good_holdout", f"good_{i:02d}.png"), good_part(2000 + i))

    # Repeatability: the SAME static part imaged many times (only sensor noise).
    # / Повторяемость: одна и та же деталь, меняется только шум сенсора.
    for i in range(20):
        cv2.imwrite(
            os.path.join(SAMPLES_DIR, "repeat", f"rep_{i:02d}.png"),
            render_synthetic_part(seed=None),
        )

    # Defects: oversize hole, bad concentricity (hole offset from the rings),
    # out-of-round outer feature.
    # / Дефекты: увеличенное отверстие, нарушенная соосность, овальность.
    cv2.imwrite(os.path.join(SAMPLES_DIR, "defects", "hole_oversize.png"),
                render_synthetic_part(hole_d_px=320.0, seed=7))   # 32 mm vs 25 mm
    cv2.imwrite(os.path.join(SAMPLES_DIR, "defects", "eccentric_hole.png"),
                render_synthetic_part(hole_offset_px=(45.0, 32.0), seed=8))  # ~5.5 mm
    cv2.imwrite(os.path.join(SAMPLES_DIR, "defects", "out_of_round.png"),
                render_synthetic_part(ellipticity=0.86, seed=9))
    print(f"[demo] samples written under '{SAMPLES_DIR}/'")


def _load_dir(sub: str) -> List[np.ndarray]:
    from calibrate import _load_images
    return _load_images(os.path.join(SAMPLES_DIR, sub))


# --------------------------------------------------------------------------- #
# End-to-end demo / Сквозная демонстрация
# --------------------------------------------------------------------------- #
def main() -> int:
    generate_samples()
    cfg = default_config()
    cfg["camera"]["fallback_source"] = "directory"
    cfg["camera"]["fallback_directory"] = os.path.join(SAMPLES_DIR, "repeat")

    print("\n=== 1) Calibration ===")
    cfg = run_calibration(cfg, os.path.join(SAMPLES_DIR, "checkerboard"))

    print("\n=== 2) Envelope (train + held-out validation) ===")
    cfg = build_envelope(_load_dir("good_train"), _load_dir("good_holdout"), cfg)

    print("\n=== 3) Repeatability / gauge R&R ===")
    run_repeatability(_load_dir("repeat"), cfg)

    save_config(cfg, "config.json")
    print("\n[demo] wrote calibrated config.json")

    print("\n=== 4) Classify good vs defective parts ===")
    good = _load_dir("good_holdout")[0]
    res = inspect(good, cfg)
    print(f"  good part      -> status={res['status']} verdict={res['verdict']} "
          f"features={res.get('features')}")

    for name in ("hole_oversize", "eccentric_hole", "out_of_round"):
        img = cv2.imread(os.path.join(SAMPLES_DIR, "defects", f"{name}.png"), cv2.IMREAD_UNCHANGED)
        res = inspect(img, cfg)
        viol = [v.get("feature") for v in res.get("violations", [])]
        print(f"  {name:14s} -> status={res['status']} verdict={res['verdict']} "
              f"violations={viol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
