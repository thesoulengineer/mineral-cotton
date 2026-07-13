"""main.py — live inspection loop: capture -> inference -> verdict.

Ties the camera, calibration and envelope together: grab an averaged frame,
extract geometry, classify against the envelope, overlay results for the
operator, and log every inspection for traceability.
/ Живой цикл контроля: захват -> измерение -> вердикт, с наложением на кадр и
  журналированием каждой проверки для прослеживаемости.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from camera import Camera
from inference import inspect
from utils import (
    FEATURE_NAMES,
    VERDICT_PASS,
    build_undistort_maps,
    list_to_matrix,
    load_config,
)


# --------------------------------------------------------------------------- #
# Logging / Журналирование
# --------------------------------------------------------------------------- #
class InspectionLogger:
    """Append every inspection to both CSV and JSONL for traceability.

    JSONL keeps the full structured record; CSV gives a flat, spreadsheet-
    friendly view. Both are opened in append mode so a crash never loses prior
    rows. / Пишет каждую проверку в CSV и JSONL (дозапись, без потери строк).
    """

    def __init__(self, csv_path: str, jsonl_path: str) -> None:
        self._csv_path = csv_path
        self._jsonl_path = jsonl_path
        # Stable, predefined column order so the header is identical regardless
        # of whether the first logged part measured OK or failed extraction.
        # / Фиксированный порядок колонок независимо от первой записи.
        roundness_cols = [n.replace("_diameter_mm", "_roundness") for n in FEATURE_NAMES]
        self._csv_fields: List[str] = (
            ["timestamp", "status", "verdict", "num_violations"]
            + list(FEATURE_NAMES)
            + ["concentricity_mm", "base_squareness_deg"]
            + roundness_cols
            + ["violations"]
        )

    def _flatten(self, record: Dict[str, Any]) -> Dict[str, Any]:
        flat: Dict[str, Any] = {
            "timestamp": record["timestamp"],
            "status": record["status"],
            "verdict": record["verdict"],
            "num_violations": len(record.get("violations", [])),
        }
        for name, val in record.get("features", {}).items():
            flat[name] = val
        for name, val in record.get("roundness", {}).items():
            flat[name] = val
        flat["violations"] = ";".join(
            v.get("feature", "?") for v in record.get("violations", [])
        )
        return flat

    def log(self, record: Dict[str, Any]) -> None:
        # JSONL — full record / полный JSON-объект на строку.
        with open(self._jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        # CSV — flattened row with a stable, predefined column schema.
        # / Плоская строка CSV с фиксированной схемой колонок.
        flat = self._flatten(record)
        write_header = not os.path.exists(self._csv_path) or os.path.getsize(self._csv_path) == 0
        with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._csv_fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(flat)


# --------------------------------------------------------------------------- #
# Overlay / Наложение
# --------------------------------------------------------------------------- #
def draw_overlay(frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    """Draw detected circles, fitted ellipses and measured values on a frame.

    / Рисует найденные круги, эллипсы и измеренные значения на кадре.
    """
    canvas = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    canvas = canvas.copy()

    passed = result.get("verdict") == VERDICT_PASS
    color = (0, 200, 0) if passed else (0, 0, 255)  # BGR / зелёный-красный

    for ellipse in result.get("ellipses", []) or []:
        if ellipse is not None:
            cv2.ellipse(canvas, ellipse, color, 2)

    for (cx, cy) in result.get("centers_px", []) or []:
        cv2.drawMarker(canvas, (int(cx), int(cy)), color, cv2.MARKER_CROSS, 14, 2)

    # Header banner / верхняя плашка.
    banner = f"{result.get('status', '?')} | {result.get('verdict', '?')}"
    cv2.putText(canvas, banner, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    y = 60
    for name, val in (result.get("features", {}) or {}).items():
        cv2.putText(canvas, f"{name}: {val:.3f}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 24
    for viol in result.get("violations", []) or []:
        txt = f"! {viol.get('feature', '?')}"
        if "value" in viol:
            lo = viol.get("lower")
            hi = viol.get("upper")
            lo_s = "-inf" if lo is None else f"{lo}"
            hi_s = "+inf" if hi is None else f"{hi}"
            txt += f" {viol['value']} not in [{lo_s},{hi_s}]"
        cv2.putText(canvas, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y += 22
    return canvas


# --------------------------------------------------------------------------- #
# Inspection record / Запись проверки
# --------------------------------------------------------------------------- #
def _build_record(result: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble the serializable inspection record (drops ndarray/ellipses)."""
    return {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": result.get("status"),
        "verdict": result.get("verdict"),
        "features": result.get("features", {}),
        "roundness": result.get("roundness", {}),
        "contrast": result.get("contrast", {}),
        "violations": result.get("violations", []),
    }


def _make_undistort_maps(cfg: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Precompute undistort remap tables once if calibration is present."""
    cal = cfg.get("calibration", {})
    cam_mtx = list_to_matrix(cal.get("camera_matrix"))
    dist = list_to_matrix(cal.get("dist_coeffs"))
    size = cal.get("image_size")
    if cam_mtx is None or dist is None or size is None:
        return None
    return build_undistort_maps(cam_mtx, dist, (int(size[0]), int(size[1])))


# --------------------------------------------------------------------------- #
# Live loop / Живой цикл
# --------------------------------------------------------------------------- #
def run(
    cfg: Dict[str, Any],
    once: bool = False,
    max_iterations: Optional[int] = None,
    show: bool = False,
    save_overlays_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run the inspection loop.

    ``once`` inspects a single part and returns. ``max_iterations`` bounds a
    continuous run (used by tests/CI). ``show`` opens an operator display
    window; ``save_overlays_dir`` writes overlay PNGs instead (headless).
    / Запускает цикл контроля: одна деталь или непрерывно; окно оператора или
      сохранение оверлеев (безоконный режим).
    """
    log_cfg = cfg.get("logging", {})
    logger = InspectionLogger(
        log_cfg.get("csv_path", "inspections.csv"),
        log_cfg.get("jsonl_path", "inspections.jsonl"),
    )
    maps = _make_undistort_maps(cfg)
    records: List[Dict[str, Any]] = []

    if save_overlays_dir:
        os.makedirs(save_overlays_dir, exist_ok=True)

    with Camera(cfg["camera"]) as cam:
        print(f"[main] camera backend = {cam.active_backend}")
        i = 0
        while True:
            frame = cam.grab_averaged()
            result = inspect(frame, cfg, maps)
            record = _build_record(result)
            logger.log(record)
            records.append(record)

            print(f"[{record['timestamp']}] status={record['status']} "
                  f"verdict={record['verdict']} "
                  f"violations={len(record['violations'])}")

            if show or save_overlays_dir:
                overlay = draw_overlay(frame, result)
                if save_overlays_dir:
                    path = os.path.join(save_overlays_dir, f"inspection_{i:04d}.png")
                    cv2.imwrite(path, overlay)
                if show:  # pragma: no cover - needs a display
                    cv2.imshow("inspection", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            i += 1
            if once:
                break
            if max_iterations is not None and i >= max_iterations:
                break

    if show:  # pragma: no cover
        cv2.destroyAllWindows()
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live 2D geometric inspection loop.")
    p.add_argument("--config", default="config.json", help="Path to config.json")
    p.add_argument("--once", action="store_true", help="Inspect one part and exit.")
    p.add_argument("--iterations", type=int, default=None,
                   help="Stop after N inspections (continuous mode).")
    p.add_argument("--show", action="store_true", help="Open operator display window.")
    p.add_argument("--save-overlays", default=None,
                   help="Directory to write overlay PNGs (headless).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    run(
        cfg,
        once=args.once,
        max_iterations=args.iterations,
        show=args.show,
        save_overlays_dir=args.save_overlays,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
