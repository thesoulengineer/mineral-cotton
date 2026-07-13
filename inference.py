"""inference.py — geometric feature extraction and envelope classification.

Pipeline / Конвейер:
  undistort -> grayscale+blur -> HoughCircles (coarse) ->
  per-circle edge isolation -> cv2.fitEllipse (sub-pixel refine) ->
  sort by radius -> assign [hole, pocket, outer] -> structured result.
/ Коррекция дисторсии -> серый+фильтр -> Hough -> уточнение эллипсом ->
  сортировка по радиусу -> назначение признаков -> структурированный результат.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils import (
    FEATURE_NAMES,
    STATUS_LOW_CONTRAST,
    STATUS_NO_CIRCLES,
    STATUS_OK,
    STATUS_WRONG_COUNT,
    VERDICT_FAIL,
    VERDICT_PASS,
    analyze_base,
    assess_contrast,
    concentricity_mm,
    list_to_matrix,
    preprocess,
    undistort,
)


# --------------------------------------------------------------------------- #
# Coarse detection / Грубое обнаружение
# --------------------------------------------------------------------------- #
def detect_circles_hough(gray: np.ndarray, cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    """Coarse-locate circular features with cv2.HoughCircles.

    Returns an (N,3) array of [x, y, r] in pixels, or None if nothing was found.
    Used to confirm circular features exist and to estimate the part's common
    center; precise geometry comes from the contour+ellipse stage below.
    / Грубо находит круги (наличие + общий центр); точная геометрия — ниже.
    """
    h = cfg["detection"]["hough"]
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=float(h["dp"]),
        minDist=float(h["min_dist_px"]),
        param1=float(h["param1"]),
        param2=float(h["param2"]),
        minRadius=int(h["min_radius_px"]),
        maxRadius=int(h["max_radius_px"]),
    )
    if circles is None:
        return None
    return np.round(circles[0]).astype(np.float64)


def coarse_center(gray: np.ndarray, cfg: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Robust common center of the concentric features from Hough votes.

    Takes the median of all Hough circle centers — robust to the spurious,
    off-center detections that soft edges produce. Returns None if Hough finds
    nothing at all (mapped to NO_CIRCLES upstream).
    / Робастный общий центр (медиана центров Hough); None если кругов нет.
    """
    circles = detect_circles_hough(gray, cfg)
    if circles is None or len(circles) == 0:
        return None
    return float(np.median(circles[:, 0])), float(np.median(circles[:, 1]))


def _contour_features(
    gray: np.ndarray,
    cfg: Dict[str, Any],
    center: Tuple[float, float],
) -> List[Dict[str, Any]]:
    """Isolate each ring's edge contour and fit it with cv2.fitEllipse.

    Canny edges -> contours -> per-contour ellipse fit, gated by circularity
    (rejects the square base and partial arcs) and by proximity to the Hough
    coarse ``center`` (rejects strays). The dilated Canny line yields a double
    edge per ring, which we merge by (radius, center) into one sub-pixel feature.
    / Изоляция края каждого кольца и подгонка эллипсом; фильтры по округлости и
      близости к грубому центру; двойной край сливается в один признак.
    """
    det = cfg["detection"]
    edges = cv2.Canny(gray, int(det["canny_low"]), int(det["canny_high"]))
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    min_pts = int(det["min_contour_points"])
    min_area = float(det["min_contour_area_px"])
    circ_min = float(det["circularity_min"])
    round_min = float(det["detect_roundness_min"])
    center_tol = float(det["center_tol_px"])
    cx0, cy0 = center

    candidates: List[Dict[str, Any]] = []
    for cnt in contours:
        if len(cnt) < min_pts:
            continue
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area < min_area or peri <= 0:
            continue
        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < circ_min:
            continue  # square base outline or ragged arc / основание или дуга
        ellipse = cv2.fitEllipse(cnt)
        (ex, ey), (axis_a, axis_b), _ang = ellipse
        roundness = min(axis_a, axis_b) / max(axis_a, axis_b) if max(axis_a, axis_b) > 0 else 0.0
        if roundness < round_min:
            continue  # elongated stray / вытянутая дуга
        if np.hypot(ex - cx0, ey - cy0) > center_tol:
            continue  # off-center stray / вне центра
        candidates.append({
            "center_px": (float(ex), float(ey)),
            "radius_px": 0.25 * (float(axis_a) + float(axis_b)),
            "roundness": float(roundness),
            "ellipse": ellipse,
        })

    # Merge the double edge of each ring (close in radius AND center).
    # / Слияние двойного края каждого кольца.
    candidates.sort(key=lambda d: d["radius_px"])
    r_tol = float(det["radius_merge_tol_px"])
    c_tol = float(det["center_merge_tol_px"])
    groups: List[List[Dict[str, Any]]] = []
    for cand in candidates:
        if groups:
            last = groups[-1][-1]
            if (abs(cand["radius_px"] - last["radius_px"]) <= r_tol
                    and np.hypot(cand["center_px"][0] - last["center_px"][0],
                                 cand["center_px"][1] - last["center_px"][1]) <= c_tol):
                groups[-1].append(cand)
                continue
        groups.append([cand])

    features: List[Dict[str, Any]] = []
    for grp in groups:
        # Averaging inner/outer dilated edges recovers the true edge location.
        # / Усреднение внутреннего/внешнего края даёт истинное положение.
        cx = float(np.mean([g["center_px"][0] for g in grp]))
        cy = float(np.mean([g["center_px"][1] for g in grp]))
        radius = float(np.mean([g["radius_px"] for g in grp]))
        roundness = float(np.mean([g["roundness"] for g in grp]))
        # A representative ellipse (mean center, mean axes) for the overlay.
        # / Репрезентативный эллипс для наложения на кадр.
        mean_axis = 2.0 * radius
        overlay_ellipse = ((cx, cy), (mean_axis * roundness, mean_axis), 0.0)
        features.append({
            "center_px": (cx, cy),
            "radius_px": radius,
            "diameter_mm": 2.0 * radius,  # scaled by mm_per_pixel by caller
            "roundness": roundness,
            "ellipse": overlay_ellipse,
        })
    return features


# --------------------------------------------------------------------------- #
# Feature extraction / Извлечение признаков
# --------------------------------------------------------------------------- #
def extract_geometry(
    frame: np.ndarray,
    cfg: Dict[str, Any],
    undistort_maps: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    measure_base: bool = True,
) -> Dict[str, Any]:
    """Extract planar geometry from a single frame, in millimeters.

    Returns a structured dict with a ``status`` field (OK, NO_CIRCLES,
    WRONG_COUNT, LOW_CONTRAST) and, when OK, the measured features. A non-OK
    status carries a human-readable ``message`` and never fabricates numbers.
    / Извлекает планарную геометрию в мм. Всегда есть поле status; при не-OK —
      сообщение и никаких выдуманных чисел.
    """
    cal = cfg.get("calibration", {})
    cam_mtx = list_to_matrix(cal.get("camera_matrix"))
    dist = list_to_matrix(cal.get("dist_coeffs"))
    mm_per_pixel = cal.get("mm_per_pixel")

    # 1) Undistort every production frame before measurement.
    # / Коррекция дисторсии перед любым измерением.
    frame_u = undistort(frame, cam_mtx, dist, undistort_maps)

    # 2) Preprocess and gate on illumination quality.
    # / Предобработка и проверка качества освещения.
    gray = preprocess(frame_u, cfg)
    contrast_ok, contrast_metrics = assess_contrast(gray, cfg)
    if not contrast_ok:
        return {
            "status": STATUS_LOW_CONTRAST,
            "message": (
                "Illumination out of range "
                f"(mean={contrast_metrics['mean_intensity']:.1f}, "
                f"std={contrast_metrics['std_intensity']:.1f}); "
                "refusing to measure."
            ),
            "contrast": contrast_metrics,
        }

    if mm_per_pixel is None:
        return {
            "status": "UNCALIBRATED",
            "message": "config.json has no mm_per_pixel; run calibrate.py first.",
            "contrast": contrast_metrics,
        }

    # 3) Coarse Hough detection: confirm circular features exist and get the
    #    robust common center used to reject off-center strays.
    # / Грубый Hough: наличие кругов + общий центр.
    center = coarse_center(gray, cfg)
    if center is None:
        return {
            "status": STATUS_NO_CIRCLES,
            "message": "HoughCircles found no circular features.",
            "contrast": contrast_metrics,
        }

    # 4) Sub-pixel refinement: isolate each ring's edge contour and fitEllipse.
    # / Уточнение: изоляция края каждого кольца и подгонка эллипсом.
    det = cfg["detection"]
    refined = _contour_features(gray, cfg, center)

    # 5) Enforce exactly the expected feature count.
    # / Требуем ровно ожидаемое число признаков.
    expected = int(det.get("expected_circles", 3))
    if len(refined) != expected:
        return {
            "status": STATUS_WRONG_COUNT,
            "message": (
                f"Expected {expected} circular features, found {len(refined)}; "
                "failing part for manual review."
            ),
            "found_count": len(refined),
            "contrast": contrast_metrics,
        }

    # 6) Sort by radius and assign to [hole, pocket, outer]. Scale to mm.
    # / Сортировка по радиусу и назначение [отверстие, карман, внешний].
    refined.sort(key=lambda d: d["radius_px"])
    centers_px = [d["center_px"] for d in refined]

    features: Dict[str, float] = {}
    roundness: Dict[str, float] = {}
    for name, feat in zip(FEATURE_NAMES, refined):
        features[name] = round(2.0 * feat["radius_px"] * float(mm_per_pixel), 4)
        rname = name.replace("_diameter_mm", "_roundness")
        roundness[rname] = round(float(feat["roundness"]), 4)

    features["concentricity_mm"] = round(concentricity_mm(centers_px, float(mm_per_pixel)), 4)

    result: Dict[str, Any] = {
        "status": STATUS_OK,
        "features": features,
        "roundness": roundness,
        "centers_px": centers_px,
        "ellipses": [d.get("ellipse") for d in refined],
        "contrast": contrast_metrics,
    }

    # 7) Base-relative metrics: squareness AND whether the circle pattern is
    #    centered on the block (distinct from circle-to-circle concentricity).
    # / Метрики относительно основания: квадратность И центровка кругов к блоку.
    if measure_base:
        base = analyze_base(gray)
        if base is not None:
            if base["squareness_deg"] is not None:
                features["base_squareness_deg"] = round(float(base["squareness_deg"]), 4)
            # Offset of the circle pattern from the block center, in mm. We
            # anchor to the RING center (mean of the two largest edges), not all
            # three circles, so an eccentric hole stays a concentricity issue
            # and does not masquerade as an off-center pattern.
            # / Смещение рисунка от центра блока: опора на центр кольца (два
            #   крупнейших края), чтобы эксцентриситет отверстия не влиял.
            ring_center = np.mean(np.asarray(centers_px[-2:], dtype=np.float64), axis=0)
            bx, by = base["center_px"]
            offset_px = float(np.hypot(ring_center[0] - bx, ring_center[1] - by))
            features["base_center_offset_mm"] = round(offset_px * float(mm_per_pixel), 4)

    return result


# --------------------------------------------------------------------------- #
# Envelope classification / Классификация по допуску
# --------------------------------------------------------------------------- #
def classify(features: Dict[str, float], envelope: Dict[str, Any]) -> Dict[str, Any]:
    """Classify measured features against a mean +/- k*sigma envelope.

    Returns {verdict: PASS|FAIL, violations: [...]}. Each violation lists the
    feature, its measured value, and the [lower, upper] bounds it broke.
    Features present in the measurement but absent from the envelope are
    reported (not silently ignored) so the operator knows they were unchecked.

    Explicit ``envelope["manual_bounds"]`` (from the drawing/spec) take
    precedence over the learned band per side, and a feature with only a manual
    bound is still checked.
    / Классифицирует признаки по допуску. Явные допуски manual_bounds имеют
      приоритет над обученной полосой.
    """
    env_feats = envelope.get("features", {})
    manual = envelope.get("manual_bounds", {})
    violations: List[Dict[str, Any]] = []
    unchecked: List[str] = []

    for name, value in features.items():
        bounds = env_feats.get(name)
        override = manual.get(name)
        if bounds is None and override is None:
            unchecked.append(name)
            continue
        # Start from the learned band, then let a spec bound override each side.
        # / Берём обученную полосу, затем допуск по чертежу перекрывает сторону.
        lower = bounds.get("lower") if bounds else None
        upper = bounds.get("upper") if bounds else None
        if override is not None:
            if "lower" in override:
                lower = override["lower"]
            if "upper" in override:
                upper = override["upper"]
        v = float(value)
        below = lower is not None and v < float(lower)
        above = upper is not None and v > float(upper)
        if below or above:
            broken = float(lower) if below else float(upper)
            violations.append({
                "feature": name,
                "value": round(v, 4),
                "lower": round(float(lower), 4) if lower is not None else None,
                "upper": round(float(upper), 4) if upper is not None else None,
                "deviation": round(v - broken, 4),
            })

    verdict = VERDICT_PASS if not violations else VERDICT_FAIL
    return {
        "verdict": verdict,
        "violations": violations,
        "unchecked_features": unchecked,
    }


def inspect(
    frame: np.ndarray,
    cfg: Dict[str, Any],
    undistort_maps: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Full single-frame inspection: extract geometry then classify.

    A non-OK extraction status maps to a FAIL verdict for manual review — a
    part we could not measure is never passed.
    / Полная проверка кадра: извлечение + классификация. Неизмеримая деталь
      получает FAIL для ручного разбора.
    """
    geo = extract_geometry(frame, cfg, undistort_maps)
    if geo.get("status") != STATUS_OK:
        return {
            **geo,
            "verdict": VERDICT_FAIL,
            "violations": [{
                "feature": "_extraction",
                "reason": geo.get("status"),
                "message": geo.get("message", ""),
            }],
        }
    # Classify diameters, concentricity, squareness AND roundness together.
    # / Классифицируем диаметры, соосность, квадратность И округлость вместе.
    measured = dict(geo["features"])
    measured.update(geo.get("roundness", {}))
    classification = classify(measured, cfg.get("envelope", {}))
    return {**geo, **classification}
