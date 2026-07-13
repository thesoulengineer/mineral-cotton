"""utils.py — calibration, preprocessing, config I/O and geometry helpers.

Shared foundation for the 2D geometric anomaly-detection system.
/ Общая основа для системы 2D-обнаружения геометрических аномалий.

Design notes / Замечания по дизайну:
  * Pure OpenCV + NumPy. No heavyweight ML dependency on the geometric path.
    / Только OpenCV + NumPy. Никаких тяжёлых ML-зависимостей.
  * All tunables live in config.json, never hard-coded in the pipeline.
    / Все настраиваемые параметры хранятся в config.json.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Status codes / Коды состояния
# --------------------------------------------------------------------------- #
# Fail loud with explicit status strings so a bad measurement is never silently
# passed downstream.
# / Явные строковые коды состояния — плохое измерение никогда не проходит молча.
STATUS_OK = "OK"
STATUS_NO_CIRCLES = "NO_CIRCLES"
STATUS_WRONG_COUNT = "WRONG_COUNT"
STATUS_LOW_CONTRAST = "LOW_CONTRAST"

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"

# Canonical feature names, ordered small -> large radius.
# / Канонические имена признаков, по возрастанию радиуса.
FEATURE_NAMES: Tuple[str, str, str] = (
    "center_hole_diameter_mm",
    "inner_pocket_diameter_mm",
    "outer_circle_diameter_mm",
)

# Tolerance sidedness / Сторонность допуска:
#   "two"   — reject too small OR too large (diameters).
#   "lower" — reject only below the bound (roundness: 1.0 is perfect).
#   "upper" — reject only above the bound (form error: 0 is perfect).
# A perfectly-round or perfectly-concentric part must never be rejected for
# being "too good", so form metrics get one-sided bands.
# / Идеально круглая/соосная деталь не должна отбраковываться за «слишком хорошо».
SIDE_TWO = "two"
SIDE_LOWER = "lower"
SIDE_UPPER = "upper"


def feature_sidedness(name: str) -> str:
    """Return the tolerance sidedness for a feature by naming convention.

    / Возвращает сторонность допуска признака по соглашению об именах.
    """
    if name.endswith("_roundness"):
        return SIDE_LOWER
    if name == "concentricity_mm" or name.endswith("_squareness_deg"):
        return SIDE_UPPER
    return SIDE_TWO


# --------------------------------------------------------------------------- #
# Configuration I/O / Ввод-вывод конфигурации
# --------------------------------------------------------------------------- #
def default_config() -> Dict[str, Any]:
    """Return the built-in default configuration.

    Used to seed a fresh config.json and to backfill any keys missing from a
    user-supplied file, so old configs keep working when new keys are added.
    / Значения по умолчанию: заполняют новый config.json и недостающие ключи.
    """
    return {
        "camera": {
            # Acquisition backend. We assume GenICam via the Harvester library
            # (the de-facto standard for GigE Vision in Python). camera.py falls
            # back to a file/synthetic source when no hardware is present.
            # / Бэкенд захвата: GenICam через Harvester; при отсутствии
            #   оборудования — источник из файлов/синтетики.
            "backend": "harvester",
            "cti_file": os.environ.get(
                "GENICAM_GENTL64_PATH", "/opt/genicam/producer.cti"
            ),
            "device_index": 0,
            "pixel_format": "Mono8",
            "exposure_us": 8000.0,
            "gain_db": 0.0,
            "average_frames": 8,
            # Fallback source used when hardware/Harvester is unavailable.
            # / Резервный источник, когда оборудование недоступно.
            "fallback_source": "synthetic",  # "synthetic" | "directory"
            "fallback_directory": "samples",
        },
        "calibration": {
            # Filled in by calibrate.py. null until calibrated.
            # / Заполняется calibrate.py. null до калибровки.
            "mm_per_pixel": None,
            "camera_matrix": None,
            "dist_coeffs": None,
            "image_size": None,  # [width, height]
            "checkerboard": {
                "cols": 9,          # inner corners per row / внутренних углов в ряду
                "rows": 6,          # inner corners per column / в столбце
                "square_size_mm": 5.0,
            },
            "rms_reproj_error_px": None,
        },
        "preprocessing": {
            "median_blur_ksize": 5,
            # CLAHE is off by default: local contrast stretching amplifies sensor
            # noise and destabilises HoughCircles. Enable only if a station's
            # lighting is genuinely uneven across the field.
            # / CLAHE выключен по умолчанию: усиливает шум и дестабилизирует Hough.
            "use_clahe": False,
            "clahe_clip": 2.0,
            "clahe_grid": 8,
        },
        "detection": {
            # HoughCircles is used as the COARSE locator: it establishes the
            # part's common center and confirms circular features are present
            # (NO_CIRCLES gate). Precise per-feature geometry then comes from
            # contour + cv2.fitEllipse, because raw Hough radii are unstable on
            # the soft, fibrous, concentric edges of mineral wool.
            # / Hough — грубый локатор (центр + наличие кругов); точная геометрия
            #   из contour + fitEllipse, т.к. радиусы Hough нестабильны.
            "hough": {
                "dp": 1.2,
                "min_dist_px": 15,
                "param1": 60,
                "param2": 30,
                "min_radius_px": 8,
                "max_radius_px": 250,
            },
            "expected_circles": 3,
            # Edge + contour refinement parameters / параметры уточнения по контуру.
            "canny_low": 40,
            "canny_high": 120,
            "min_contour_points": 40,
            "min_contour_area_px": 300.0,
            # Circularity 4*pi*area/perimeter^2: ~1 for a circle, ~0.785 for a
            # square — rejects the square base outline and partial arcs.
            # / Округлость контура: ~1 круг, ~0.785 квадрат; отсекает основание.
            "circularity_min": 0.80,
            # Ellipse axis-ratio gate that rejects highly elongated stray arcs
            # while still admitting genuinely out-of-round features for grading.
            # / Порог по осям эллипса: отсекает вытянутые дуги, но пропускает
            #   реально овальные признаки для оценки.
            "detect_roundness_min": 0.45,
            # Merge tolerances that collapse the double edge (inner/outer of the
            # dilated Canny line) of each ring into one feature.
            # / Допуски слияния двойного края каждого кольца в один признак.
            "radius_merge_tol_px": 12.0,
            "center_merge_tol_px": 15.0,
            # Max distance (px) a feature center may sit from the Hough coarse
            # center to be accepted — large enough to admit an eccentric hole.
            # / Макс. отклонение центра признака от грубого центра Hough.
            "center_tol_px": 60.0,
        },
        "contrast": {
            "min_mean_intensity": 25.0,
            "max_mean_intensity": 245.0,
            "min_std_intensity": 12.0,
        },
        "envelope": {
            # mean +/- k*sigma tolerance band per feature.
            # / Допуск среднее +/- k*sigma для каждого признака.
            "k_sigma": 3.0,
            # Sigma floor as a fraction of the feature mean, so a nearly-constant
            # metric does not yield an impossibly tight band. / Пол sigma.
            "min_sigma_frac_of_mean": 0.01,
            "features": {},  # name -> {mean, sigma, side, lower, upper, n}
            "held_out_false_reject_rate": None,
            "measurement_repeatability": {},  # name -> sigma (gauge R&R)
        },
        "logging": {
            "csv_path": "inspections.csv",
            "jsonl_path": "inspections.jsonl",
        },
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base``.

    / Рекурсивно накладывает override на копию base.
    """
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str = "config.json") -> Dict[str, Any]:
    """Load config from ``path``, backfilling missing keys with defaults.

    Raises a clear error if the file exists but is not valid JSON — we never
    guess our way past a corrupt configuration.
    / Загружает конфиг, дополняя отсутствующие ключи значениями по умолчанию.
      Явная ошибка при повреждённом JSON.
    """
    defaults = default_config()
    if not os.path.exists(path):
        return defaults
    try:
        with open(path, "r", encoding="utf-8") as fh:
            user_cfg = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config file '{path}' is not valid JSON: {exc}") from exc
    if not isinstance(user_cfg, dict):
        raise ValueError(f"Config file '{path}' must contain a JSON object.")
    return _deep_merge(defaults, user_cfg)


def save_config(config: Dict[str, Any], path: str = "config.json") -> None:
    """Persist ``config`` to ``path`` as pretty-printed JSON.

    / Сохраняет конфиг в файл в виде отформатированного JSON.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)  # atomic swap / атомарная замена


# --------------------------------------------------------------------------- #
# Camera-matrix (de)serialization helpers / Сериализация матрицы камеры
# --------------------------------------------------------------------------- #
def matrix_to_list(mat: Optional[np.ndarray]) -> Optional[List[List[float]]]:
    """Convert a NumPy matrix to nested lists for JSON, or pass through None."""
    if mat is None:
        return None
    return np.asarray(mat, dtype=float).tolist()


def list_to_matrix(data: Optional[Sequence]) -> Optional[np.ndarray]:
    """Convert JSON nested lists back to a float64 NumPy array, or None."""
    if data is None:
        return None
    return np.asarray(data, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Calibration / Калибровка
# --------------------------------------------------------------------------- #
@dataclass
class CalibrationResult:
    """Outcome of a checkerboard calibration run.

    / Результат калибровки по шахматной доске.
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]           # (width, height)
    rms_reproj_error_px: float
    mm_per_pixel: float
    num_views: int


def find_checkerboard_corners(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
    refine: bool = True,
) -> Optional[np.ndarray]:
    """Locate inner checkerboard corners with sub-pixel refinement.

    ``pattern_size`` is (cols, rows) of *inner* corners. Returns an (N,1,2)
    float32 array or None if the board was not found.
    / Находит внутренние углы шахматной доски с субпиксельным уточнением.
    """
    gray = to_gray(image)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if not found:
        return None
    if refine:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def _mean_neighbour_spacing_px(corners: np.ndarray, pattern_size: Tuple[int, int]) -> float:
    """Mean pixel distance between horizontally adjacent checker corners.

    Used to convert the known physical square size into mm-per-pixel.
    / Среднее расстояние в пикселях между соседними углами по горизонтали.
    """
    cols, rows = pattern_size
    grid = corners.reshape(rows, cols, 2)
    # Horizontal neighbours within each row / горизонтальные соседи в ряду.
    diffs = np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=2)
    return float(np.mean(diffs))


def calibrate_camera_from_images(
    images: Sequence[np.ndarray],
    pattern_size: Tuple[int, int],
    square_size_mm: float,
) -> CalibrationResult:
    """Run ``cv2.calibrateCamera`` over checkerboard views.

    Also derives ``mm_per_pixel`` as the physical square size divided by the
    mean pixel spacing of the detected corners, averaged across all valid views.
    / Запускает cv2.calibrateCamera и вычисляет mm_per_pixel.

    Raises ValueError if fewer than three valid views are supplied — calibration
    on one or two views is not trustworthy.
    / Ошибка при менее чем трёх валидных видах.
    """
    cols, rows = pattern_size
    # Object points for one board: (0,0,0),(1,0,0),... scaled to mm.
    # / Опорные 3D-точки доски в мм.
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_mm)

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    spacings_px: List[float] = []
    image_size: Optional[Tuple[int, int]] = None

    for img in images:
        gray = to_gray(img)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (w, h)
        corners = find_checkerboard_corners(img, pattern_size, refine=True)
        if corners is None:
            continue
        obj_points.append(objp)
        img_points.append(corners)
        spacings_px.append(_mean_neighbour_spacing_px(corners, pattern_size))

    if len(obj_points) < 3:
        raise ValueError(
            f"Only {len(obj_points)} valid checkerboard view(s) found; "
            "need at least 3 for a trustworthy calibration."
        )
    assert image_size is not None

    rms, cam_mtx, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    mean_spacing_px = float(np.mean(spacings_px))
    mm_per_pixel = float(square_size_mm) / mean_spacing_px

    return CalibrationResult(
        camera_matrix=cam_mtx,
        dist_coeffs=dist,
        image_size=image_size,
        rms_reproj_error_px=float(rms),
        mm_per_pixel=mm_per_pixel,
        num_views=len(obj_points),
    )


def build_undistort_maps(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute remap tables for fast per-frame undistortion.

    ``image_size`` is (width, height). Uses the ORIGINAL camera matrix as the
    new matrix so the result is pixel-for-pixel identical to ``cv2.undistort``
    (same scale, same size) — this keeps ``mm_per_pixel`` valid and makes the
    live-loop remap path agree with the envelope-building path.
    / Использует исходную матрицу как новую: результат совпадает с cv2.undistort,
      сохраняя mm_per_pixel и согласованность путей.
    """
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix, image_size, cv2.CV_16SC2
    )
    return map1, map2


def undistort(
    image: np.ndarray,
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
    maps: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """Undistort a frame; a no-op pass-through if calibration is absent.

    Passing precomputed ``maps`` avoids rebuilding remap tables every frame.
    / Корректирует дисторсию; без калибровки — возврат без изменений.
    """
    if camera_matrix is None or dist_coeffs is None:
        return image
    if maps is not None:
        return cv2.remap(image, maps[0], maps[1], interpolation=cv2.INTER_LINEAR)
    return cv2.undistort(image, camera_matrix, dist_coeffs)


# --------------------------------------------------------------------------- #
# Preprocessing / Предобработка
# --------------------------------------------------------------------------- #
def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 image, converting from BGR if needed.

    / Возвращает одноканальное uint8-изображение (при нужде из BGR).
    """
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    else:
        raise ValueError(f"Unsupported image shape for grayscale: {image.shape}")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def preprocess(image: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Grayscale -> optional CLAHE -> median blur, per config.

    Median blur tames the soft, fibrous edges of mineral wool without smearing
    them the way a Gaussian would.
    / Оттенки серого -> CLAHE -> медианный фильтр. Медиана щадит волокнистые края.
    """
    gray = to_gray(image)
    pp = cfg.get("preprocessing", {})
    if pp.get("use_clahe", True):
        clahe = cv2.createCLAHE(
            clipLimit=float(pp.get("clahe_clip", 2.0)),
            tileGridSize=(int(pp.get("clahe_grid", 8)), int(pp.get("clahe_grid", 8))),
        )
        gray = clahe.apply(gray)
    ksize = int(pp.get("median_blur_ksize", 5))
    if ksize >= 3:
        if ksize % 2 == 0:
            ksize += 1  # median kernel must be odd / ядро медианы должно быть нечётным
        gray = cv2.medianBlur(gray, ksize)
    return gray


def assess_contrast(gray: np.ndarray, cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, float]]:
    """Judge whether illumination is adequate for a trustworthy measurement.

    Returns (ok, metrics). Inconsistent illumination is the top cause of Hough
    instability, so we gate on it before measuring anything.
    / Оценивает освещённость. Нестабильный свет — главная причина сбоев Hough.
    """
    cc = cfg.get("contrast", {})
    mean_i = float(np.mean(gray))
    std_i = float(np.std(gray))
    ok = (
        mean_i >= float(cc.get("min_mean_intensity", 25.0))
        and mean_i <= float(cc.get("max_mean_intensity", 245.0))
        and std_i >= float(cc.get("min_std_intensity", 12.0))
    )
    return ok, {"mean_intensity": mean_i, "std_intensity": std_i}


# --------------------------------------------------------------------------- #
# Geometry helpers / Геометрические помощники
# --------------------------------------------------------------------------- #
def ellipse_diameter_mm(ellipse: Tuple, mm_per_pixel: float) -> float:
    """Diameter (mm) from a cv2.fitEllipse result, as mean of both axes.

    fitEllipse returns ((cx,cy),(MA,ma),angle) with axis *lengths* (diameters)
    in pixels. Averaging the two axes gives an orientation-independent diameter.
    / Диаметр (мм) как среднее двух осей эллипса.
    """
    (_cx, _cy), (axis_a, axis_b), _angle = ellipse
    mean_axis_px = 0.5 * (float(axis_a) + float(axis_b))
    return mean_axis_px * float(mm_per_pixel)


def ellipse_roundness(ellipse: Tuple) -> float:
    """Roundness = minor/major axis ratio in [0,1]; 1.0 is a perfect circle.

    A proxy for tilt or true out-of-round.
    / Округлость = отношение малой оси к большой; 1.0 — идеальный круг.
    """
    (_cx, _cy), (axis_a, axis_b), _angle = ellipse
    major = max(float(axis_a), float(axis_b))
    minor = min(float(axis_a), float(axis_b))
    if major <= 0:
        return 0.0
    return minor / major


def ellipse_center(ellipse: Tuple) -> Tuple[float, float]:
    """Return the (cx, cy) pixel center of a fitEllipse result."""
    (cx, cy), (_a, _b), _angle = ellipse
    return float(cx), float(cy)


def concentricity_mm(
    centers_px: Sequence[Tuple[float, float]], mm_per_pixel: float
) -> float:
    """Max deviation of feature centers from their mean center, in mm.

    / Максимальное отклонение центров признаков от их среднего центра, в мм.
    """
    if not centers_px:
        return 0.0
    pts = np.asarray(centers_px, dtype=np.float64)
    mean_c = pts.mean(axis=0)
    dists = np.linalg.norm(pts - mean_c, axis=1)
    return float(np.max(dists)) * float(mm_per_pixel)


def base_squareness(
    gray: np.ndarray, mm_per_pixel: float
) -> Optional[float]:
    """Max corner-angle deviation (degrees) of the square base from 90 deg.

    Finds the largest 4-vertex convex contour (the base outline) and measures
    how far each interior corner departs from a right angle. Returns None if a
    convincing quadrilateral is not found.
    / Максимальное отклонение углов квадратного основания от 90 градусов.
    """
    # Binarize and take the largest external contour as the base outline.
    # / Бинаризация; крупнейший внешний контур — основание.
    _thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    base = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(base, True)
    approx = cv2.approxPolyDP(base, 0.02 * peri, True)
    if len(approx) != 4:
        return None
    pts = approx.reshape(4, 2).astype(np.float64)
    max_dev = 0.0
    for i in range(4):
        p_prev = pts[(i - 1) % 4]
        p_cur = pts[i]
        p_next = pts[(i + 1) % 4]
        v1 = p_prev - p_cur
        v2 = p_next - p_cur
        cos_a = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        cos_a = max(-1.0, min(1.0, cos_a))
        angle = np.degrees(np.arccos(cos_a))
        max_dev = max(max_dev, abs(angle - 90.0))
    return float(max_dev)
