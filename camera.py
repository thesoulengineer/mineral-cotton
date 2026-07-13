"""camera.py — GigE Vision acquisition wrapper.

Assumption / Допущение:
  The plant camera is a GigE Vision device driven through GenICam via the
  Harvester library (``harvesters`` on PyPI) — the de-facto standard, vendor-
  neutral Python backend for GigE Vision. If your line already standardised on
  Aravis or a vendor SDK, swap the body of ``_HarvesterSource`` only; the public
  interface below is what the rest of the system depends on.
  / Камера GigE Vision через GenICam/Harvester. Для Aravis/SDK замените только
    _HarvesterSource — публичный интерфейс остаётся прежним.

When no camera or Harvester install is present (e.g. an engineering laptop or
CI), the wrapper transparently falls back to a directory of images or a
synthetic part renderer so the full pipeline stays runnable and demonstrable.
/ Без камеры/Harvester — прозрачный переход на источник из файлов или синтетику.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Synthetic part renderer / Синтетический рендер детали
# --------------------------------------------------------------------------- #
def render_synthetic_part(
    image_size: tuple = (720, 720),
    outer_d_px: float = 380.0,
    pocket_d_px: float = 240.0,
    hole_d_px: float = 90.0,
    center: Optional[tuple] = None,
    center_jitter_px: tuple = (0.0, 0.0),
    hole_offset_px: tuple = (0.0, 0.0),
    ellipticity: float = 1.0,
    noise_sigma: float = 2.0,
    brightness: float = 200.0,
    edge_softness: float = 1.2,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Render a mono frame resembling the mineral-wool part.

    Square base with three concentric circular features, representative of the
    part under diffuse/coaxial metrology lighting: high-contrast ring edges that
    are still slightly soft (Gaussian) with additive sensor noise, rather than a
    razor-sharp CAD render. Parameters inject defects for testing: ``ellipticity``
    (out-of-round), ``hole_offset_px`` (concentricity), feature diameters (size).
    / Рендер моно-кадра при метрологическом освещении: контрастные, слегка мягкие
      края с шумом. Параметры вносят дефекты (овальность, эксцентриситет, размер).
    """
    rng = np.random.default_rng(seed)
    h, w = image_size
    img = np.full((h, w), 30.0, dtype=np.float32)  # dark background / тёмный фон

    cx, cy = (w / 2.0, h / 2.0) if center is None else center
    cx += center_jitter_px[0]
    cy += center_jitter_px[1]

    # Square base (bright plate) / квадратное основание (светлая плита).
    half = int(min(h, w) * 0.46)
    cv2.rectangle(
        img,
        (int(w / 2 - half), int(h / 2 - half)),
        (int(w / 2 + half), int(h / 2 + half)),
        float(brightness),
        thickness=-1,
    )

    def _axes(diameter_px: float) -> tuple:
        r = diameter_px / 2.0
        return (int(round(r)), int(round(r * ellipticity)))

    # Nested rings with strong inter-feature contrast so edges are unambiguous.
    # / Вложенные кольца с сильным контрастом между признаками.
    # Outer recessed circle (mid-dark) / внешний утопленный круг.
    cv2.ellipse(img, (int(cx), int(cy)), _axes(outer_d_px), 0, 0, 360,
                110.0, thickness=-1)
    # Inner shallow pocket (mid-bright) / внутренний неглубокий карман.
    cv2.ellipse(img, (int(cx), int(cy)), _axes(pocket_d_px), 0, 0, 360,
                165.0, thickness=-1)
    # Center hole (dark), optionally offset for a concentricity defect.
    # / Центральное отверстие (тёмное), со смещением для дефекта соосности.
    cv2.ellipse(img, (int(cx + hole_offset_px[0]), int(cy + hole_offset_px[1])),
                _axes(hole_d_px), 0, 0, 360, 20.0, thickness=-1)

    # Soften edges + sensor noise / смягчение краёв и шум сенсора.
    if edge_softness > 0:
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=float(edge_softness))
    img += rng.normal(0.0, noise_sigma, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Acquisition sources / Источники захвата
# --------------------------------------------------------------------------- #
class _AcquisitionSource:
    """Backend interface shared by hardware and fallback sources.

    / Общий интерфейс для аппаратного и резервного источников.
    """

    def open(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def read(self) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def apply_settings(self, exposure_us: float, gain_db: float) -> None:
        """Best-effort exposure/gain application; fallbacks ignore it."""
        return None


class _HarvesterSource(_AcquisitionSource):
    """GigE Vision acquisition through the Harvester / GenICam stack.

    / Захват GigE Vision через Harvester/GenICam.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        self._h = None            # harvesters.core.Harvester
        self._ia = None           # image acquirer / получатель изображений

    def open(self) -> None:
        # Imported lazily so the module loads without harvesters installed.
        # / Ленивая загрузка, чтобы модуль работал без harvesters.
        from harvesters.core import Harvester  # type: ignore

        cti = self._cfg["cti_file"]
        if not os.path.exists(cti):
            raise FileNotFoundError(
                f"GenTL producer .cti not found: {cti}. Set camera.cti_file "
                "or GENICAM_GENTL64_PATH."
            )
        self._h = Harvester()
        self._h.add_file(cti)
        self._h.update()
        if not self._h.device_info_list:
            raise RuntimeError("No GigE Vision devices discovered by Harvester.")
        self._ia = self._h.create(self._cfg.get("device_index", 0))
        node_map = self._ia.remote_device.node_map
        try:
            node_map.PixelFormat.value = self._cfg.get("pixel_format", "Mono8")
        except Exception:  # noqa: BLE001 - node naming varies by vendor
            pass
        self.apply_settings(
            float(self._cfg.get("exposure_us", 8000.0)),
            float(self._cfg.get("gain_db", 0.0)),
        )
        self._ia.start()

    def apply_settings(self, exposure_us: float, gain_db: float) -> None:
        if self._ia is None:
            return
        node_map = self._ia.remote_device.node_map
        # Node names differ across vendors; try common GenICam SFNC names.
        # / Имена узлов различаются у вендоров; пробуем стандартные SFNC.
        for name, value in (("ExposureTime", exposure_us), ("Gain", gain_db)):
            try:
                getattr(node_map, name).value = value
            except Exception:  # noqa: BLE001
                pass

    def read(self) -> np.ndarray:
        assert self._ia is not None, "Harvester source not opened."
        with self._ia.fetch(timeout=5.0) as buffer:
            comp = buffer.payload.components[0]
            frame = comp.data.reshape(comp.height, comp.width).copy()
        return frame

    def close(self) -> None:
        if self._ia is not None:
            try:
                self._ia.stop()
            except Exception:  # noqa: BLE001
                pass
            self._ia.destroy()
            self._ia = None
        if self._h is not None:
            self._h.reset()
            self._h = None


class _DirectorySource(_AcquisitionSource):
    """Replay frames from a directory of images (offline / testing).

    / Воспроизведение кадров из каталога изображений (офлайн/тесты).
    """

    def __init__(self, directory: str) -> None:
        self._directory = directory
        self._paths: List[str] = []
        self._idx = 0

    def open(self) -> None:
        patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        paths: List[str] = []
        for pat in patterns:
            paths.extend(glob.glob(os.path.join(self._directory, pat)))
        self._paths = sorted(paths)
        if not self._paths:
            raise FileNotFoundError(
                f"No images found in fallback directory '{self._directory}'."
            )

    def read(self) -> np.ndarray:
        path = self._paths[self._idx % len(self._paths)]
        self._idx += 1
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to read image: {path}")
        return img

    def close(self) -> None:
        self._paths = []


class _SyntheticSource(_AcquisitionSource):
    """Emit freshly rendered synthetic parts with per-frame sensor noise.

    / Генерирует синтетические детали со свежим шумом на каждом кадре.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        self._n = 0

    def open(self) -> None:
        self._n = 0

    def read(self) -> np.ndarray:
        self._n += 1
        # Fresh noise seed each call so grab_averaged() actually reduces noise.
        # / Новый seed на каждый вызов, чтобы усреднение реально снижало шум.
        return render_synthetic_part(seed=None)

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# Public camera wrapper / Публичная обёртка камеры
# --------------------------------------------------------------------------- #
class Camera:
    """GigE Vision camera wrapper with a context-manager interface.

    Usage / Использование::

        with Camera(cfg["camera"]) as cam:
            frame = cam.grab_averaged(8)

    / Обёртка камеры GigE Vision с интерфейсом контекстного менеджера.
    """

    def __init__(self, camera_cfg: Dict[str, Any]) -> None:
        self._cfg = camera_cfg
        self._source: Optional[_AcquisitionSource] = None
        self._connected = False
        self.active_backend: str = "unopened"

    # -- lifecycle / жизненный цикл -- #
    def connect(self) -> "Camera":
        """Open the configured backend, falling back on failure.

        Tries the hardware backend first; if it raises (no camera, no
        harvesters, missing .cti), it falls back to the configured offline
        source and records which backend is actually in use.
        / Открывает бэкенд; при сбое — переход на офлайн-источник.
        """
        backend = self._cfg.get("backend", "harvester")
        if backend == "harvester":
            try:
                self._source = _HarvesterSource(self._cfg)
                self._source.open()
                self.active_backend = "harvester"
                self._connected = True
                return self
            except Exception as exc:  # noqa: BLE001 - fall back deliberately
                print(
                    f"[camera] Hardware backend unavailable ({exc}); "
                    f"falling back to '{self._cfg.get('fallback_source')}'."
                )
        self._connect_fallback()
        return self

    def _connect_fallback(self) -> None:
        fb = self._cfg.get("fallback_source", "synthetic")
        if fb == "directory":
            self._source = _DirectorySource(self._cfg.get("fallback_directory", "samples"))
            self.active_backend = "directory"
        else:
            self._source = _SyntheticSource(self._cfg)
            self.active_backend = "synthetic"
        self._source.open()
        self._connected = True

    def disconnect(self) -> None:
        """Close the backend and release resources. / Закрывает бэкенд."""
        if self._source is not None:
            self._source.close()
            self._source = None
        self._connected = False

    # -- context manager / контекстный менеджер -- #
    def __enter__(self) -> "Camera":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    # -- acquisition / захват -- #
    def _require(self) -> _AcquisitionSource:
        if not self._connected or self._source is None:
            raise RuntimeError("Camera is not connected; call connect() first.")
        return self._source

    def set_exposure_gain(self, exposure_us: float, gain_db: float) -> None:
        """Update exposure (us) and gain (dB) at runtime. / Меняет экспозицию/усиление."""
        self._require().apply_settings(exposure_us, gain_db)

    def grab_frame(self) -> np.ndarray:
        """Grab a single frame as an ndarray (mono or BGR). / Один кадр."""
        return self._require().read()

    def grab_averaged(self, n: Optional[int] = None) -> np.ndarray:
        """Average N frames to cut sensor noise before measurement.

        Averaging is done in float to avoid uint8 rounding, then cast back.
        / Усредняет N кадров для снижения шума сенсора (в float, затем uint8).
        """
        src = self._require()
        count = int(n if n is not None else self._cfg.get("average_frames", 8))
        count = max(1, count)
        acc: Optional[np.ndarray] = None
        for _ in range(count):
            frame = src.read().astype(np.float32)
            acc = frame if acc is None else acc + frame
        assert acc is not None
        averaged = acc / count
        return np.clip(averaged, 0, 255).astype(np.uint8)
