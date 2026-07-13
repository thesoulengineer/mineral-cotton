"""camera.py — industrial-camera acquisition wrapper.

Primary backend / Основной бэкенд:
  The plant camera is driven through the **Huaray MV Viewer SDK** (the vendor
  ``MvCamera`` / ``IMVApi`` Python binding shipped with the camera). This is the
  ``IMV_*`` GenICam API: enumerate -> create handle -> open -> set features
  (ExposureTime / Gain / PixelFormat) -> start grabbing -> IMV_GetFrame. The SDK
  works for the vendor's GigE, USB3, CameraLink and CoaXPress cameras.
  / Основной бэкенд — SDK Huaray MV Viewer (IMVApi / MvCamera).

An optional ``harvester`` backend (vendor-neutral GenICam) is kept for sites
that standardised on it. When no camera or SDK is present (engineering laptop or
CI), the wrapper transparently falls back to a directory of images or a
synthetic part renderer so the full pipeline stays runnable and demonstrable.
/ Дополнительно — Harvester; без камеры/SDK — переход на файлы/синтетику.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Synthetic part renderer / Синтетический рендер детали
# --------------------------------------------------------------------------- #
# Nominal part geometry (real drawing) at the demo scale of 0.10 mm/pixel:
#   square base 100 mm; a ring of OD 75 mm / ID 60 mm; center hole 25 mm.
# The three concentric circular EDGES the vision system measures are the ring's
# outer edge (75 mm), the ring's inner edge (60 mm) and the hole (25 mm).
# / Геометрия: основание 100 мм; кольцо НД 75 / ВД 60 мм; отверстие 25 мм.
def render_synthetic_part(
    image_size: tuple = (1200, 1200),
    ring_od_px: float = 750.0,     # 75 mm ring outer diameter
    ring_id_px: float = 600.0,     # 60 mm ring inner diameter
    hole_d_px: float = 250.0,      # 25 mm center hole
    base_side_px: float = 1000.0,  # 100 mm square base
    center: Optional[tuple] = None,
    center_jitter_px: tuple = (0.0, 0.0),
    hole_offset_px: tuple = (0.0, 0.0),
    ellipticity: float = 1.0,
    noise_sigma: float = 2.0,
    brightness: float = 200.0,
    edge_softness: float = 1.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Render a mono frame resembling the mineral-wool part (see CAD).

    Square base with a shallow circular pocket; near the pocket rim an annular
    ring (OD 75 mm, ID 60 mm) reads as a darker band, and a center hole sits in
    the middle. Rendered as it would look under diffuse/coaxial metrology
    lighting: distinct but slightly soft (Gaussian) edges with sensor noise,
    rather than a razor-sharp CAD image. Parameters inject defects for testing:
    ``ellipticity`` (out-of-round), ``hole_offset_px`` (concentricity), and the
    feature diameters (size).
    / Основание с неглубоким карманом; кольцевая канавка (НД 75 / ВД 60 мм) —
      тёмная полоса; центральное отверстие. Параметры вносят дефекты.
    """
    rng = np.random.default_rng(seed)
    h, w = image_size
    img = np.full((h, w), 30.0, dtype=np.float32)  # dark background / тёмный фон

    cx, cy = (w / 2.0, h / 2.0) if center is None else center
    cx += center_jitter_px[0]
    cy += center_jitter_px[1]

    # Square base (bright plate) / квадратное основание (светлая плита).
    half = int(round(base_side_px / 2.0))
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

    # Concentric shades giving three unambiguous edges at OD / ID / hole.
    # / Концентрические уровни: три чётких края на НД / ВД / отверстии.
    # Ring band (OD..ID): a darker annulus near the pocket rim -> edge at 75 mm.
    # / Кольцевая полоса (НД..ВД): тёмное кольцо -> край на 75 мм.
    cv2.ellipse(img, (int(cx), int(cy)), _axes(ring_od_px), 0, 0, 360,
                115.0, thickness=-1)
    # Pocket floor inside the ring -> edge at 60 mm (ring inner diameter).
    # / Дно кармана внутри кольца -> край на 60 мм (внутренний диаметр кольца).
    cv2.ellipse(img, (int(cx), int(cy)), _axes(ring_id_px), 0, 0, 360,
                175.0, thickness=-1)
    # Center hole (dark), optionally offset for a concentricity defect.
    # / Центральное отверстие (тёмное), со смещением для дефекта соосности.
    cv2.ellipse(img, (int(cx + hole_offset_px[0]), int(cy + hole_offset_px[1])),
                _axes(hole_d_px), 0, 0, 360, 25.0, thickness=-1)

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


# --------------------------------------------------------------------------- #
# Huaray MV Viewer SDK helpers / Помощники SDK Huaray MV Viewer
# --------------------------------------------------------------------------- #
def _imv_frame_to_ndarray(frame: Any, defs: Any) -> np.ndarray:
    """Copy an IMV_Frame into an owned NumPy array (mono or BGR).

    The SDK buffer is only valid until IMV_ReleaseFrame, so we always copy. We
    handle Mono8, RGB8/BGR8 and 8-bit Bayer directly; higher bit depths raise a
    clear error (set PixelFormat=Mono8, or extend via IMV_PixelConvert).
    / Копирует кадр IMV_Frame в собственный массив NumPy; буфер SDK живёт только
      до IMV_ReleaseFrame. Mono8/RGB8/BGR8/Bayer8 — напрямую; иначе явная ошибка.
    """
    info = frame.frameInfo
    w, h, pf = int(info.width), int(info.height), int(info.pixelFormat)
    raw = np.ctypeslib.as_array(frame.pData, shape=(int(info.size),)).copy()

    if pf == defs.IMV_EPixelType.gvspPixelMono8:
        return raw[: w * h].reshape(h, w)
    if pf == defs.IMV_EPixelType.gvspPixelBGR8:
        return raw[: w * h * 3].reshape(h, w, 3)
    if pf == defs.IMV_EPixelType.gvspPixelRGB8:
        return cv2.cvtColor(raw[: w * h * 3].reshape(h, w, 3), cv2.COLOR_RGB2BGR)

    # 8-bit Bayer -> demosaic to BGR (best effort; station runs Mono8).
    # / 8-битный Bayer -> демозаик в BGR (по возможности).
    bayer_codes = {
        defs.IMV_EPixelType.gvspPixelBayGR8: cv2.COLOR_BayerGR2BGR,
        defs.IMV_EPixelType.gvspPixelBayRG8: cv2.COLOR_BayerRG2BGR,
        defs.IMV_EPixelType.gvspPixelBayGB8: cv2.COLOR_BayerGB2BGR,
        defs.IMV_EPixelType.gvspPixelBayBG8: cv2.COLOR_BayerBG2BGR,
    }
    if pf in bayer_codes:
        return cv2.cvtColor(raw[: w * h].reshape(h, w), bayer_codes[pf])

    raise RuntimeError(
        f"Unsupported IMV pixelFormat 0x{pf:08X}. Set camera PixelFormat to "
        "'Mono8' (recommended for metrology) or extend _imv_frame_to_ndarray."
    )


class _HuaraySource(_AcquisitionSource):
    """Acquisition through the Huaray MV Viewer SDK (``IMVApi.MvCamera``).

    / Захват через SDK Huaray MV Viewer (IMVApi.MvCamera).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self._cfg = cfg
        self._cam = None       # IMVApi.MvCamera
        self._defs = None      # IMVDefines module
        self._api = None       # IMVApi module

    def _check(self, code: int, what: str) -> None:
        """Raise with the IMV return code on failure. / Ошибка по коду возврата."""
        if code != self._defs.IMV_OK:
            raise RuntimeError(f"{what} failed (IMV code {code}).")

    def open(self) -> None:
        # The vendor binding lives beside the SDK; add it to the path and import
        # lazily so this module loads without the SDK present.
        # / Ленивый импорт биндинга вендора из sdk_path.
        sdk_path = self._cfg.get("sdk_path", "CameraSDK")
        if sdk_path and sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
        import IMVApi  # type: ignore
        import IMVDefines  # type: ignore
        self._api, self._defs = IMVApi, IMVDefines

        # 1) Enumerate devices across all interfaces. / Перечисление устройств.
        dev_list = IMVDefines.IMV_DeviceList()
        iface_all = IMVDefines.IMV_EInterfaceType.interfaceTypeAll
        self._check(IMVApi.MvCamera.IMV_EnumDevices(dev_list, iface_all),
                    "IMV_EnumDevices")
        if dev_list.nDevNum == 0:
            raise RuntimeError("No Huaray cameras discovered by IMV_EnumDevices.")

        # 2) Create a handle by device index and open the device.
        # / Создание хендла по индексу и открытие устройства.
        from ctypes import c_int, byref  # local; ctypes always available
        index = int(self._cfg.get("device_index", 0))
        self._cam = IMVApi.MvCamera()
        self._check(
            self._cam.IMV_CreateHandle(
                IMVDefines.IMV_ECreateHandleMode.modeByIndex, byref(c_int(index))),
            "IMV_CreateHandle")
        self._check(self._cam.IMV_Open(), "IMV_Open")

        # 3) Pixel format + exposure/gain. / Формат пикселей + экспозиция/усиление.
        pixel_format = self._cfg.get("pixel_format", "Mono8")
        try:
            self._cam.IMV_SetEnumFeatureSymbol("PixelFormat", pixel_format)
        except Exception:  # noqa: BLE001 - some models fix the format
            pass
        self.apply_settings(
            float(self._cfg.get("exposure_us", 8000.0)),
            float(self._cfg.get("gain", 1.0)),
        )

        # 4) Start the stream. / Запуск потока.
        self._check(self._cam.IMV_StartGrabbing(), "IMV_StartGrabbing")

    def apply_settings(self, exposure_us: float, gain: float) -> None:
        if self._cam is None:
            return
        exp_feat = self._cfg.get("exposure_feature", "ExposureTime")
        gain_feat = self._cfg.get("gain_feature", "GainRaw")
        # Set only what the device exposes; ignore unsupported nodes.
        # / Пишем только доступные узлы, недоступные пропускаем.
        for feat, value in ((exp_feat, exposure_us), (gain_feat, gain)):
            try:
                if self._cam.IMV_FeatureIsWriteable(feat):
                    self._cam.IMV_SetDoubleFeatureValue(feat, float(value))
            except Exception:  # noqa: BLE001
                pass

    def read(self) -> np.ndarray:
        from ctypes import byref
        frame = self._defs.IMV_Frame()
        timeout = int(self._cfg.get("grab_timeout_ms", 2000))
        self._check(self._cam.IMV_GetFrame(frame, timeout), "IMV_GetFrame")
        try:
            return _imv_frame_to_ndarray(frame, self._defs)
        finally:
            # Always return the buffer to the SDK pool. / Всегда возвращаем буфер.
            self._cam.IMV_ReleaseFrame(frame)

    def close(self) -> None:
        if self._cam is None:
            return
        try:
            if self._cam.IMV_IsGrabbing():
                self._cam.IMV_StopGrabbing()
            if self._cam.IMV_IsOpen():
                self._cam.IMV_Close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._cam.IMV_DestroyHandle()
        except Exception:  # noqa: BLE001
            pass
        self._cam = None


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
    _HARDWARE_BACKENDS = {
        "huaray": _HuaraySource,
        "harvester": _HarvesterSource,
    }

    def connect(self) -> "Camera":
        """Open the configured backend, falling back on failure.

        Tries the configured hardware backend first; if it raises (no camera, no
        SDK, missing library), it falls back to the configured offline source
        and records which backend is actually in use.
        / Открывает бэкенд; при сбое — переход на офлайн-источник.
        """
        backend = self._cfg.get("backend", "huaray")
        source_cls = self._HARDWARE_BACKENDS.get(backend)
        if source_cls is not None:
            try:
                self._source = source_cls(self._cfg)
                self._source.open()
                self.active_backend = backend
                self._connected = True
                return self
            except Exception as exc:  # noqa: BLE001 - fall back deliberately
                print(
                    f"[camera] Hardware backend '{backend}' unavailable ({exc}); "
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

    def set_exposure_gain(self, exposure_us: float, gain: float) -> None:
        """Update exposure (us) and gain (camera units) at runtime.

        / Меняет экспозицию (мкс) и усиление (в единицах камеры) на лету.
        """
        self._require().apply_settings(exposure_us, gain)

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
