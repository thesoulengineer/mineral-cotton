# 2D Geometric Anomaly Detection — Single Industrial Camera
# 2D-обнаружение геометрических аномалий — одна промышленная камера

Production 2D dimensional inspection for a hydroponic **mineral-wool** part: a
square base with a shallow circular pocket containing a **ring** (outer/inner
diameters) and a center hole — three concentric circular edges in all. One
fixed, top-down GigE Vision camera measures planar geometry and flags parts
outside a learned tolerance envelope.

Промышленный 2D-контроль размеров детали из **минеральной ваты**: квадратное
основание с неглубоким карманом, в котором **кольцо** (наружный/внутренний
диаметры) и центральное отверстие — три концентрических круговых края. Одна
неподвижная камера GigE Vision (вид сверху) измеряет планарную геометрию.

**Nominal part geometry / Номинальная геометрия детали:**

| Feature / Признак | Nominal / Номинал |
|-------------------|-------------------|
| Square base / Квадратное основание | 100 × 100 mm |
| Block height / Высота блока | 60 mm (not measured / не измеряется) |
| Ring outer Ø / Наружный диаметр кольца | 75 mm |
| Ring inner Ø / Внутренний диаметр кольца | 60 mm |
| Center hole Ø / Центральное отверстие | 25 mm |

The **ring** (OD 75 / ID 60 mm) sits near the pocket rim; the 75 mm and 60 mm
are its outer and inner edges. / **Кольцо** (НД 75 / ВД 60 мм) у кромки кармана.

> **Single 2D view only.** Depth (the 60 mm height, pocket/hole depth) is **not**
> measurable from this nadir view. We measure planar features only — diameters,
> concentricity, base centering, roundness, base squareness.
> **Только один 2D-вид.** Глубина (высота 60 мм) **не** измеряется; только
> планарные признаки.

The synthetic demo (`demo.py`) renders these exact nominals at 0.10 mm/pixel, so
measured diameters read ~25 / 60 / 75 mm against known ground truth.
/ Демо рендерит эти номиналы при 0.10 мм/пиксель для проверки по эталону.

---

## 1. Modules / Модули

| File | Role / Роль |
|------|-------------|
| `camera.py` | Acquisition via the Huaray MV Viewer SDK (IMVApi) + hardware-free fallback. / Захват через SDK Huaray MV Viewer + резервный источник. |
| `utils.py` | Calibration, preprocessing, config I/O, geometry helpers. / Калибровка, предобработка, конфиг, геометрия. |
| `inference.py` | Feature extraction + envelope classification. / Извлечение признаков + классификация. |
| `calibrate.py` | One-time calibration + envelope + gauge-R&R CLI. / Калибровка + допуск + повторяемость (CLI). |
| `main.py` | Live loop: capture → inference → verdict + overlay + logging. / Живой цикл контроля. |
| `config.json` | Persisted calibration constants + tolerance envelope. / Константы калибровки + допуск. |
| `demo.py` | End-to-end demonstration on synthetic samples (smoke test). / Сквозная демонстрация. |

---

## 2. Install / Установка

```bash
pip install -r requirements.txt
```

`opencv-python-headless` is fine on servers without a display.
На серверах без дисплея подойдёт `opencv-python-headless`.

### Camera SDK setup / Настройка SDK камеры

The camera is the **ContrasTech Mars2300S-40gc** (GigE, color), driven by the
**Huaray/ContrasTech MV Viewer SDK** (the vendor `IMVApi.py` / `IMVDefines.py`
Python binding + runtime DLLs). It is **not** on PyPI and is not committed here
(large, platform-specific, licensed). On the production station:

Камера **ContrasTech Mars2300S-40gc** (GigE, цветная) управляется **SDK Huaray/
ContrasTech MV Viewer** (`IMVApi.py`/`IMVDefines.py` + DLL). Не в PyPI и не в
репозитории. На рабочей станции:

1. Install the MV Viewer SDK (ships with the camera). / Установите SDK.
2. Point `camera.sdk_path` in `config.json` at the vendor binding folder. The
   default is the station install path `C:/API.RP.1.3.8/APIContrastech`;
   `camera.py` adds it to `sys.path` and imports `IMVApi`.
   / Задайте `camera.sdk_path` (по умолчанию `C:/API.RP.1.3.8/APIContrastech`).
3. `IMVApi.py` loads the SDK DLL by an absolute path near its top. If your
   install location differs, adjust that one line or install where it expects.
   / `IMVApi.py` грузит DLL по абсолютному пути — при необходимости поправьте.

**Pixel format.** For metrology set `camera.pixel_format = "Mono8"` (kept
single-channel). If the color sensor streams Bayer/YUV/packed instead, the
acquisition path converts it to BGR8 via the SDK's `IMV_PixelConvert`
automatically, so any format works. / Для метрологии — `Mono8`; иначе кадр
конвертируется в BGR8 через `IMV_PixelConvert`.

Without the SDK (engineering laptop / CI), `camera.py` prints a notice and
falls back to a synthetic or directory source, so everything below still runs.
/ Без SDK — переход на синтетику/файлы; всё нижеописанное работает.

To use vendor-neutral **Harvester** instead, set `camera.backend` to
`"harvester"`, `pip install harvesters`, and point `camera.cti_file` at the
GenTL producer `.cti`. / Для Harvester: `backend="harvester"`, `cti_file`.

---

## 3. Quick start (no hardware) / Быстрый старт (без оборудования)

```bash
python demo.py
```

This generates synthetic sample images and runs the full
**calibrate → envelope → classify** flow, printing calibration constants, the
held-out false-reject rate, gauge-R&R sigmas, and PASS/FAIL on good vs
defective parts. It is also the project smoke test.

Скрипт создаёт синтетические образцы и выполняет весь цикл
**калибровка → допуск → классификация**, печатая константы калибровки, долю
ложных отбраковок на отложенной выборке, повторяемость и вердикты.

---

## 4. Wiring / Подключение

1. Mount the camera on a fixed nadir (straight-down) bracket over the fixture
   nest; the part sits at a repeatable position. / Камера жёстко закреплена
   строго сверху над гнездом; деталь — в повторяемом положении.
2. Diffuse ring or coaxial illumination, controllable. Set exposure/gain in
   `config.json` (`camera.exposure_us`, `camera.gain_db`). / Диффузное кольцевое
   или коаксиальное освещение; экспозиция/усиление — в `config.json`.
3. Connect the GigE camera to a dedicated NIC (jumbo frames recommended).
   Install the GenTL producer and point `camera.cti_file` at its `.cti`.
   / Подключите камеру к выделенной сетевой карте; укажите `.cti` в конфиге.

**Backend / Бэкенд.** `camera.py` drives the camera via the **Huaray MV Viewer
SDK** (`IMVApi.MvCamera`): enumerate → create handle → open → set
`ExposureTime` / `GainRaw` / `PixelFormat` → start grabbing → `IMV_GetFrame`.
Exposure/gain node names are configurable (`camera.exposure_feature`,
`camera.gain_feature`) since gain naming is vendor-specific. A `"harvester"`
backend is available as an alternative, and without hardware/SDK `Camera` falls
back to a directory of images or a synthetic renderer
(`camera.fallback_source`). / Основной бэкенд — SDK Huaray MV Viewer; есть
Harvester и резервные источники.

---

## 5. Calibration procedure / Процедура калибровки

Print a checkerboard with **9×6 inner corners** and known **square size (mm)**;
set both in `config.json → calibration.checkerboard`. Capture 8–15 views of the
board tilted/translated across the field of view, save them to a folder, then:

Распечатайте шахматную доску **9×6 внутренних углов** с известным **размером
клетки (мм)**; задайте их в `config.json`. Снимите 8–15 видов доски под разными
углами по всему полю и сохраните в папку, затем:

```bash
python calibrate.py calibrate --images path/to/checkerboard_images
```

This runs `cv2.calibrateCamera`, then persists `mm_per_pixel`, the camera
matrix, the distortion coefficients and `image_size` to `config.json`. Every
production frame is undistorted before measurement.

> **Same resolution.** Calibration images **must** be captured at the same
> camera resolution as production parts — one physical camera images both.
> **Одинаковое разрешение** доски и деталей — снимает одна и та же камера.

---

## 6. Envelope-building procedure / Построение допуска

Collect images of **known-good** parts, split into a **train** set and a
separate **held-out** set (different physical parts), then:

Соберите изображения **заведомо годных** деталей, разделите на **обучающую** и
отдельную **отложенную** выборки (разные детали), затем:

```bash
python calibrate.py envelope --train good_train_dir --holdout good_holdout_dir
# or live: python calibrate.py envelope --live 30   (grabs N, splits 70/30)
```

The envelope is `mean ± k·σ` per feature (default `k=3`), computed **only** on
the train set. The **false-reject rate (FRR)** is then measured on the
**held-out** set and printed. This is the critical anti-bias step: computing the
threshold and its FRR on the same parts guarantees a meaningless in-sample
FRR≈0 and hides a threshold that fails to generalize. A held-out **FRR ≥ 0.5**
triggers a warning — the threshold does not generalize; increase `k`, add good
parts, or fix illumination/repeatability.

Допуск `среднее ± k·σ` строится **только** на обучающей выборке; доля ложных
отбраковок (**FRR**) считается на **отложенной** выборке. Это защита от смещения
калибровки: если считать порог и FRR на одних деталях, FRR≈0 бессмысленна.
**FRR ≥ 0.5** — предупреждение: порог не обобщается.

**Spec tolerances (manual overrides) / Допуски по чертежу.** To tolerance a
feature by drawing spec instead of learned good-part spread, set an explicit
bound in `config.json → envelope.manual_bounds`. It takes precedence over the
learned band per side and is applied at classification time (edit + restart,
no re-training needed). The held-out FRR is reported against the effective
bounds. Example — the circle pattern must sit within 2 mm of the block center:

```json
"manual_bounds": { "base_center_offset_mm": { "upper": 2.0 } }
```

/ Явный допуск по чертежу задаётся в `envelope.manual_bounds`; он перекрывает
обученную полосу и применяется при классификации (без переобучения). Пример
выше: центровка рисунка кругов — в пределах 2 мм от центра блока.

**Sidedness / Сторонность.** Tolerances are applied correctly by type:

| Feature / Признак | Bound / Граница |
|-------------------|-----------------|
| diameters / диаметры | two-sided (too small **or** too large) |
| roundness / округлость | lower only (1.0 = perfect) |
| concentricity / соосность | upper only (0 = perfect) |
| base centering / центровка к блоку | upper only (0 = perfect) |
| base squareness / квадратность | upper only (0° dev = perfect) |

A perfectly round or perfectly concentric part is **never** rejected for being
"too good". / Идеальная деталь не отбраковывается за «слишком хорошо».

### Repeatability (gauge R&R) / Повторяемость

Image **one static part** N times and report per-feature measurement σ:

```bash
python calibrate.py repeatability --images repeat_dir     # or --live 20
```

Measurement σ must be **small relative to the tolerance band** (`sigma/band`
and `sigma/margin` are printed). If it is not, the gauge cannot discriminate
good from bad regardless of the envelope. / Шум измерения должен быть мал
относительно полосы допуска, иначе прибор не различает годное/брак.

---

## 7. Live inspection / Живой контроль

```bash
python main.py                       # continuous / непрерывно
python main.py --once                # one part / одна деталь
python main.py --iterations 100      # bounded run / ограниченный прогон
python main.py --show                # operator display window / окно оператора
python main.py --save-overlays out/  # write overlay PNGs (headless)
```

Each cycle: `grab_averaged` (N-frame noise reduction) → undistort → extract
geometry → classify → overlay (detected circles, fitted ellipses, measured
values) → log. / Каждый цикл: усреднение кадров → коррекция → измерение →
классификация → наложение → журнал.

For per-part triggering, wire your part-present signal to call `main.run(cfg,
once=True)` once per part. / Для потактового запуска вызывайте `run(once=True)`
по сигналу присутствия детали.

---

## 8. Interpreting verdicts & logs / Чтение вердиктов и журналов

**Extraction status / Статус измерения** (`inference.extract_geometry`):

| Status | Meaning / Значение |
|--------|--------------------|
| `OK` | Measured successfully. / Измерено успешно. |
| `LOW_CONTRAST` | Illumination out of range — refused to measure. / Плохой свет — измерение отклонено. |
| `NO_CIRCLES` | No circular features found. / Круги не найдены. |
| `WRONG_COUNT` | Not exactly 3 features — fails for manual review. / Не ровно 3 признака. |

Any non-`OK` status maps to a **FAIL** verdict — a part we cannot measure is
never passed. / Любой не-`OK` статус даёт **FAIL**: неизмеримая деталь не
проходит.

**Verdict / Вердикт** (`inference.classify`): `PASS` if every checked feature
is within its envelope, else `FAIL` with a per-feature `violations` list
(`feature`, `value`, `lower`, `upper`, `deviation`). A `None` bound means that
side is unbounded (one-sided tolerance).

**Logs / Журналы** (paths in `config.json → logging`):
- `inspections.jsonl` — one full structured record per inspection (all features,
  roundness, contrast, verdict, violations). / Полная запись на строку.
- `inspections.csv` — flattened, spreadsheet-friendly row per inspection.
  / Плоская строка для таблиц.

Both are append-only for traceability. / Дозапись — для прослеживаемости.

---

## 9. Robustness / Устойчивость

- **Lighting.** Mean/contrast below threshold → `LOW_CONTRAST`, not a bogus
  measurement (inconsistent light is the top cause of Hough instability).
  / Недостаточный свет → `LOW_CONTRAST`.
- **Wrong count.** ≠3 features → `WRONG_COUNT`, part flagged for manual review.
  / ≠3 признаков → `WRONG_COUNT`.
- **Fibrous soft edges.** Coarse `cv2.HoughCircles` only locates the part
  center and confirms features exist; precise geometry comes from **edge
  contour + `cv2.fitEllipse`** (not raw Hough radii), with a **generous σ**
  envelope. / Точность — из `fitEllipse`, а не из радиусов Hough; щедрая σ.
- **Concentric features.** Detection exploits the known concentric structure:
  candidates are filtered by circularity (rejects the square base and stray
  arcs) and proximity to the coarse center, then the double edge of each ring is
  merged into one sub-pixel feature. / Учитывается концентрическая структура.

---

## 10. Configuration reference / Справочник конфигурации

All tunables live in `config.json` — nothing is hard-coded. Key groups:

- `camera` — `backend` (`huaray`|`harvester`), `sdk_path`, `device_index`,
  `pixel_format`, `exposure_us`, `gain`, `exposure_feature`, `gain_feature`,
  `grab_timeout_ms`, `average_frames`, `cti_file` (Harvester),
  `fallback_source` (`synthetic`|`directory`).
- `calibration` — `mm_per_pixel`, `camera_matrix`, `dist_coeffs`, `image_size`,
  `checkerboard {cols, rows, square_size_mm}`.
- `preprocessing` — `median_blur_ksize`, `use_clahe` (off by default; it
  amplifies noise and destabilizes Hough), `clahe_*`.
- `detection` — `hough {dp, min_dist_px, param1, param2, min/max_radius_px}`,
  `expected_circles`, `canny_low/high`, `circularity_min`, `detect_roundness_min`,
  merge/center tolerances.
- `contrast` — `min/max_mean_intensity`, `min_std_intensity`.
- `envelope` — `k_sigma`, `min_sigma_frac_of_mean`, `features`, `manual_bounds`
  (spec tolerance overrides, e.g. `base_center_offset_mm.upper = 2.0`),
  `held_out_false_reject_rate`, `measurement_repeatability`.
- `logging` — `csv_path`, `jsonl_path`.

Все параметры — в `config.json`, ничего не «зашито» в код.

---

## 11. Measured features / Измеряемые признаки

- `center_hole_diameter_mm`, `ring_inner_diameter_mm`, `ring_outer_diameter_mm`
  — from the mean of the fitted ellipse axes × `mm_per_pixel`. (The ring's inner
  and outer edges are 60 mm and 75 mm; the hole is 25 mm.)
- `concentricity_mm` — max deviation of the three fitted circle centers from
  their mean center: are the circles concentric **to each other**?
  / Соосность кругов между собой.
- `base_center_offset_mm` — distance from the ring center (the two largest
  edges) to the square-block center: is the circle pattern **centered on the
  block**? This is distinct from concentricity — the ring and hole can be
  concentric with each other yet sit off-center on the block. Anchoring to the
  ring center keeps an eccentric hole from masquerading as an off-center pattern.
  Toleranced by spec at **2 mm** (`envelope.manual_bounds`), not learned.
  / Центровка рисунка кругов относительно блока (отдельно от соосности);
    допуск по чертежу **2 мм**.
- `<feature>_roundness` — minor/major ellipse-axis ratio (1.0 = perfect;
  proxy for tilt/out-of-round). / Отношение осей эллипса.
- `base_squareness_deg` (optional) — max corner-angle deviation of the square
  base from 90°. / Отклонение углов основания от 90°.
