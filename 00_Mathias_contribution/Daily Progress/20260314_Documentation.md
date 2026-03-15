# Documentation – Saturday, March 14, 2026

**Goal:** Prepare the full pipeline for a first Galamsey detection test in the Bono region using Prithvi-EO v2.

---

## Progress Today

| Step | Status | Notes |
|------|--------|--------|
| 1. Prepare dataset (`01_prepare_dataset.py`) | Done | 4,270/4,270 patches copied → 2,983 training + 1,287 validation |
| 2. Upload `GhanaMiningPrithvi` to Google Drive | Done | — |
| 3. Extract Bono test patches (`02_extract_bono_test_patches.py`) | Done | 16 patches in `data/patches_bono_test/`, center 8.054635, -2.025502, UTM E=607379.5 N=890465.8 |
| 4. Train on Colab (`Colab Notebook/BA_Thesis_Model_Training_SmallMinesDS_data.ipynb`) | Done | Training durchgeführt; bester Checkpoint auf Drive in `checkpoints/prithvi-v2-300/` |
| 5. Download checkpoint → run inference (`04_inference_bono.py`) | Pending / optional lokal | Checkpoint in `models/` ablegen; lokal Conda/venv mit gefilterten requirements (ohne bitsandbytes/NVIDIA/triton) oder Inference in Colab |

**Environment:** macOS — use `python3` (not `python`) for all scripts.

---

## Context

The SmallMinesDS dataset (4,270 labeled patches, SW Ghana, 2016 + 2022) is available locally. The Bono mosaic (22 GB, January 2025) is already merged. The plan is:

1. Prepare SmallMinesDS for training → run on Colab → get checkpoint
2. Extract 5×5 km Bono test area around a known Galamsey site
3. Run inference → first prediction map

---

## Files Created

- **Scripts:** `00_Mathias_contribution/scripts/` — all new scripts are independent from the original repo (no files modified).
- **Colab training:** `00_Mathias_contribution/Colab Notebook/BA_Thesis_Model_Training_SmallMinesDS_data.ipynb` — the main notebook used for training on Google Colab.

---

### `01_prepare_dataset.py`

**Purpose:** Copies and renames SmallMinesDS files into the flat `training/` and `validation/` folder structure expected by the training script.

**The problem it solves:**
The HuggingFace data uses naming like `IMG_GH_1755_2022.tif` / `MASK_GH_1755_2022.tif`. The training script expects files ending in `*_IMG.tif` and `*_MASK.tif` (suffix-based grep). Without renaming, the training script finds no files.

**What it does:**
- Reads `train_test_splits_2022.csv` and `train_test_splits_2016.csv`
- For each patch: copies IMG and MASK to `data/GhanaMiningPrithvi/training/` or `.../validation/` based on the `split` column
- Renames: `IMG_GH_1755_2022.tif` → `GH_1755_2022_IMG.tif`

**Output:**
```
data/GhanaMiningPrithvi/
    training/    ← ~2,983 patch pairs (train split from both years)
    validation/  ← ~1,287 patch pairs (test split from both years)
```

**Usage (run from repo root):**
```bash
cd /Users/mathias/dev/BA_Thesis_SmallMinesDS
python3 00_Mathias_contribution/scripts/01_prepare_dataset.py
```

---

### `02_extract_bono_test_patches.py`

**Purpose:** Extracts a 5×5 km area from the merged Bono mosaic and cuts it into 128×128 patches ready for inference.

**Test area:**
- Center: lat=`8.054635`, lon=`-2.025502` (known Galamsey site in Bono region)
- Area: 5,000 × 5,000 m = 500 × 500 pixels at 10 m/px
- Padded to 512 × 512 → **16 patches** (4×4 grid)

**Key design decisions:**

| Decision | Reasoning |
|---|---|
| Values × 10,000 | Bono data exported from GEE with `/10000` → values 0–1. Training data is raw DN 0–10,000. Must undo GEE normalization before inference. |
| Only 6 of 13 bands | SmallMinesDS has 13 bands (Sentinel-2 + SAR + DEM). Prithvi model uses 6: Blue, Green, Red, NIR, SWIR1, SWIR2. Bono mosaic already contains exactly these 6 bands in the right order. |
| Georeferenced output | Each patch saved with correct UTM coordinates, enabling QGIS visualization and reassembly |
| patch_index.csv | Records row/col position of each patch for reassembly after inference |

**Output:**
```
data/patches_bono_test/
    patch_0000_r0_c0.tif   ← 6-band GeoTIFF, 128×128 px, raw DN scale
    patch_0001_r0_c1.tif
    ... (16 patches total)
    patch_index.csv         ← grid position + UTM bounds per patch
```

**Usage (run from repo root):**
```bash
cd /Users/mathias/dev/BA_Thesis_SmallMinesDS
pip3 install pyproj
python3 00_Mathias_contribution/scripts/02_extract_bono_test_patches.py
```

**Dependency:** `pyproj` (for coordinate conversion).

**Example output:** Test area center (UTM 30N) E=607379.5, N=890465.8; bounding box (604880, 887966) → (609880, 892966); 500×500 px read, padded to 512×512; value range after ×10,000 rescaling ~240–4663; 16 patches saved to `data/patches_bono_test/`.

**Viewing the 5×5 km area in QGIS:** The full area is the 16 patch files together. To get a **true-color (Google-Maps-like)** view: Layer Properties → Symbology → Render type **Multiband color** → set **Red band = 3**, **Green band = 2**, **Blue band = 1** (bands 1–3 are Blue, Green, Red). Use min/max stretch for better contrast. Apply to each patch layer.

---

### Colab Training Notebook: `Colab Notebook/BA_Thesis_Model_Training_SmallMinesDS_data.ipynb`

**Purpose:** Main notebook for training Prithvi-EO v2 on SmallMinesDS (GhanaMiningPrithvi) on Google Colab with free GPU.

**Notebook structure (run in order):**

| Zelle | Inhalt |
|-------|--------|
| **0** | Prüft `DATASET_PATH` und ob `validation/` existiert (Hinweis: Auf Drive liegt oft `GhanaMiningPrithvi/GhanaMiningPrithvi/` mit `training/` und `validation/` darin). |
| **1** | Drive mounten (`drive.mount('/content/drive')`) → im Browser freigeben. |
| **2** | Pakete installieren: `terratorch==0.99.7`, `segmentation-models-pytorch==0.3.4`, `lightning==2.4.0`, `albumentations==1.4.10`, `rasterio==1.3.11` (einmalig, 1–2 Min). |
| **3** | Daten von Drive nach lokalem Speicher kopieren: `GhanaMiningPrithvi/GhanaMiningPrithvi` → `/content/GhanaMiningPrithvi`. Einmalig 5–10 Min; danach ist Training deutlich schneller (SSD statt Netzwerk). |
| **4** | Training: `DATASET_PATH='/content/GhanaMiningPrithvi'`, Checkpoints auf Drive unter `checkpoints/prithvi-v2-300/`. 2,983 Training- und 1,287 Validation-Patches; Early Stopping (patience=10), ModelCheckpoint, TensorBoard. Laufzeit ca. **3–5 h** auf T4. |

**Wichtige Pfade im Notebook:**
- Daten auf Drive: `/content/drive/MyDrive/GhanaMiningPrithvi` (darin ggf. Unterordner `GhanaMiningPrithvi` mit `training/` und `validation/`).
- Lokale Kopie für Training: `/content/GhanaMiningPrithvi`.
- Checkpoints: `/content/drive/MyDrive/checkpoints/prithvi-v2-300/`.

**Setup in Colab:**
1. **File → Upload notebook** → `00_Mathias_contribution/Colab Notebook/BA_Thesis_Model_Training_SmallMinesDS_data.ipynb`
2. Runtime → Change runtime type → **GPU (z. B. T4)** → Save
3. Zellen 0 → 1 → 2 → 3 → 4 nacheinander ausführen.

**Expected output:** Bester Checkpoint auf Drive in `checkpoints/prithvi-v2-300/`. Von dort herunterladen und lokal als `models/prithvi-v2-300-best.ckpt` (oder erste `.ckpt` in `models/`) für `04_inference_bono.py` ablegen.

**Alternativ:** In `scripts/` liegt weiterhin `03_train_colab.ipynb` (ältere Zellenstruktur); die alte .py-Version ist in `scripts/archive/03_train_colab.py`.

---

### `04_inference_bono.py`

**Purpose:** Loads the trained checkpoint, runs inference on all Bono test patches, and reassembles predictions into a georeferenced prediction map.

**Data flow:**
```
patch_XXXX.tif (6 bands, 128×128, raw DN)
     ↓  normalize: (value − mean) / std
     ↓  forward pass: Prithvi backbone + UperNet decoder
     ↓  softmax → Mining probability per pixel
     ↓  threshold at 0.5 → binary mask
reassemble 16 patches → full 512×512 px prediction map
```

**Normalization applied:**
```
normalized = (raw_value − mean) / std
means = [1473.81, 1703.35, 1696.68, 3832.40, 3156.11, 2226.07]
stds  = [  223.44,  285.54,  413.82,  389.61,  451.50,  468.27]
```

**Outputs:**

| File | Content |
|---|---|
| `prediction_prob.tif` | Float32 GeoTIFF, 0.0–1.0 Mining probability per pixel |
| `prediction_binary.tif` | Uint8 GeoTIFF, 0=Non-Mining / 1=Mining |
| `prediction_visualization.png` | 3-panel figure: false-color + heatmap + binary mask |

**Usage (run from repo root; requires checkpoint at `models/prithvi-v2-300-best.ckpt`):**
```bash
cd /Users/mathias/dev/BA_Thesis_SmallMinesDS
python3 00_Mathias_contribution/scripts/04_inference_bono.py
```

**Can also run on Colab** (GPU speeds up inference significantly, though 16 patches is manageable on CPU too).

---

### `05_inference_bono_full.py` (gesamte Bono-Region)

**Zweck:** Inferenz auf dem **gesamten** Bono-Mosaik (`Bono_Merged_2025.tif`), ohne vorher Patches zu extrahieren. Liest das Raster fensterweise (128×128), führt das Modell aus und schreibt die Vorhersagen in zwei GeoTIFFs.

**Ausgabe:** `data/inference_bono_full/prediction_prob.tif`, `prediction_binary.tif` (gleiche Bedeutung wie bei der Testregion).

**Laufzeit:** Abhängig von der Mosaik-Größe. Bei z.B. 20.000×20.000 px → ~24.400 Patches; auf CPU grob 2–5 s/Patch → **ca. 14–34 h**. Deutlich schneller mit GPU (Colab oder lokales CUDA/MPS).

**Schnelltest:** Im Skript `LIMIT_PATCHES = 100` setzen, dann werden nur 100 Patches verarbeitet (ein paar Minuten).

**Aufruf (Conda-Umgebung smallmines):**
```bash
conda activate smallmines
python 00_Mathias_contribution/scripts/05_inference_bono_full.py
```

---

### `06_ghana_map_galamsey_bono.py` (Karte Ghana + Galamsey Bono)

**Zweck:** Erzeugt eine Karte von Ghana mit weißem Hintergrund: Landesgrenzen, Regionsgrenzen (Ashanti, Bono, …) und in **Rot** die vom Modell vorhergesagten Galamsey-Flächen in der Bono-Region.

**Eingabe:** Nutzt automatisch `prediction_binary.tif` aus der vollen Bono-Inferenz (`data/inference_bono_full/`) oder, falls nicht vorhanden, aus der Testregion (`data/patches_bono_test/`). Grenzdaten: GADM Level 1 für Ghana (einmaliger Download nach `data/cache/`).

**Ausgabe:** `data/ghana_map_galamsey_bono.png`

**Abhängigkeit:** `geopandas` (ggf. `pip install geopandas`).

**Aufruf:**
```bash
python 00_Mathias_contribution/scripts/06_ghana_map_galamsey_bono.py
```

---

## Pipeline Overview

```
SmallMinesDS (local)           Bono_Merged_2025.tif (local)
      │                                    │
      ▼                                    ▼
01_prepare_dataset.py          02_extract_bono_test_patches.py
      │                                    │
      ▼                                    ▼
data/GhanaMiningPrithvi/       data/patches_bono_test/
  training/ + validation/        16 × 128×128 GeoTIFFs
      │
      ▼
Colab Notebook/BA_Thesis_Model_Training_SmallMinesDS_data.ipynb  (Google Colab — ~3–5 h)
      │
      ▼
models/prithvi-v2-300-best.ckpt  (download from Drive)
      │
      ▼ (combined with patches)
04_inference_bono.py  (Testregion 5×5 km)
      │
      ▼
prediction_* in data/patches_bono_test/

Optional: Gesamte Bono-Region
      │
      ▼
05_inference_bono_full.py  (volles Mosaik, fensterweise)
      │
      ▼
prediction_* in data/inference_bono_full/
```

---

## Key Technical Notes

### Why the × 10,000 rescaling matters
GEE's `maskS2clouds()` function divides all values by 10,000 before export:
```javascript
return image.updateMask(mask).divide(10000)
```
SmallMinesDS data has raw DN values (Blue band mean ≈ 1,418). If we feed 0–1 values to a model expecting 0–10,000, the normalization produces wildly wrong z-scores and the model outputs noise. The rescaling is the single most important preprocessing step.

### Why no `val` split in the CSVs
The CSVs only have `train` and `test` — no `val`. The training script reuses the test split as validation (`val_data_root = test_data_root`). This is consistent with the original paper's setup.

### Band order consistency
| Index | Bono mosaic | SmallMinesDS label |
|---|---|---|
| 0 | B2 | BLUE |
| 1 | B3 | GREEN |
| 2 | B4 | RED |
| 3 | B8A | VNIR_5 |
| 4 | B11 | SWIR_1 |
| 5 | B12 | SWIR_2 |
Both datasets use the same 6-band selection and order. No reordering needed.

### Viewing the 5×5 km patches in QGIS (true color)
The 16 patches live in `data/patches_bono_test/`. By default QGIS may show a false-color composite. For a **Google-Maps-like (true color)** view: Layer Properties → Symbology → Render type **Multiband color** → **Red band = 3**, **Green band = 2**, **Blue band = 1**. Set min/max to **Min/max** or **Cumulative count cut** for better contrast (values are 0–10,000).

---

## Next Steps

- [x] Run `01_prepare_dataset.py` locally to prepare the data
- [x] Upload `data/GhanaMiningPrithvi/` to Google Drive
- [x] Run `02_extract_bono_test_patches.py` (16 patches in `data/patches_bono_test/`)
- [x] Train on Colab (`Colab Notebook/BA_Thesis_Model_Training_SmallMinesDS_data.ipynb`) — Checkpoints auf Drive
- [ ] Download checkpoint from Drive to `models/` (z. B. `prithvi-v2-300-best.ckpt`)
- [ ] Run `04_inference_bono.py` (lokal mit Conda oder in Colab) → first prediction map
- [ ] Open `prediction_binary.tif` and `prediction_prob.tif` in QGIS; compare with true-color patches (R=3, G=2, B=1)
- [ ] Write mail to Franzi with results + request for micro-data coordinates
