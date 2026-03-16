# Documentation – Saturday, March 14, 2026

**Goal:** Prepare the full pipeline for a first Galamsey detection test in the Bono region using Prithvi-EO v2.

---

## Progress Today

| Step | Status | Notes |
|------|--------|--------|
| 1. Prepare dataset (`01_prepare_dataset.py`) | **Neu: v2 mit Band-Fix** | Extrahiert jetzt die 6 korrekten Bänder [0,1,2,7,8,9] aus 13-Band-TIFs → 6-Band-GeoTIFFs. **Muss erneut ausgeführt werden!** |
| 2. Upload `GhanaMiningPrithvi` to Google Drive | Erneut nötig | Neue 6-Band-Dateien müssen hochgeladen werden |
| 3. Extract Bono test patches (`02_extract_bono_test_patches.py`) | Done | 16 patches in `data/patches_bono_test/`, center 8.054635, -2.025502, UTM E=607379.5 N=890465.8 |
| 4. Train on Colab (Colab Notebook) | **Erneut nötig** | Neutraining mit korrekt extrahierten 6-Band-Daten (gleicher Trainingsscript, gleiche Means/Stds) |
| 5. Download checkpoint → run inference (`04_inference_bono.py`) | Erneut nötig | Mit neuem Checkpoint |
| 6. Inferenz auf gesamter Bono-Region (`05_inference_bono_full.py`) | Optional / getestet | Mit `LIMIT_PATCHES=100` getestet; volle Region: viele Stunden auf CPU |
| 7. Ghana-Karte mit Galamsey-Overlay (`06_ghana_map_galamsey_bono.py`) | Done | `data/ghana_map_galamsey_bono.png`; nutzt Downsampling für große Raster (OOM-Vermeidung) |
| 8. Modell-Check / Proof (`plot_model_proof.py`) | Angepasst (6-Band) | An 6-Band-Format angepasst; nach Neutraining erneut ausführen |
| 9. Bono-Vergleichsbild (`plot_bono_test_comparison.py`) | Done | Gleiches Layout wie Proof, für Bono-Testregion → nach Neutraining erneut erstellen |
| 10. Diagnose Bono-Inferenz | Done | **Root Cause gefunden: Band-Mismatch** (Training: B2–B7; Inferenz: B2,B3,B4,B8A,B11,B12). Siehe Technical Report v1.3 |

**Environment:** macOS — use Conda env `smallmines`; `python` bzw. `python3` für alle Skripte. MPS (Metal) für PyTorch deaktiviert in `05_*` (Adaptive-Pool-Bug).

---

## Context

The SmallMinesDS dataset (4,270 labeled patches, SW Ghana, 2016 + 2022) is available locally. The Bono mosaic (22 GB, January 2025) is already merged. The plan is:

1. Prepare SmallMinesDS for training → run on Colab → get checkpoint
2. Extract 5×5 km Bono test area around a known Galamsey site
3. Run inference → first prediction map

---

## Files Created

- **Scripts:** `00_Mathias_contribution/scripts/` — alle Skripte sind unabhängig vom Original-Repo (keine Änderung an Repo-Dateien).
- **Colab training:** `00_Mathias_contribution/Colab Notebook/BA_Thesis_Model_Training_SmallMinesDS_data.ipynb` — Haupt-Notebook für das Training auf Google Colab.

**Übersicht aller Skripte (in Ausführungsreihenfolge):**

| Skript | Kurzbeschreibung |
|--------|------------------|
| `01_prepare_dataset.py` | SmallMinesDS aus HuggingFace-Struktur → flache Ordner `training/` und `validation/` |
| `02_extract_bono_test_patches.py` | 5×5 km Bono-Testgebiet → 16× 128×128 GeoTIFFs in `data/patches_bono_test/` |
| `03_train_colab.ipynb` / Colab Notebook | Training Prithvi-EO v2 auf SmallMinesDS (Colab GPU) |
| `04_inference_bono.py` | Inferenz auf 16 Bono-Test-Patches → GeoTIFFs + 3-Panel-Visualisierung |
| `05_inference_bono_full.py` | Inferenz auf ganzem Bono-Mosaik (fensterweise); `LIMIT_PATCHES` für Schnelltest |
| `06_ghana_map_galamsey_bono.py` | Karte Ghana + Bono mit Mining-Wahrscheinlichkeit (rot) |
| `make_bono_test_mosaic_png.py` | Stitch der 16 Patches zu einem 512×512 True-Color-PNG |
| `plot_galamsey_probability_map.py` | Weiß-Rot-Heatmap der P(Mining) für Testregion aus `prediction_prob.tif` |
| `plot_model_proof.py` | **Proof:** Satellitenbild + Mining-Wahrscheinlichkeit für SmallMinesDS-Trainings-Patches |
| `plot_bono_test_comparison.py` | **Bono-Vergleich:** Gleiches Layout wie Proof für die 5×5 km Bono-Testregion (Satellitenbild | P(Mining)) → zeigt nicht plausible Vorhersage |

---

### `01_prepare_dataset.py` (v2 — Band-Fix)

**Purpose:** Extracts the 6 correct Prithvi bands from 13-band SmallMinesDS GeoTIFFs and saves them as 6-band files in the flat `training/` and `validation/` folder structure.

**The problem it solves (v2 — Band-Fix):**
1. **Renaming:** HuggingFace uses `IMG_GH_1755_2022.tif`; training expects `*_IMG.tif`.
2. **Band-Extraktion (NEU):** SmallMinesDS-TIFs haben 13 Bänder. TerraTorch nimmt bei `dataset_bands = output_bands = [6 Namen]` immer die **ersten 6 Kanäle** (Indizes 0–5). In den 13-Band-Dateien sind das B2, B3, B4, **B5, B6, B7** — aber das Prithvi-Modell erwartet B2, B3, B4, **B8A, B11, B12** (Indizes 0, 1, 2, 7, 8, 9). Dieser **Band-Mismatch** war die Root Cause für die fehlgeschlagene Bono-Inferenz (→ Technical Report v1.3). Die v2 des Scripts extrahiert die korrekten 6 Bänder direkt aus den 13-Band-Dateien.

**What it does:**
- Reads `train_test_splits_2022.csv` and `train_test_splits_2016.csv`
- For each patch: reads the 13-band IMG TIF, extracts bands `[0, 1, 2, 7, 8, 9]` (B2, B3, B4, B8A, B11, B12), writes a new 6-band GeoTIFF
- Masks (1 band) are copied unchanged
- Output naming: `GH_1755_2022_IMG.tif` (6 bands), `GH_1755_2022_MASK.tif`

**Output:**
```
data/GhanaMiningPrithvi/
    training/    ← ~2,983 patch pairs (6-band IMG + 1-band MASK)
    validation/  ← ~1,287 patch pairs
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
Aktuell können **Bono-spezifische** Mittelwerte/Std genutzt werden (im Skript als Alternative zu SmallMinesDS auskommentiert bzw. umschaltbar). Für die **Trainingsdomäne** (Proof) müssen SmallMinesDS-Statistiken verwendet werden.

- **SmallMinesDS (Training/Proof):**  
  `means = [1473.81, 1703.35, 1696.68, 3832.40, 3156.11, 2226.07]`,  
  `stds = [223.44, 285.54, 413.82, 389.61, 451.50, 468.27]`
- **Bono (Testregion):** Werte aus allen 16 Test-Patches geschätzt; gleiche Formel `(raw − mean) / std`. Mit Bono-Stats steigt die vorhergesagte Mining-Wahrscheinlichkeit stark an, die räumliche Verteilung wirkt aber **zufällig** (Domain Shift, siehe Diagnose).

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

### `make_bono_test_mosaic_png.py`

**Zweck:** Stellt die 16 Bono-Test-Patches zu einem einzigen 512×512 True-Color-PNG zusammen (10 m/px). Nützlich, um die Testregion auf einen Blick zu sehen, ohne in QGIS alle Patches zu laden.

**Ausgabe:** `data/patches_bono_test/full_test_area_truecolor.png`

**Aufruf:** Nach `02_extract_bono_test_patches.py` und optional nach Inferenz:
```bash
python 00_Mathias_contribution/scripts/make_bono_test_mosaic_png.py
```

---

### `plot_galamsey_probability_map.py`

**Zweck:** Erzeugt eine Weiß-Rot-Heatmap der Mining-Wahrscheinlichkeit für die Bono-Testregion. Liest `prediction_prob.tif` aus `data/patches_bono_test/` und skaliert die Darstellung an den tatsächlichen Wertebereich (auch sehr kleine Werte sichtbar).

**Ausgabe:** `data/patches_bono_test/galamsey_probability_map.png`

**Voraussetzung:** `04_inference_bono.py` muss zuvor gelaufen sein.

---

### `plot_model_proof.py` (Modell-Check / Proof)

**Zweck:** Beweis, dass das Modell auf der **Trainingsdomäne** (SmallMinesDS) korrekt funktioniert. Zeigt für mehrere Trainings-Patches (mit bekanntem Mining-Anteil) nebeneinander:
- **Links:** 10×10 m Satellitenbild (True Color)
- **Rechts:** Mining-Wahrscheinlichkeit (weiß = kein Mining, rot = Mining)

Mining-Patches werden mit hoher Wahrscheinlichkeit rot erkannt; Non-Mining-Patches bleiben weiß. Damit ist belegt, dass das Problem bei der Bono-Inferenz **nicht** am Modell oder am Code liegt, sondern am **Domain Shift** (andere Region/Zeit).

**Ausgabe:** `data/model_proof_on_training_patches.png`

**Aufruf:** Checkpoint in `models/` erforderlich.
```bash
python 00_Mathias_contribution/scripts/plot_model_proof.py
```

---

### `plot_bono_test_comparison.py` (Bono-Vergleichsbild)

**Zweck:** Erzeugt ein **Vergleichsbild im gleichen Layout wie der Proof** – aber für die **Bono-Testregion** (5×5 km). Links: True-Color-Satellitenbild der Testfläche, Rechts: Mining-Wahrscheinlichkeit (weiß–rot). Dient als Gegenstück zum Proof: Hier funktioniert die Vorhersage **nicht** plausibel (Domain Shift); die räumliche Verteilung wirkt zufällig bzw. passt nicht zu bekannten Galamsey-Standorten.

**Ausgabe:** `data/patches_bono_test/bono_test_comparison.png`

**Voraussetzung:** `02_extract_bono_test_patches.py` und `04_inference_bono.py` müssen ausgeführt sein (`prediction_prob.tif` in `patches_bono_test/`).

**Aufruf:**
```bash
python 00_Mathias_contribution/scripts/plot_bono_test_comparison.py
```

**Für die Arbeit:** Proof (`model_proof_on_training_patches.png`) und Bono-Vergleich (`bono_test_comparison.png`) zusammen zeigen: Modell funktioniert auf Trainingsdomäne, liefert auf Bono keine plausible Galamsey-Erkennung.

---

## Diagnose & Interpretation der Inferenz-Ergebnisse

**Beobachtung:** Auf der Bono-Region liefert das Modell entweder nahezu überall 0 % Mining oder – bei Bono-Normalisierung – hohe, aber räumlich **zufällig wirkende** Mining-Wahrscheinlichkeiten, die nicht mit bekannten Galamsey-Standorten übereinstimmen.

**Root Cause (verifiziert): Band-Mismatch im Training**

Der Hauptfehler wurde durch eine detaillierte Analyse des TerraTorch-Codes und der SmallMinesDS-Bandordnung identifiziert (→ `Technical_Report_Inference_Diagnosis.md` v1.3, Section 11.3–11.4):

| | Training (IST) | Training (SOLL) | Inferenz (Bono) |
|--|---------------|-----------------|-----------------|
| **Bänder** | B2, B3, B4, **B5, B6, B7** (Indizes 0–5 der 13-Band-Datei) | B2, B3, B4, **B8A, B11, B12** (Indizes 0,1,2,7,8,9) | B2, B3, B4, **B8A, B11, B12** |
| **Means/Stds** | für B2,B3,B4,B8A,B11,B12 (korrekt berechnet, aber auf falsche Bänder angewendet) | gleich | gleich |

**Warum:** TerraTorch berechnet `filter_indices = [dataset_bands.index(b) for b in output_bands]`. Da `dataset_bands` und `output_bands` beide `["BLUE","GREEN","RED","VNIR_5","SWIR_1","SWIR_2"]` sind, ergibt das `[0,1,2,3,4,5]` → immer die ersten 6 Kanäle der Datei. Bei 13-Band-Dateien sind das B2–B7 statt der gewünschten B2,B3,B4,B8A,B11,B12.

**Fix implementiert:** `01_prepare_dataset.py` extrahiert jetzt Bänder `[0,1,2,7,8,9]` aus den 13-Band-TIFs und speichert 6-Band-Dateien. Damit liest TerraTorch die richtigen Kanäle.

**Zusätzliche Faktoren (sekundär):**

| Ursache | Bewertung | Maßnahme |
|--------|------------|----------|
| Domain Shift (Ort/Zeit: SW Ghana 2016/2022 vs. Bono 2025) | Mittel–Hoch | Nach Neutraining evaluieren; ggf. Fine-Tuning mit Bono-Labels |
| Normalisierung (Bono vs. SmallMinesDS) | Getestet | Bono-Stats → hohe, unplausible Vorhersagen; SmallMinesDS-Stats nach Band-Fix erneut testen |
| Backbone eingefroren | Mittel | Beim nächsten Training ggf. Backbone (teilweise) auftauen |
| Klassenungleichgewicht | Gering | Modell-Check auf Trainings-Patches bestätigt: Modell arbeitet korrekt |

**Proof vs. Bono-Vergleich:**
- **Proof:** `plot_model_proof.py` → `data/model_proof_on_training_patches.png`. Zeigt: Modell erkennt auf SmallMinesDS-Patches Mining (Mining → rot, Non-Mining → weiß). Hinweis: bisheriger Proof basiert auf falschem Band-Set; nach Neutraining erneut erstellen.
- **Bono-Vergleich:** `plot_bono_test_comparison.py` → `data/patches_bono_test/bono_test_comparison.png`. Vorhersage mit altem Modell nicht plausibel → nach Neutraining erneut evaluieren.

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
      │                                    │
      ▼                                    ├→ make_bono_test_mosaic_png.py → full_test_area_truecolor.png
Colab Notebook/BA_Thesis_Model_Training_SmallMinesDS_data.ipynb  (Google Colab — ~3–5 h)
      │
      ▼
models/*.ckpt  (download from Drive)
      │
      ├→ plot_model_proof.py  (Proof auf Trainings-Patches) → data/model_proof_on_training_patches.png
      │
      ▼ (combined with patches)
04_inference_bono.py  (Testregion 5×5 km)
      │
      ▼
prediction_*.tif, prediction_visualization.png in data/patches_bono_test/
      │
      ├→ plot_galamsey_probability_map.py → galamsey_probability_map.png
      ├→ plot_bono_test_comparison.py → bono_test_comparison.png (Vergleichsbild wie Proof, für Bono)
      │
Optional: Gesamte Bono-Region
      │
      ▼
05_inference_bono_full.py  (volles Mosaik, fensterweise; LIMIT_PATCHES für Test)
      │
      ▼
prediction_* in data/inference_bono_full/
      │
      ▼
06_ghana_map_galamsey_bono.py → data/ghana_map_galamsey_bono.png
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

### macOS / Apple Silicon
- **05_inference_bono_full.py:** Verwendet bewusst `device="cpu"`, da MPS (Metal) bei PyTorch einen Bug bei `adaptive_avg_pool2d` hat (nicht teilbare Input-/Output-Größen).
- **06_ghana_map_galamsey_bono.py:** Große Inferenz-Raster werden vor der Reprojektion auf max. 2000×2000 px herunterskaliert, um Speicherüberlauf (OOM) zu vermeiden.

---

## Next Steps — Band-Fix Workflow

### Sofort (Band-Fix → Neutraining)

1. [ ] `01_prepare_dataset.py` **erneut ausführen** → erzeugt 6-Band-GeoTIFFs in `data/GhanaMiningPrithvi/`
2. [ ] Alte `data/GhanaMiningPrithvi/` auf Drive **ersetzen** (oder löschen + neu hochladen)
3. [ ] **Neutraining auf Colab** (`Colab Notebook/BA_Thesis_Model_Training_SmallMinesDS_data.ipynb`) — gleicher Script, gleiche Means/Stds, gleiche Hyperparameter; einziger Unterschied: korrekte 6-Band-Eingabedaten
4. [ ] Neuen Checkpoint herunterladen → `models/`
5. [ ] `04_inference_bono.py` mit **SmallMinesDS Means/Stds** auf Bono-Test-Patches ausführen
6. [ ] `plot_model_proof.py` erneut ausführen (neuer Proof mit korrektem Modell)
7. [ ] `plot_bono_test_comparison.py` erneut ausführen → Bono-Ergebnis evaluieren

### Danach evaluieren

- [ ] Wenn Bono-Ergebnis plausibel: `05_inference_bono_full.py` auf gesamter Region ausführen
- [ ] Wenn Bono-Ergebnis weiterhin unplausibel: Domain Shift bestätigt → Fine-Tuning mit Bono-Labels nötig
- [ ] QGIS: Ergebnisse mit True-Color-Patches vergleichen
- [ ] Franzi: Ergebnisse + Diagnose mitteilen

### Bereits erledigt

- [x] Run `01_prepare_dataset.py` locally (v1: 13-Band-Kopie)
- [x] Upload `data/GhanaMiningPrithvi/` to Google Drive
- [x] Run `02_extract_bono_test_patches.py` (16 patches)
- [x] Train on Colab (v1: mit falschem Band-Set)
- [x] Download checkpoint; Run inference → prediction maps
- [x] Modell-Check: `plot_model_proof.py` (v1: auf falschem Band-Set)
- [x] Diagnose: **Root Cause = Band-Mismatch** (Technical Report v1.3); Domain Shift als Sekundärfaktor
- [x] Band-Fix implementiert in `01_prepare_dataset.py` (v2) und `plot_model_proof.py`
