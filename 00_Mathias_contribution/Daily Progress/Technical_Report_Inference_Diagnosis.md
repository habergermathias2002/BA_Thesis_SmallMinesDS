# Technical Report: Inference Failure Diagnosis (Galamsey Detection)

**Purpose:** Extract from code all technical details needed to diagnose why the Prithvi model predicts ~50–70% mining everywhere on 2025 Bono (GEE) data while training predictions look reasonable.  
**Method:** Information below is taken **only from the repository code and config**; no assumptions.

---

## 1. DATA SOURCES

### Training data (SmallMinesDS)

| Attribute | Value (from code/docs) |
|----------|------------------------|
| **Sensor** | Sentinel-2 (optical) + Sentinel-1 (SAR) + Copernicus DEM in full dataset; **model uses only Sentinel-2** |
| **Processing level** | Sentinel-2 **L2A** (Hugging_Face_Input/README.md: "Sentinel-2 L2A [blue, green, red, ...]") |
| **Source** | Hugging Face dataset `ellaampy/SmallMinesDS`; prepared locally via `01_prepare_dataset.py` into `data/GhanaMiningPrithvi/` |
| **Year / season** | **2016** and **2022**, January (dry season) – Hugging_Face_Input/README.md, README.md |
| **Spatial resolution** | 10 m (patches are 128×128 px; band order implies 10 m S2 bands; 20 m bands resampled in source) |
| **Bands used by model** | 6 bands only: BLUE, GREEN, RED, VNIR_5, SWIR_1, SWIR_2 (see Section 2) |
| **Full image shape** | 13×128×128 per patch (README: "13 bands for each image"; mask 1×128×128) |

Training and validation TIFs are **13-band** GeoTIFFs copied as-is from HuggingFace (no band subset in `01_prepare_dataset.py`). Band selection to 6 is done inside the **terratorch** `GenericNonGeoSegmentationDataModule` via `dataset_bands` / `output_bands` (not implemented in this repo).

### Inference data (Bono 2025)

| Attribute | Value (from code) |
|----------|-------------------|
| **Sensor** | Sentinel-2 only |
| **Processing level** | **L2A** – `00_Mathias_contribution/GEE_data_Export_Bono_Bono-East_Region.js` line 51: `ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')` (Surface Reflectance / L2A) |
| **Source** | **Google Earth Engine** – same script: filter Bono/Bono East, export to Drive |
| **Year / season** | **January 2025** – GEE script: `.filterDate('2025-01-01', '2025-01-31')` |
| **Spatial resolution** | **10 m** – GEE export: `scale: 10`; B11/B12 resampled to 10 m by GEE |
| **Bands used** | 6 bands: B2, B3, B4, B8A, B11, B12 – GEE: `medianMosaic.select(['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'])` |
| **Mosaic** | Median composite over January 2025; cloud masking via QA60 (bits 10 & 11); exported as 6-band GeoTIFF, then merged to `data/raw/Bono_Merged_2025.tif` |

### Same preprocessing pipeline?

**No.**  
- **Training:** SmallMinesDS L2A from Copernicus/HuggingFace; raw reflectance scale (see Section 3); 13-band patches; band selection and normalization inside terratorch.  
- **Inference:** GEE `COPERNICUS/S2_SR_HARMONIZED` L2A, median composite, QA60 cloud mask, **values divided by 10,000 in GEE** before export (0–1); 6-band export; locally rescaled ×10,000 then normalized in inference scripts. Different region, time, and scaling path.

---

## 2. FEATURE STACK

### Where the model input tensor is built

- **Training:** `GenericNonGeoSegmentationDataModule` (terratorch) loads `*_IMG.tif`, applies `means`/`stds`, and passes bands per `dataset_bands`/`output_bands`. Band indexing is inside the library (not in repo).  
- **Inference:**  
  - `04_inference_bono.py`: `patch = src.read()` → shape **(6, 128, 128)**; then `normalize(patch)`; then `torch.FloatTensor(patch_norm).unsqueeze(0)` → **(1, 6, 128, 128)**.  
  - `05_inference_bono_full.py`: `src.read(BAND_INDICES, window=window)` with `BAND_INDICES = list(range(1, 7))` → 6 bands; ×10000; normalize; same tensor shape.

### Band order and names (from code)

| Index (0-based) | Band name (model) | Sentinel-2 band |
|-----------------|-------------------|-----------------|
| 0 | BLUE | B2 |
| 1 | GREEN | B3 |
| 2 | RED | B4 |
| 3 | VNIR_5 | B8A |
| 4 | SWIR_1 | B11 |
| 5 | SWIR_2 | B12 |

Defined in:  
- `scripts/train-prithvi-v2-300.py` lines 17–24: `ghana_mining_bands = ["BLUE", "GREEN", "RED", "VNIR_5", "SWIR_1", "SWIR_2"]`  
- `04_inference_bono.py` line 70, `05_inference_bono_full.py` line 56: same list.  
- GEE export: `select(['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'])` – same order.  
- Bono mosaic / patches: `02_extract_bono_test_patches.py` lines 56–58: "Bono mosaic band order: B2(0), B3(1), B4(2), B8A(3), B11(4), B12(5)" and `BAND_INDICES = [1, 2, 3, 4, 5, 6]` (rasterio 1-based).

### SmallMinesDS 13-band order (HuggingFace README)

- Index 0–9: Sentinel-2 L2A **[blue, green, red, red edge 1, red edge 2, red edge 3, near infrared, red edge 4, swir1, swir2]**  
- Index 10–11: Sentinel-1 VV, VH  
- Index 12: DEM  

So for the **6 model bands** (B2, B3, B4, B8A, B11, B12) the expected indices in the 13-band stack are **0, 1, 2, 7, 8, 9** (blue, green, red, red edge 4/B8A, swir1, swir2). **Verification (Section 11.3):** TerraTorch was checked: with `dataset_bands = output_bands = [BLUE, GREEN, RED, VNIR_5, SWIR_1, SWIR_2]`, `filter_indices` becomes **[0, 1, 2, 3, 4, 5]** → first six channels of the file = **B2, B3, B4, B5, B6, B7** (no B8A, no B11, B12). **Training used the wrong bands; inference sends the correct six.** See script `verify_training_band_selection.py`.

### Indices (NDVI, NDWI, etc.)

No code in the repo adds spectral indices. Input to the model is the 6 bands only.

### Tensor shape fed to the model

- **Batch shape:** **(1, 6, 128, 128)** (N, C, H, W).  
- **Patch size:** 128×128.

---

## 3. RADIOMETRIC SCALING

### Training pipeline (from code)

- **Script:** `scripts/train-prithvi-v2-300.py` (and Colab notebook `03_train_colab.ipynb` same logic).  
- **Raw pixel range:** Not explicitly stated in repo. SmallMinesDS README describes L2A optical bands; Sentinel-2 L2A reflectance is typically **0–1 or 0–10,000** depending on product. Training code uses **means/stds on the order of 10³** (e.g. Blue mean 1473.81), so the **training data are treated as raw DN in roughly 0–10,000** scale.  
- **Scaling:** No `/10000` or `*10000` in training scripts. Data are read and normalized directly.  
- **Clipping:** Not applied in training code.  
- **Normalization:** Z-score per band: **normalized = (value - mean) / std**, with:

```text
means = [1473.81388377, 1703.35249650, 1696.67685941, 3832.39764247, 3156.11122121, 2226.06822112]
stds  = [ 223.43533204,  285.53613398,  413.82320306,  389.61483882,  451.49534791,  468.26765909]
```

(`train-prithvi-v2-300.py` lines 27–42; same in Colab notebook.)  
Normalization is applied inside the datamodule (means/stds passed to `GenericNonGeoSegmentationDataModule`).

### Inference pipeline – Bono extraction (`02_extract_bono_test_patches.py`)

- **Raw in file:** GEE exports **0–1** (after `.divide(10000)` in GEE).  
- **Scaling:** Code explicitly **multiplies by 10,000**: `data = data.astype(np.float32) * 10000.0` (line 100) to bring to **0–10,000** range for consistency with “raw DN” expectation.  
- **No clipping** in this script.

### Inference pipeline – normalization (`04_inference_bono.py`, `04_inference_bono_2.0.py`, `05_inference_bono_full.py`)

- **04_inference_bono.py** (lines 54–59):  
  Uses **Bono-specific** statistics (comment: "TEST A: Bono-specific statistics (computed from all 16 test patches)"):

```text
MEANS = np.array([ 583.63,  851.72, 1241.71, 2411.21, 3027.37, 2290.58], dtype=np.float32)
STDS  = np.array([ 157.83,  227.45,  348.81,  717.34,  828.90,  609.14], dtype=np.float32)
```

  Comment in code: "SmallMinesDS stats (original): [1473.81, 1703.35, 1696.68, 3832.40, 3156.11, 2226.07]". So **inference 04 currently does NOT use training (SmallMinesDS) statistics.**

- **04_inference_bono_2.0.py** (Z-Score Domain Alignment):  
  Uses the **same Bono** MEANS/STDS (`BONO_MEANS`, `BONO_STDS`). The script documents the intent to bring Bono data into a "neutral" z-score space via `(x - Bono_mean) / Bono_std` so that the Prithvi encoder sees compatible inputs. **Outcome:** Inferenz liefert weiterhin **~60 % Mining-Anteil** (binär) bzw. **~55 % mittlere P(Mining)** – kein plausibles räumliches Muster, gleicher Fehler wie 04. Siehe Abschnitt 11.

- **05_inference_bono_full.py** (lines 42–43):  
  Uses **SmallMinesDS** (training) statistics:

```text
MEANS = np.array([1473.81, 1703.35, 1696.68, 3832.40, 3156.11, 2226.07], dtype=np.float32)
STDS  = np.array([ 223.44,  285.54,  413.82,  389.61,  451.50,  468.27], dtype=np.float32)
```

- **Formula in both:** `(patch - MEANS[:, None, None]) / STDS[:, None, None]` after patch values are on **0–10,000** scale (02 or 05 apply ×10000 where needed).

### Summary: TRAINING vs INFERENCE

| Step | Training | Inference (04) | Inference (05) |
|------|----------|----------------|----------------|
| Raw range | ~0–10,000 (implied by means) | 0–1 in GEE → ×10000 → 0–10,000 | Same (×10000 in 05) |
| Normalization | (x - mean) / std, SmallMinesDS means/stds | (x - mean) / std, **Bono** means/stds | (x - mean) / std, **SmallMinesDS** means/stds |

**Critical difference:**  
- **04** uses **Bono** mean/std. That makes Bono input sit in a different z-score space than training; the decoder was trained on SmallMinesDS z-scores. Result in practice: **~50–70% mining** everywhere (as reported).  
- **05** uses **SmallMinesDS** mean/std. If Bono raw values (after ×10000) are systematically lower than in SW Ghana (different season/region), then (x - SmallMinesDS_mean) / SmallMinesDS_std can be **strongly negative** (e.g. 3–4 std below training), which can push outputs toward one class (e.g. previously observed “0% mining” everywhere with SmallMinesDS stats on Bono).

So: **radiometric scaling and choice of statistics differ between training and inference, and between 04 and 05.**

---

## 4. PATCH GENERATION

### Training

- **No patch generation in this repo.** Training uses **pre-cut 128×128 patches** from SmallMinesDS (HuggingFace).  
- **Patch size:** 128×128 (README, and all scripts assume 128).  
- **Stride / overlap:** N/A (fixed patches).  
- **Padding:** Not used for training data.  
- **Normalization:** Global (same means/stds for all patches), applied in the datamodule.

### Inference (Bono)

- **02_extract_bono_test_patches.py:** Reads a 5×5 km window from the mosaic; pads to a multiple of 128 with **zeros** (`mode="constant", constant_values=0`); then cuts non-overlapping **128×128** patches (stride 128).  
- **05_inference_bono_full.py:** Reads the full mosaic in **non-overlapping** 128×128 windows; no padding of the raster; output size is `(height // PATCH_SIZE) * PATCH_SIZE` (truncates remainder).

---

## 5. MODEL CONFIGURATION

All from `scripts/train-prithvi-v2-300.py` and Colab notebook / inference load_model().

| Item | Value |
|------|--------|
| **Prithvi version** | `prithvi_eo_v2_300` (backbone key) |
| **Backbone size** | 300M |
| **Input channels** | 6 |
| **Pretrained** | Training: `pretrained: True` (HuggingFace); inference: `pretrained: False` (weights from checkpoint) |
| **Decoder** | UperNetDecoder |
| **rescale** | True |
| **backbone_num_frames** | 1 |
| **head_dropout** | 0.1 |
| **decoder_scale_modules** | True |
| **Loss** | `loss="ce"` (cross-entropy) |
| **Optimizer** | AdamW |
| **Learning rate** | 1e-3 |
| **Optimizer weight_decay** | 0.05 |
| **Class weights** | Not used – `class_weights=[0.1, 0.9]` is commented out in train-prithvi-v2-300.py |
| **freeze_backbone** | True (only decoder/head trained) |
| **class_names** | ['Non_mining', 'Mining'] |
| **num_classes** | 2 |

---

## 6. TRAINING DATASET STATISTICS

- **Number of patches:** 4,270 total (README); 2,983 training + 1,287 validation (from `01_prepare_dataset.py` and docs).  
- **Mining vs background:** From split CSVs (`train_test_splits_2022.csv`, `train_test_splits_2016.csv`), column `class_percentage` (mining % per patch):  
  - 2022: n=2,135, **mean mining % ≈ 7.65%**, min=0, max≈91.91.  
  - 2016: n=2,135, **mean mining % ≈ 5.49%**, min=0, max≈82.28.  
- **Class imbalance:** Strong – most patches and pixels are non-mining; mining is a minority. Class weights are not applied in the current training script.

---

## 7. INFERENCE PIPELINE (FULL TRACE)

### 04_inference_bono.py (test patches)

1. **Load imagery:** Read pre-extracted 6-band GeoTIFFs from `data/patches_bono_test/` (created by 02). Each patch already **0–10,000** (02 applied ×10000).  
2. **Preprocessing:** None beyond normalization.  
3. **Tiling:** Already 128×128 per file; no tiling in 04.  
4. **Normalization:** `(patch - MEANS) / STDS` with **Bono** MEANS/STDS (see Section 3).  
5. **Model:** `task.model(tensor)`; logits from `out.output`; softmax; channel 1 = Mining probability.  
6. **Post-processing:** Reassemble patches via `patch_index.csv`; threshold 0.5 for binary mask; write GeoTIFFs and PNG.

### 04_inference_bono_2.0.py (Z-Score Domain Alignment, test patches)

Same flow as 04; normalization is **Bono** only: `align_and_normalize(patch) = (patch - BONO_MEANS) / BONO_STDS`. Outputs: `prediction_prob_aligned.tif`, `prediction_binary_aligned.tif`, `prediction_visualization_aligned.png`. **Result:** Still ~60 % mining, implausible (Section 11).

### 05_inference_bono_full.py (full mosaic)

1. **Load imagery:** `rasterio.open(MOSAIC_PATH)`; read 6 bands (indices 1–6) in 128×128 windows.  
2. **Preprocessing:** `patch * 10000.0` (GEE export is 0–1).  
3. **Tiling:** Non-overlapping 128×128 windows; optional `LIMIT_PATCHES`.  
4. **Normalization:** `(patch - MEANS) / STDS` with **SmallMinesDS** MEANS/STDS.  
5. **Model:** Same as 04.  
6. **Post-processing:** Write probability and binary rasters per window.

### Differences vs training pipeline

- **Input source:** Training = 13-band TIFs (terratorch selects 6); inference = 6-band TIFs (Bono order B2,B3,B4,B8A,B11,B12).  
- **Radiometry:** Training = raw ~0–10k + SmallMinesDS z-score. Inference 04 = 0–10k + **Bono** z-score. Inference 05 = 0–10k + SmallMinesDS z-score.  
- **Augmentation:** Training uses HorizontalFlip/VerticalFlip; inference uses none.  
- **Spatial:** Same 128×128; no overlap in inference.

---

## 8. POTENTIAL DOMAIN SHIFT

| Factor | Training (SmallMinesDS) | Inference (Bono 2025) |
|--------|-------------------------|------------------------|
| **Region** | Southwestern Ghana (5 districts) | Bono / Bono East |
| **Time** | January 2016, January 2022 | January 2025 |
| **Band distributions** | Means (Blue) ≈ 1474; (NIR) ≈ 3832; etc. | Bono (after ×10k): means Blue ≈ 584, NIR ≈ 2411 (from 04 script) – **much lower** than training |
| **Scaling** | No ×10000 in training path; data already in ~0–10k scale | GEE 0–1 → ×10000 → 0–10k |
| **Band availability** | Same 6 S2 bands (B2,B3,B4,B8A,B11,B12) | Same 6 bands |
| **Spatial resolution** | 10 m (effective) | 10 m |
| **Season** | January (dry) | January (dry) |

**Per-band comparison (from code):**  
- Training (SmallMinesDS) means: [1473.81, 1703.35, 1696.68, 3832.40, 3156.11, 2226.07].  
- Bono (04 script) means: [583.63, 851.72, 1241.71, 2411.21, 3027.37, 2290.58].  
Bono values after ×10,000 are **systematically lower** than training (e.g. Blue 583 vs 1474). So with **SmallMinesDS** normalization, Bono pixels sit several standard deviations below the training distribution; with **Bono** normalization, they are centered differently and the decoder sees a distribution it was not trained on → both can cause poor or biased predictions.

---

## 9. LIKELY FAILURE POINTS (RANKED)

1. **Band selection mismatch (training vs inference) — VERIFIED**  
   **Probability: Very high; root cause identified.**  
   Training uses 13-band SmallMinesDS TIFs. The script passes `dataset_bands = output_bands = [BLUE, GREEN, RED, VNIR_5, SWIR_1, SWIR_2]`. TerraTorch interprets this as “the file has 6 bands in this order” and uses **filter_indices = [0, 1, 2, 3, 4, 5]** → the **first six channels** of the 13-band file. Per HuggingFace README, indices 0–5 are **B2, B3, B4, B5, B6, B7** (blue, green, red, RE1, RE2, RE3). So **training saw no B8A, no B11, B12 (no narrow NIR, no SWIR)**. Inference (04, 05, Bono mosaic) sends **B2, B3, B4, B8A, B11, B12** in that order. So the model was **trained on the wrong spectral channels** and is applied at inference to the correct Prithvi bands → total band mismatch. See Section 11.3 and script `verify_training_band_selection.py`.

2. **Domain shift (geographic / temporal / feature distribution)**  
   **Probability: Very high.**  
   Training: SW Ghana, 2016/2022. Inference: Bono, 2025. Different region, year, and likely atmosphere/vegetation/soil. The decoder was trained on SmallMinesDS feature distributions; Bono inputs (even after z-score alignment) lie in a different part of feature space. **Evidence:** 04_inference_bono_2.0.py (“Z-Score Domain Alignment” with Bono stats) was run explicitly to correct normalization; result **unchanged** (~60 % mining, implausible). So normalization alone does **not** fix the failure (see Section 11). Domain shift remains a major factor; the band mismatch above can fully explain wrong predictions even without domain shift.

3. **Radiometric / value-range domain shift**  
   **Probability: High.**  
   Even with correct (SmallMinesDS) normalization, Bono raw values (after ×10k) are lower than training. So (x - train_mean) / train_std is more negative for Bono → out-of-distribution inputs and unstable class probabilities (e.g. all “non-mining” when using SmallMinesDS stats, or “random” when using Bono stats).

4. **Normalization statistics (contributing, but not sufficient to fix)**  
   **Probability: High as contributor; insufficient as sole fix.**  
   Using Bono means/stds (04, 04_2.0) yields ~50–70 % mining; using SmallMinesDS means/stds on Bono can yield ~0 % (out-of-distribution). So normalization choice strongly affects output, but **aligning to Bono statistics (04_2.0) did not produce plausible predictions** (Section 11). The failure persists beyond normalization.

5. **Class imbalance and no class weights**  
   **Probability: Medium.**  
   Mining is rare (mean ~5–8% per patch); CE loss without weights can bias toward majority class or, after domain shift, produce skewed probabilities.

6. **GEE scaling or cloud masking**  
   **Probability: Lower.**  
   GEE divide(10000) and QA60 masking are documented; inference explicitly undoes ×10000. Remaining risk: different L2A processing or harmonization between SmallMinesDS source and GEE S2_SR_HARMONIZED.

---

## 10. DEBUGGING PLAN

### Step 1: Align inference 04 with training normalization — **TRIED (Bono alignment in 04_2.0)**

- **Code:** `04_inference_bono.py` (SmallMinesDS stats in 05); **Bono alignment** implemented in `04_inference_bono_2.0.py` (Z-Score Domain Alignment with Bono MEANS/STDS).  
- **What was done:** 04_2.0 was run to test whether centering Bono data with Bono statistics fixes the implausible predictions.  
- **Result:** **No change.** ~60 % mining (binary), ~55 % mean P(Mining); spatial pattern still implausible. See Section 11.  
- **Conclusion:** Normalization choice affects the numbers (SmallMinesDS stats → very low probs; Bono stats → high probs) but **neither yields plausible Bono predictions**. So “align with training” or “align with Bono” alone is not a sufficient fix.

### Step 2: Verify band order and content on one training patch — **DONE (mismatch confirmed)**

- **Code:** `00_Mathias_contribution/scripts/verify_training_band_selection.py` loads a training TIF, prints per-band means, and documents terratorch’s logic.  
- **Check:** TerraTorch source: `generic_pixel_wise_dataset.py` uses `filter_indices = [dataset_bands.index(band) for band in output_bands]`. With `dataset_bands = output_bands = [BLUE, GREEN, RED, VNIR_5, SWIR_1, SWIR_2]` (6 names), this yields **[0, 1, 2, 3, 4, 5]** → first 6 channels of the file.  
- **Result:** **Confirmed.** Training used bands **0–5** = B2, B3, B4, B5, B6, B7 (no B8A, no B11, B12). Inference sends B2, B3, B4, B8A, B11, B12. **Band mismatch is verified.** See Section 11.3.

### Step 3: Run model on a known mining training patch (sanity check)

- **Code:** Use same pipeline as `plot_model_proof.py` but ensure the 6 bands passed to the model are **0,1,2,7,8,9** (not 0:6) if the TIF is 13-band. Load one high-mining patch (e.g. from PATCHES list), normalize with SmallMinesDS means/stds, forward pass.  
- **Check:** Mining probability should be high for that patch.  
- **Expected:** If high mining probability on a high-mining training patch, model and band order are consistent; if not, band selection or normalization is wrong.

### Step 4: Compare per-band statistics (training vs Bono)

- **Code:** (a) Over a sample of training patches (e.g. 100), compute mean and std per band (over pixels and patches). (b) Over Bono patches (16 or from 02), compute same after ×10000.  
- **Check:** Tabulate side-by-side; compute (Bono_mean - Train_mean) / Train_std per band.  
- **Expected:** Large offsets (e.g. >2) indicate strong distribution shift; supports adapting normalization or doing domain adaptation.

### Step 5: Single-patch ablation on Bono

- **Code:** Pick one Bono patch; run inference with (a) SmallMinesDS stats, (b) Bono stats, (c) per-patch z-score (mean/std of that patch only).  
- **Check:** Log mean mining probability and variance for each.  
- **Expected:** (c) removes global scaling issues; if (c) gives more plausible spatial pattern, the main issue is global normalization vs Bono distribution.

### Step 6: Confirm GEE export scale

- **Code:** In GEE or on exported TIF: sample a few pixels in a vegetated and a bright (e.g. bare soil) area; check values before any ×10000.  
- **Check:** Values should be in 0–1 (and after ×10000 in 0–10k). Compare with typical SmallMinesDS values (e.g. Blue ~1400).  
- **Expected:** Confirms that ×10000 is the right correction and that remaining gap is distributional (region/season), not a missing factor.

---

## 11. ATTEMPTED FIXES (FAILED)

The following debugging steps were carried out. **None resolved the implausible ~50–70% mining prediction on Bono.**

### 11.1 Z-Score Domain Alignment (04_inference_bono_2.0.py)

- **Hypothesis:** The high/random mining probability is caused by a normalization mismatch. Centering Bono data with Bono mean/std (“Z-Score Domain Alignment”) should put inputs in a neutral statistical space and restore compatibility with the Prithvi encoder.
- **Implementation:** `00_Mathias_contribution/scripts/04_inference_bono_2.0.py` – same pipeline as 04, with `align_and_normalize(patch) = (patch - BONO_MEANS) / BONO_STDS` (identical to 04’s Bono stats). Outputs: `prediction_prob_aligned.tif`, `prediction_binary_aligned.tif`, `prediction_visualization_aligned.png`; comparison image: `plot_bono_test_comparison_aligned.py` → `bono_test_comparison_aligned.png`.
- **Result:** **Unchanged.** Binary mining share ~**60.68 %**; mean P(Mining) ~**54.7 %**; heatmap still shows high, non-plausible mining probability across most of the test area. No improvement over 04_inference_bono.py with Bono stats.
- **Conclusion:** Normalization choice (Bono vs SmallMinesDS) is **not** the sole cause. Aligning to Bono statistics does not fix the failure. The decoder/head, trained on SmallMinesDS feature distributions, produces similarly wrong outputs when fed Bono-normalized inputs → **domain shift in feature space** (region/time/spectral response) dominates; adjusting input z-scores to Bono does not compensate for that.

### 11.2 Summary of attempted fixes

| Attempt | What was tried | Outcome |
|--------|----------------|---------|
| Bono normalization (04, 04_2.0) | Use Bono mean/std so inputs are z-scored to local statistics | ~55–60 % mining; implausible, no spatial coherence |
| SmallMinesDS normalization (05, earlier 04 tests) | Use training mean/std on Bono (after ×10000) | Either ~0 % mining (out-of-distribution inputs) or still poor when combined with other factors |

**Takeaway:** Radiometric alignment (Bono or training stats) alone does **not** fix the Bono inference. The model’s failure is consistent with **domain shift** (different region, time, and possibly acquisition/processing), not solely a scaling bug. Next steps would need to address domain adaptation (e.g. fine-tuning on Bono labels, unfreezing backbone, or adding Bono to training).

### 11.3 Band selection verification (Diagnose zuerst)

- **Hypothesis:** TerraTorch might use the first 6 bands of the 13-band TIF (indices 0–5 = B2, B3, B4, B5, B6, B7) instead of the correct Prithvi set (0, 1, 2, 7, 8, 9 = B2, B3, B4, B8A, B11, B12). That would cause a **total band mismatch** between training and inference.
- **Implementation:** Script `00_Mathias_contribution/scripts/verify_training_band_selection.py` loads a training GeoTIFF, prints per-band means, and documents TerraTorch’s band logic from `terratorch/datasets/generic_pixel_wise_dataset.py`.
- **TerraTorch logic (from source):** `dataset_bands` and `output_bands` are both set to `["BLUE", "GREEN", "RED", "VNIR_5", "SWIR_1", "SWIR_2"]`. The code computes `filter_indices = [self.dataset_bands.index(band) for band in self.output_bands]` → **[0, 1, 2, 3, 4, 5]**. So the dataset uses the **first six channels** of the raster. For a 13-band SmallMinesDS file (order: blue, green, red, RE1, RE2, RE3, NIR, B8A, swir1, swir2, …), that is **B2, B3, B4, B5, B6, B7** — **no B8A, no B11, B12**.
- **Result:** **Mismatch confirmed.** Training: channels 0–5 (wrong set). Inference: Bono and 04/05 send B2, B3, B4, B8A, B11, B12 (correct set). The model was trained on different spectral inputs than it receives at inference; this alone can explain wrong and unstable predictions.
- **Recommended fix:** Preprocess SmallMinesDS so that the **6-band GeoTIFFs** passed to the training pipeline have exactly bands **[0, 1, 2, 7, 8, 9]** (B2, B3, B4, B8A, B11, B12) in that order before copying to `GhanaMiningPrithvi/`. Then `dataset_bands = output_bands = [BLUE, GREEN, RED, VNIR_5, SWIR_1, SWIR_2]` will refer to the correct channels. Alternatively, use a data pipeline that explicitly selects indices 0, 1, 2, 7, 8, 9 from the 13-band files (e.g. in a custom dataset or in 01_prepare_dataset).

### 11.4 Band-Fix implementiert (01_prepare_dataset.py v2)

- **Implementation:** `01_prepare_dataset.py` wurde überarbeitet. Statt die 13-Band-TIFs unverändert zu kopieren, extrahiert das Script jetzt die 6 korrekten Bänder (Indizes `[0, 1, 2, 7, 8, 9]` = B2, B3, B4, B8A, B11, B12) mit `rasterio` und speichert sie als neue 6-Band-GeoTIFFs. Masken bleiben unverändert (1 Band).
- **Auswirkung auf Normalisierung:** Die Training-Means/Stds (`[1473.81, 1703.35, 1696.68, 3832.40, 3156.11, 2226.07]`) entsprechen den korrekten Prithvi-Bändern und passen nun zu den tatsächlich in den 6-Band-Dateien enthaltenen Kanälen. Keine Neuberechnung nötig.
- **Auswirkung auf TerraTorch:** Mit 6-Band-Dateien ergeben `filter_indices = [0,1,2,3,4,5]` korrekt die 6 Prithvi-Bänder (da die Datei nur noch diese enthält). Der Trainingsscript (`train-prithvi-v2-300.py`) muss nicht geändert werden.
- **Auswirkung auf Inferenz:** Keine Änderung nötig — die Inferenz-Scripts (04, 05) nutzen bereits B2, B3, B4, B8A, B11, B12 aus dem Bono-Export. Training und Inferenz verwenden nun dieselben Spektralkanäle.
- **Angepasste Scripts:**
  - `01_prepare_dataset.py` — extrahiert jetzt 6 Bänder statt 13 zu kopieren
  - `plot_model_proof.py` — liest nun 6-Band-Dateien, korrigierte True-Color-Indizes (R=2, G=1, B=0)
- **Nächste Schritte:**
  1. `01_prepare_dataset.py` lokal ausführen → neue 6-Band-Training/Validation-Dateien erzeugen
  2. `data/GhanaMiningPrithvi/` auf Google Drive hochladen
  3. Modell auf Colab neu trainieren (gleicher Trainingsscript, gleiche Means/Stds)
  4. Neues Checkpoint herunterladen und Inferenz auf Bono wiederholen
  5. Ergebnis evaluieren

---

**Document version:** 1.3 – Band-Fix in `01_prepare_dataset.py` implementiert (Section 11.4); `plot_model_proof.py` an 6-Band-Format angepasst.  
**Files cited:**  
`scripts/train-prithvi-v2-300.py`, `00_Mathias_contribution/scripts/01_prepare_dataset.py`, `00_Mathias_contribution/scripts/02_extract_bono_test_patches.py`, `04_inference_bono.py`, `04_inference_bono_2.0.py`, `05_inference_bono_full.py`, `00_Mathias_contribution/scripts/plot_bono_test_comparison_aligned.py`, `00_Mathias_contribution/scripts/verify_training_band_selection.py`, `00_Mathias_contribution/scripts/plot_model_proof.py`, `terratorch/datasets/generic_pixel_wise_dataset.py` (site-packages), `00_Mathias_contribution/GEE_data_Export_Bono_Bono-East_Region.js`, `Hugging_Face_Input/README.md`, `01_prepare_dataset.py`, Colab notebook `03_train_colab.ipynb`.
