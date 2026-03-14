# TODO – Saturday, March 14, 2026

**Goal:** Train Prithvi-EO v2 on SmallMinesDS and run a first Galamsey inference test on a small area in the Bono region.

---

## Morning (Prep & Training Start)

### 1. Identify test area in Bono (~20 min)
- [ ] Open the false-color visualization from `Notebook.ipynb` (B11/B8A/B4)
- [ ] Look for bright teal/white patches near rivers → likely Galamsey candidates
- [ ] Cross-check coordinates in Google Earth (high-res imagery shows mining pits clearly)
- [ ] Note the bounding box (lat/lon or UTM coordinates) for a ~10×10 km test area

### 2. Prepare SmallMinesDS for training (~30 min)
- [ ] Read `train_test_splits_2022.csv` and `train_test_splits_2016.csv` to identify train/validation splits
- [ ] Rename files from `IMG_GH_XXXX_YYYY.tif` / `MASK_GH_XXXX_YYYY.tif` → `XXXX_YYYY_IMG.tif` / `XXXX_YYYY_MASK.tif` (format expected by training script)
- [ ] Create folder structure:
  ```
  GhanaMiningPrithvi/
      training/
          *_IMG.tif + *_MASK.tif
      validation/
          *_IMG.tif + *_MASK.tif
  ```
- [ ] Note: Images have 13 bands, training script only uses 6 → handled automatically by `dataset_bands` / `output_bands` in the script

### 3. Set up Google Colab & start training (~30 min setup, then ~3–5 h runtime)
- [ ] Upload SmallMinesDS to Google Drive (or mount from Colab)
- [ ] Install `terratorch` and dependencies (`pip install -r requirements.txt`)
- [ ] Adapt `train-prithvi-v2-300.py`:
  - [ ] Change `DATASET_PATH` to Colab/Drive path
  - [ ] Reduce `num_workers` (Colab has fewer CPU cores, try `num_workers=2`)
  - [ ] Ensure checkpoint saves to Drive (so it survives session timeout)
- [ ] Start training → **runs in background while you continue with the next steps**

---

## Midday (Pipeline Prep — while training runs)

### 4. Write patch extraction script for Bono (~30 min)
- [ ] Read the 10×10 km test area from `Bono_Merged_2025.tif` using rasterio windowed reading
- [ ] Cut into non-overlapping 128×128 patches
- [ ] Rescale pixel values × 10,000 (Bono data is 0–1 from GEE, training data is 0–10,000)
- [ ] Select only the 6 bands the model expects: B2, B3, B4, B8A, B11, B12 (indices 0–5 in Bono data)
- [ ] Save patches as individual TIFs in `data/patches_bono_test/`

### 5. Write inference script (~30 min)
- [ ] Load trained checkpoint from step 3
- [ ] Load Bono patches
- [ ] Normalize using SmallMinesDS statistics (means/stds from training script)
- [ ] Run forward pass → save both binary mask AND softmax probabilities per patch
- [ ] Reassemble patches into a single GeoTIFF prediction map (with correct georeferencing)

---

## Afternoon (Inference & Results)

### 6. Run inference on test area (~15–30 min)
- [ ] On Colab (GPU): fast, ~seconds for a 10×10 km area
- [ ] Or locally on Mac (CPU): ~5–10 min for a small test area
- [ ] Check output: does the prediction map show plausible mining locations?

### 7. Visualize results (~30 min)
- [ ] Create a 3-panel figure in Notebook.ipynb:
  - Panel 1: False-color satellite image of test area
  - Panel 2: Binary mining prediction mask
  - Panel 3: Mining probability heatmap (softmax output)
- [ ] Compare visually against Google Earth imagery
- [ ] Screenshot the result → this is what goes in the mail to Franzi

---

## Evening (Wrap-up)

### 8. Document results & write mail (~30 min)
- [ ] Update `Daily Progress/` with findings
- [ ] Write mail to Franzi with:
  - Methodology explanation
  - First result image
  - Request for micro-data (to focus analysis on field research locations)
  - Open questions (multi-temporal, output format, author contact)

---

## Potential Blockers & Workarounds

| Blocker | Workaround |
|---------|------------|
| Colab session disconnects during training | Save checkpoints every 10 epochs to Drive; use Kaggle as alternative (30h/week free) |
| `terratorch` installation fails on Colab | Pin versions from `requirements.txt`; try Kaggle if incompatible |
| Training doesn't converge | Use the paper's hyperparameters exactly; check that band selection and normalization match |
| Bono predictions look nonsensical | Verify rescaling (×10,000) and band order; try on SmallMinesDS validation set first as sanity check |

---

## Time Estimate

| Block | Duration | Cumulative |
|-------|----------|------------|
| 1. Identify test area | 20 min | 0:20 |
| 2. Prepare SmallMinesDS | 30 min | 0:50 |
| 3. Colab setup + start training | 30 min | 1:20 |
| *Training runs in background* | *3–5 h* | — |
| 4. Patch extraction script | 30 min | 1:50 |
| 5. Inference script | 30 min | 2:20 |
| 6. Run inference | 15–30 min | 2:50 |
| 7. Visualize | 30 min | 3:20 |
| 8. Document & mail | 30 min | 3:50 |
| **Total active work** | **~4 hours** | |
| **Total wall time (incl. training)** | **~6–8 hours** | |
