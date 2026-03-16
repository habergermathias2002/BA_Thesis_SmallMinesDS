"""
01_prepare_dataset.py
=====================

Kurz: Bereitet SmallMinesDS für das Training vor. Liest die Train/Test-Splits aus den
CSVs, extrahiert die 6 korrekten Spektralbänder (B2, B3, B4, B8A, B11, B12) aus den
13-Band-TIFs und speichert sie als 6-Band-GeoTIFFs in training/ und validation/.

WICHTIG – Band-Fix (v2):
  SmallMinesDS-TIFs haben 13 Bänder. Das Prithvi-Modell erwartet 6 Bänder:
    B2(Blue), B3(Green), B4(Red), B8A(VNIR_5), B11(SWIR_1), B12(SWIR_2)
  Die entsprechenden 0-basierten Indizes im 13-Band-Stack sind: [0, 1, 2, 7, 8, 9].
  Frühere Version kopierte die 13-Band-Dateien unverändert; terratorch nahm dann
  Bänder 0–5 (B2,B3,B4,B5,B6,B7) → kein B8A, kein SWIR → Band-Mismatch bei Inferenz.
  Diese Version extrahiert die richtigen 6 Bänder direkt.

HuggingFace 13-Band-Reihenfolge (README):
  0=Blue, 1=Green, 2=Red, 3=RE1, 4=RE2, 5=RE3, 6=NIR, 7=B8A, 8=SWIR1, 9=SWIR2,
  10=S1_VV, 11=S1_VH, 12=DEM

Output (6-Band-GeoTIFFs):
  data/GhanaMiningPrithvi/training/GH_1755_2022_IMG.tif   (6 bands)
  data/GhanaMiningPrithvi/training/GH_1755_2022_MASK.tif  (1 band, unchanged)
  data/GhanaMiningPrithvi/validation/...

Usage:
  python 00_Mathias_contribution/scripts/01_prepare_dataset.py
  (run from repo root)
"""

import os
import shutil
import csv
import numpy as np
import rasterio

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF_ROOT         = os.path.join(REPO_ROOT, "Hugging_Face_Input", "SmallMinesDS")
CSV_2022        = os.path.join(REPO_ROOT, "Hugging_Face_Input", "train_test_splits_2022.csv")
CSV_2016        = os.path.join(REPO_ROOT, "Hugging_Face_Input", "train_test_splits_2016.csv")
OUT_TRAIN       = os.path.join(REPO_ROOT, "data", "GhanaMiningPrithvi", "training")
OUT_VAL         = os.path.join(REPO_ROOT, "data", "GhanaMiningPrithvi", "validation")

# 0-based indices of the 6 Prithvi bands in the 13-band SmallMinesDS stack
# [Blue, Green, Red, B8A, SWIR1, SWIR2] = indices [0, 1, 2, 7, 8, 9]
BAND_INDICES_6 = [0, 1, 2, 7, 8, 9]

os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_VAL, exist_ok=True)


def read_splits(csv_path):
    """Returns dict: {patch_number_str: split}  e.g. {'1755_2022': 'train'}"""
    splits = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["patch_name"]
            stem = name.replace("MASK_", "").replace(".tif", "")
            splits[stem] = row["split"]
    return splits


def extract_and_save_patch(year, patch_id, split):
    """
    Reads a 13-band IMG TIF, extracts the 6 correct bands, and writes a new
    6-band GeoTIFF. Masks (1 band) are copied unchanged.
    """
    stem = f"GH_{patch_id}_{year}"

    src_img  = os.path.join(HF_ROOT, year, "IMAGE", f"IMG_{stem}.tif")
    src_mask = os.path.join(HF_ROOT, year, "MASK",  f"MASK_{stem}.tif")

    dst_dir  = OUT_TRAIN if split == "train" else OUT_VAL
    dst_img  = os.path.join(dst_dir, f"{stem}_IMG.tif")
    dst_mask = os.path.join(dst_dir, f"{stem}_MASK.tif")

    if not os.path.exists(src_img):
        print(f"  [WARN] Image not found: {src_img}")
        return False
    if not os.path.exists(src_mask):
        print(f"  [WARN] Mask not found: {src_mask}")
        return False

    # Extract 6 bands from the 13-band image
    with rasterio.open(src_img) as src:
        profile = src.profile.copy()
        # rasterio uses 1-based band indices
        data_6 = src.read([i + 1 for i in BAND_INDICES_6])  # (6, H, W)

    profile.update(count=6)
    with rasterio.open(dst_img, "w", **profile) as dst:
        dst.write(data_6)

    # Mask: copy unchanged (1 band)
    shutil.copy2(src_mask, dst_mask)
    return True


def main():
    print("Reading split CSVs...")
    splits_2022 = read_splits(CSV_2022)
    splits_2016 = read_splits(CSV_2016)

    print(f"Band extraction: 13-band → 6-band (indices {BAND_INDICES_6})")
    print(f"  = B2(Blue), B3(Green), B4(Red), B8A(VNIR_5), B11(SWIR_1), B12(SWIR_2)")

    total, copied, skipped = 0, 0, 0

    for year, splits in [("2022", splits_2022), ("2016", splits_2016)]:
        print(f"\nProcessing year {year} ({len(splits)} patches)...")
        for stem, split in splits.items():
            patch_id = stem.split("_")[0]
            total += 1
            success = extract_and_save_patch(year, patch_id, split)
            if success:
                copied += 1
            else:
                skipped += 1
            if copied % 500 == 0 and copied > 0:
                print(f"  {copied} patches extracted...")

    print(f"\n{'='*50}")
    print(f"Done. {copied}/{total} patches extracted (6 bands), {skipped} skipped.")
    print(f"\nOutput:")
    n_train = len([f for f in os.listdir(OUT_TRAIN) if f.endswith("_IMG.tif")])
    n_val   = len([f for f in os.listdir(OUT_VAL)   if f.endswith("_IMG.tif")])
    print(f"  Training:   {n_train} patch pairs  → {OUT_TRAIN}")
    print(f"  Validation: {n_val} patch pairs   → {OUT_VAL}")

    # Sanity check: verify first output file has 6 bands
    sample = [f for f in os.listdir(OUT_TRAIN) if f.endswith("_IMG.tif")]
    if sample:
        with rasterio.open(os.path.join(OUT_TRAIN, sample[0])) as s:
            print(f"\n  Verification: {sample[0]} has {s.count} bands (expected 6)")

    print(f"\nNext step: Upload data/GhanaMiningPrithvi/ to Drive, then train on Colab.")


if __name__ == "__main__":
    main()
