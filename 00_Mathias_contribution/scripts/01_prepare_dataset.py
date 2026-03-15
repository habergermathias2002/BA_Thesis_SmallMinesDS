"""
01_prepare_dataset.py
=====================

Kurz: Bereitet SmallMinesDS für das Training vor. Liest die Train/Test-Splits aus den
CSVs, kopiert die Bild- und Masken-Dateien aus der HuggingFace-Ordnerstruktur und
benennt sie so um, dass sie in die flachen Ordner training/ und validation/ passen
(Endung _IMG.tif / _MASK.tif). So findet das Training-Skript alle Patches.

What it does:
  - Reads train/test splits from the two CSV files (2016 + 2022)
  - Copies and renames files from the HuggingFace folder structure into the
    flat training/ and validation/ folders expected by the training script

HuggingFace structure (input):
  Hugging_Face_Input/SmallMinesDS/2022/IMAGE/IMG_GH_1755_2022.tif
  Hugging_Face_Input/SmallMinesDS/2022/MASK/MASK_GH_1755_2022.tif

Expected training structure (output):
  data/GhanaMiningPrithvi/training/GH_1755_2022_IMG.tif
  data/GhanaMiningPrithvi/training/GH_1755_2022_MASK.tif
  data/GhanaMiningPrithvi/validation/GH_0001_2022_IMG.tif
  data/GhanaMiningPrithvi/validation/GH_0001_2022_MASK.tif

Why this renaming?
  The training script uses glob patterns: img_grep="*_IMG.tif" and
  label_grep="*_MASK.tif". The HuggingFace filenames start with IMG_/MASK_
  which would NOT match these patterns. Renaming puts the suffix at the end.

Usage:
  python 00_Mathias_contribution/scripts/01_prepare_dataset.py
  (run from repo root)
"""

import os
import shutil
import csv

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF_ROOT         = os.path.join(REPO_ROOT, "Hugging_Face_Input", "SmallMinesDS")
CSV_2022        = os.path.join(REPO_ROOT, "Hugging_Face_Input", "train_test_splits_2022.csv")
CSV_2016        = os.path.join(REPO_ROOT, "Hugging_Face_Input", "train_test_splits_2016.csv")
OUT_TRAIN       = os.path.join(REPO_ROOT, "data", "GhanaMiningPrithvi", "training")
OUT_VAL         = os.path.join(REPO_ROOT, "data", "GhanaMiningPrithvi", "validation")

os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_VAL, exist_ok=True)


def read_splits(csv_path):
    """Returns dict: {patch_number_str: split}  e.g. {'1755_2022': 'train'}"""
    splits = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # patch_name looks like: MASK_1755_2022.tif
            name = row["patch_name"]            # → MASK_1755_2022.tif
            stem = name.replace("MASK_", "").replace(".tif", "")  # → 1755_2022
            splits[stem] = row["split"]
    return splits


def copy_patch(year, patch_id, split):
    """
    patch_id: e.g. '1755'  (zero-padded 4 digits)
    year:     e.g. '2022'
    Copies IMG + MASK to the right output folder with correct suffix naming.
    """
    stem = f"GH_{patch_id}_{year}"  # e.g. GH_1755_2022

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

    shutil.copy2(src_img,  dst_img)
    shutil.copy2(src_mask, dst_mask)
    return True


def main():
    print("Reading split CSVs...")
    splits_2022 = read_splits(CSV_2022)
    splits_2016 = read_splits(CSV_2016)

    total, copied, skipped = 0, 0, 0

    for year, splits in [("2022", splits_2022), ("2016", splits_2016)]:
        print(f"\nProcessing year {year} ({len(splits)} patches)...")
        for stem, split in splits.items():
            # stem is like '1755_2022' — extract just the ID part
            patch_id = stem.split("_")[0]  # '1755'
            total += 1
            success = copy_patch(year, patch_id, split)
            if success:
                copied += 1
            else:
                skipped += 1

    print(f"\n{'='*50}")
    print(f"Done. {copied}/{total} patches copied, {skipped} skipped.")
    print(f"\nOutput:")
    print(f"  Training:   {len(os.listdir(OUT_TRAIN))//2} patch pairs  → {OUT_TRAIN}")
    print(f"  Validation: {len(os.listdir(OUT_VAL))//2} patch pairs   → {OUT_VAL}")
    print(f"\nNext step: Run 03_train_colab.py on Google Colab or Kaggle.")


if __name__ == "__main__":
    main()
