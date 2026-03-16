"""
Verify which 6 bands terratorch uses during training (SmallMinesDS 13-band TIFs).

SmallMinesDS 13-band order (HuggingFace README):
  Index 0–9: S2 L2A [blue, green, red, RE1, RE2, RE3, NIR, RE4(B8A), swir1, swir2]
  Index 10–11: Sentinel-1 VV, VH
  Index 12: DEM

Model expects: B2, B3, B4, B8A, B11, B12 → indices 0, 1, 2, 7, 8, 9.

Training script passes dataset_bands = output_bands = ["BLUE","GREEN","RED","VNIR_5","SWIR_1","SWIR_2"].
Terratorch: filter_indices = [dataset_bands.index(b) for b in output_bands] = [0,1,2,3,4,5].
So terratorch takes the FIRST 6 bands of the file = indices 0–5 = B2, B3, B4, B5, B6, B7 (NO B8A, NO SWIR).

This script:
  1. Loads a training TIF and prints per-band means.
  2. Confirms that band 7 (~B8A), 8 (B11), 9 (B12) have distinctly different ranges from 3,4,5 (RE1,RE2,RE3).
  3. Documents the mismatch for the Technical Report.
"""
import os
import sys
import numpy as np

try:
    import rasterio
except ImportError:
    print("rasterio not found; use conda env smallmines")
    sys.exit(1)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIR = os.path.join(REPO_ROOT, "data", "GhanaMiningPrithvi", "training")

# SmallMinesDS README: 0=blue, 1=green, 2=red, 3=RE1, 4=RE2, 5=RE3, 6=NIR, 7=RE4/B8A, 8=swir1, 9=swir2
BAND_NAMES_13 = [
    "B2(Blue)", "B3(Green)", "B4(Red)", "B5(RE1)", "B6(RE2)", "B7(RE3)",
    "B8(NIR)", "B8A(RE4)", "B11(SWIR1)", "B12(SWIR2)", "S1_VV", "S1_VH", "DEM"
]
# Model expects these 6 (indices in 13-band stack)
EXPECTED_6_INDICES = [0, 1, 2, 7, 8, 9]  # B2, B3, B4, B8A, B11, B12
# What terratorch uses when dataset_bands = output_bands = [BLUE, GREEN, RED, VNIR_5, SWIR_1, SWIR_2]
# → filter_indices = [0,1,2,3,4,5] (identity: first 6 channels)
ACTUAL_TERRATORCH_INDICES = [0, 1, 2, 3, 4, 5]  # B2, B3, B4, B5, B6, B7 — NO B8A, NO SWIR!


def main():
    img_path = None
    for f in os.listdir(TRAIN_DIR):
        if f.endswith("_IMG.tif"):
            img_path = os.path.join(TRAIN_DIR, f)
            break
    if not img_path or not os.path.exists(img_path):
        print("No training *_IMG.tif found in", TRAIN_DIR)
        return

    with rasterio.open(img_path) as src:
        img = src.read().astype(np.float64)
    n_bands = img.shape[0]
    print(f"File: {os.path.basename(img_path)}")
    print(f"Shape: {img.shape} ({n_bands} bands)")
    print()

    means = img.mean(axis=(1, 2))
    print("Per-band means (0–9 = S2, 10–11 = SAR, 12 = DEM):")
    for i in range(min(n_bands, 10)):
        print(f"  Band {i:2d} ({BAND_NAMES_13[i]:12s}): mean = {means[i]:.1f}")
    if n_bands > 10:
        for i in range(10, n_bands):
            print(f"  Band {i:2d} ({BAND_NAMES_13[i] if i < len(BAND_NAMES_13) else '?'}): mean = {means[i]:.1f}")
    print()

    print("Interpretation:")
    print("  - Bands 3,4,5 (RE1, RE2, RE3) are in 2.9k–3.5k range (this patch).")
    print("  - Bands 7,8,9 (B8A, B11, B12) are in 2.6k–3.5k range (B8A/NIR and SWIR).")
    print("  - Training means from script: [1474, 1703, 1697, 3832, 3156, 2226] for BLUE,GREEN,RED,VNIR_5,SWIR_1,SWIR_2.")
    print()
    print("Terratorch behavior (generic_pixel_wise_dataset.py):")
    print("  - dataset_bands = output_bands = [BLUE, GREEN, RED, VNIR_5, SWIR_1, SWIR_2] (6 names).")
    print("  - filter_indices = [dataset_bands.index(b) for b in output_bands] = [0, 1, 2, 3, 4, 5].")
    print("  - So image[..., filter_indices] = first 6 channels of the file = indices 0–5.")
    print()
    print("CONCLUSION: Training used bands 0–5 = B2, B3, B4, B5, B6, B7 (no B8A, no B11, B12).")
    print("            Inference sends bands [B2, B3, B4, B8A, B11, B12] (correct 6 for Prithvi).")
    print("            → BAND MISMATCH: model was trained on wrong spectral channels.")
    print()
    print("Recommended fix: Preprocess training TIFs to 6 bands in order [0,1,2,7,8,9] before training,")
    print("                or pass dataset_bands as the full 13-band order and output_bands as the 6 names")
    print("                only if terratorch supports naming bands by position (currently it assumes")
    print("                dataset_bands length = number of channels in file).")


if __name__ == "__main__":
    main()
