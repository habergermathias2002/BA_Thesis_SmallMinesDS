"""
02_extract_bono_test_patches.py
================================

Kurz: Schneidet aus dem Bono-Mosaik eine 5×5 km Testfläche um einen bekannten
Galamsey-Standort aus, skaliert die Pixelwerte mit ×10.000 (Rückgängigmachen der
GEE-Normalisierung), unterteilt in 128×128-Patches und speichert sie als GeoTIFFs
plus patch_index.csv. Diese Patches sind die Eingabe für die Inferenz (Skript 04/05).

Test area center: lat=8.054635, lon=-2.025502 (WGS84 / EPSG:4326)
Mosaic CRS:       EPSG:32630 (UTM Zone 30N)

What it does:
  1. Converts the WGS84 center coordinates to UTM
  2. Reads a 5×5 km window from Bono_Merged_2025.tif (no full file load)
  3. Selects the 6 bands needed by the model (already in correct order in the mosaic)
  4. Rescales pixel values × 10,000:
       Bono data was exported from GEE with /10000 applied → values 0–1
       SmallMinesDS training data has raw DN values → values 0–10,000
       The model was trained on the raw scale, so we undo the GEE normalization.
  5. Pads the area to the next multiple of 128 (adds black border)
  6. Saves each 128×128 patch as a GeoTIFF with correct georeferencing
  7. Saves a patch index CSV (for reassembly after inference)

Output:
  data/patches_bono_test/
      patch_0000_r0_c0.tif      ← 6-band GeoTIFF, 128×128, raw DN scale
      patch_0001_r0_c1.tif
      ...
      patch_index.csv            ← row, col, UTM bounds per patch

Usage:
  python 00_Mathias_contribution/scripts/02_extract_bono_test_patches.py
  (run from repo root)
"""

import os
import csv
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import from_origin
from pyproj import Transformer

# ── Configuration ──────────────────────────────────────────────────────────────
CENTER_LAT  =  8.054635   # WGS84 latitude of test area center
CENTER_LON  = -2.025502   # WGS84 longitude of test area center
AREA_M      =  5000       # Side length of test area in meters (5 km × 5 km)
PATCH_SIZE  =  128        # Pixels per patch side (must match model input)

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MOSAIC_PATH = os.path.join(REPO_ROOT, "data", "raw", "Bono_Merged_2025.tif")
OUT_DIR     = os.path.join(REPO_ROOT, "data", "patches_bono_test")

# Bands to select from the mosaic (0-indexed):
# Bono mosaic band order: B2(0), B3(1), B4(2), B8A(3), B11(4), B12(5)
# These match the training bands: BLUE, GREEN, RED, VNIR_5, SWIR_1, SWIR_2
BAND_INDICES = [1, 2, 3, 4, 5, 6]  # rasterio uses 1-based band indices

# Normalization info (for reference — NOT applied here, done in inference script)
# Model expects raw DN values (0–10,000), normalization happens inside the model.
MEANS = [1473.81, 1703.35, 1696.68, 3832.40, 3156.11, 2226.07]
STDS  = [ 223.44,  285.54,  413.82,  389.61,  451.50,  468.27]


def latlon_to_utm(lat, lon):
    """Convert WGS84 lat/lon to UTM Zone 30N (EPSG:32630) easting/northing."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32630", always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return easting, northing


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Step 1: Convert center to UTM
    center_e, center_n = latlon_to_utm(CENTER_LAT, CENTER_LON)
    half = AREA_M / 2
    left   = center_e - half
    right  = center_e + half
    bottom = center_n - half
    top    = center_n + half

    print(f"Test area center (UTM 30N): E={center_e:.1f}, N={center_n:.1f}")
    print(f"Bounding box: ({left:.0f}, {bottom:.0f}) → ({right:.0f}, {top:.0f})")
    print(f"Area: {AREA_M/1000:.1f} × {AREA_M/1000:.1f} km")

    # Step 2: Read 5×5 km window from mosaic
    with rasterio.open(MOSAIC_PATH) as src:
        print(f"\nMosaic CRS: {src.crs}, resolution: {src.res[0]:.1f} m/px")

        window = from_bounds(left, bottom, right, top, transform=src.transform)
        data   = src.read(BAND_INDICES, window=window)  # shape: (6, H, W)
        win_transform = src.window_transform(window)

    actual_h, actual_w = data.shape[1], data.shape[2]
    print(f"Read window: {actual_w} × {actual_h} px  "
          f"({actual_w * src.res[0] / 1000:.2f} × {actual_h * src.res[0] / 1000:.2f} km)")

    # Step 3: Rescale × 10,000 (undo GEE /10000 normalization)
    print("\nRescaling pixel values × 10,000 (0–1 → 0–10,000)...")
    data = data.astype(np.float32) * 10000.0
    print(f"Value range after rescaling: min={np.nanmin(data):.1f}, max={np.nanmax(data):.1f}")

    # Step 4: Pad to multiple of 128
    pad_h = (PATCH_SIZE - actual_h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - actual_w % PATCH_SIZE) % PATCH_SIZE
    if pad_h > 0 or pad_w > 0:
        data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
        print(f"Padded to {data.shape[2]} × {data.shape[1]} px "
              f"(added {pad_w} cols, {pad_h} rows of zeros)")

    padded_h, padded_w = data.shape[1], data.shape[2]
    n_rows = padded_h // PATCH_SIZE
    n_cols = padded_w // PATCH_SIZE
    print(f"\nExtracting {n_rows} × {n_cols} = {n_rows * n_cols} patches of {PATCH_SIZE}×{PATCH_SIZE} px...")

    # Step 5: Extract and save patches
    patch_records = []
    patch_idx = 0

    for r in range(n_rows):
        for c in range(n_cols):
            row_start = r * PATCH_SIZE
            col_start = c * PATCH_SIZE
            patch = data[:, row_start:row_start + PATCH_SIZE, col_start:col_start + PATCH_SIZE]

            # Compute georeferenced transform for this patch
            patch_transform = rasterio.transform.Affine(
                win_transform.a,
                win_transform.b,
                win_transform.c + col_start * win_transform.a,
                win_transform.d,
                win_transform.e,
                win_transform.f + row_start * win_transform.e,
            )

            # Save as GeoTIFF
            filename = f"patch_{patch_idx:04d}_r{r}_c{c}.tif"
            filepath = os.path.join(OUT_DIR, filename)

            with rasterio.open(
                filepath, "w",
                driver="GTiff",
                height=PATCH_SIZE, width=PATCH_SIZE,
                count=patch.shape[0],
                dtype=patch.dtype,
                crs="EPSG:32630",
                transform=patch_transform,
            ) as dst:
                dst.write(patch)

            patch_records.append({
                "patch_file": filename,
                "row": r,
                "col": c,
                "row_px_start": row_start,
                "col_px_start": col_start,
                "utm_left":   win_transform.c + col_start * win_transform.a,
                "utm_top":    win_transform.f + row_start * win_transform.e,
            })
            patch_idx += 1

    # Step 6: Save patch index CSV
    index_path = os.path.join(OUT_DIR, "patch_index.csv")
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=patch_records[0].keys())
        writer.writeheader()
        writer.writerows(patch_records)

    print(f"\nSaved {patch_idx} patches to: {OUT_DIR}")
    print(f"Patch index saved to: {index_path}")
    print(f"\nNext step: Train the model (03_train_colab.py), then run 04_inference_bono.py.")


if __name__ == "__main__":
    main()
