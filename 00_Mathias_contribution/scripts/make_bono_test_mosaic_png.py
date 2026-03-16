"""
Stitch all 16 Bono test patches into one 512×512 true-color PNG.
Output: data/patches_bono_test/full_test_area_truecolor.png
"""
import os
import csv
import numpy as np
import rasterio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATCHES_DIR = os.path.join(REPO_ROOT, "data", "patches_bono_test")
PATCH_SIZE = 128
OUT_SIZE = 4 * PATCH_SIZE  # 512

# Band indices 0-based: 0=Blue, 1=Green, 2=Red → true color
R, G, B = 2, 1, 0


def main():
    index_path = os.path.join(PATCHES_DIR, "patch_index.csv")
    with open(index_path) as f:
        records = list(csv.DictReader(f))

    rgb = np.zeros((OUT_SIZE, OUT_SIZE, 3), dtype=np.float32)
    for rec in records:
        path = os.path.join(PATCHES_DIR, rec["patch_file"])
        row_s = int(rec["row_px_start"])
        col_s = int(rec["col_px_start"])
        with rasterio.open(path) as src:
            patch = src.read()  # (6, 128, 128)
        rgb[row_s : row_s + PATCH_SIZE, col_s : col_s + PATCH_SIZE, 0] = patch[R]
        rgb[row_s : row_s + PATCH_SIZE, col_s : col_s + PATCH_SIZE, 1] = patch[G]
        rgb[row_s : row_s + PATCH_SIZE, col_s : col_s + PATCH_SIZE, 2] = patch[B]

    # Stretch to 0–255 for PNG (percentile to avoid outliers)
    lo, hi = np.nanpercentile(rgb, (2, 98))
    rgb = np.clip((rgb.astype(np.float64) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

    import matplotlib.pyplot as plt
    out_path = os.path.join(PATCHES_DIR, "full_test_area_truecolor.png")
    plt.imsave(out_path, rgb)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
