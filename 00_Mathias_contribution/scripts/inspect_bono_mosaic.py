"""
Liest Metadaten und Band-Infos aus Bono_Merged_2025.tif (oder einer GEE-Kachel).

Usage (Repo-Root):
  python 00_Mathias_contribution/scripts/inspect_bono_mosaic.py
  python 00_Mathias_contribution/scripts/inspect_bono_mosaic.py path/to/file.tif
"""
import os
import sys
import numpy as np
import rasterio
from rasterio.windows import Window

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PATH = os.path.join(REPO_ROOT, "data", "raw", "Bono_Merged_2025.tif")

BAND_NAMES = [
    "Band 1: B2  (Blue)",
    "Band 2: B3  (Green)",
    "Band 3: B4  (Red)",
    "Band 4: B8A (NIR / VNIR_5)",
    "Band 5: B11 (SWIR 1)",
    "Band 6: B12 (SWIR 2)",
]

QGIS_TRUE_COLOR = "Red=Band 3, Green=Band 2, Blue=Band 1"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    if not os.path.isfile(path):
        print(f"Datei nicht gefunden: {path}")
        sys.exit(1)

    print(f"Datei: {path}\n")
    with rasterio.open(path) as src:
        print(f"Driver:     {src.driver}")
        print(f"CRS:        {src.crs}")
        print(f"Size:       {src.width} x {src.height} px")
        print(f"Resolution: {src.res[0]:.1f} m/px")
        print(f"Bands:      {src.count}")
        print(f"Dtype:      {src.dtypes}")
        print(f"nodata:     {src.nodata}")
        print(f"nodatavals: {src.nodatavals}")
        print(f"Descriptions: {src.descriptions}")
        print(f"\nErwartete Band-Reihenfolge (aus GEE-Export + Notebook):")
        for name in BAND_NAMES:
            print(f"  {name}")
        print(f"\nQGIS True Color: {QGIS_TRUE_COLOR}")

        # Sample center window
        w = min(512, src.width), min(512, src.height)
        cx, cy = src.width // 2, src.height // 2
        data = src.read(window=Window(cx, cy, w[0], w[1]))
        print(f"\nStichprobe ({w[0]}x{w[1]} px, Mitte des Rasters):")
        for i in range(src.count):
            band = data[i]
            valid = band[~np.isnan(band)]
            if valid.size:
                print(
                    f"  {BAND_NAMES[i]}: "
                    f"min={valid.min():.6f}, max={valid.max():.6f}, mean={valid.mean():.6f}, "
                    f"NaN={np.isnan(band).sum()}"
                )
            else:
                print(f"  {BAND_NAMES[i]}: nur NaN in Stichprobe")

        nan_total = np.isnan(data).sum()
        print(f"\nHinweis NoData:")
        print("  - Im GeoTIFF-Header ist nodata=None gesetzt.")
        print("  - Maskierte GEE-Pixel sind inhaltlich NaN (float32), nicht 0 oder -9999.")
        print("  - Gültige Pixel: Float32, Reflektanz 0–1 (GEE hat durch 10.000 geteilt).")
        print("  - Für Modell-Inferenz: ×10.000 → DN-Skala 0–10.000 (siehe 02_extract_bono_test_patches.py).")


if __name__ == "__main__":
    main()
