"""
06_ghana_map_galamsey_bono.py
==============================

Kurz: Erzeugt eine Übersichtskarte von Ghana (weißer Hintergrund) mit Landes- und
Regionsgrenzen (GADM). Die Modell-Vorhersage für die Bono-Region (aus Skript 04 oder
05) wird als rote Überlagerung (Mining-Wahrscheinlichkeit) eingeblendet. Ausgabe:
data/ghana_map_galamsey_bono.png. Bei sehr großen Rastern wird für die Karte
automatisch heruntergerechnet, um Speicher zu sparen.

Requirements: geopandas, matplotlib, rasterio (and dependencies).
  pip install geopandas matplotlib rasterio

Usage:
  python 00_Mathias_contribution/scripts/06_ghana_map_galamsey_bono.py
  (run from repo root)

Output:
  data/ghana_map_galamsey_bono.png  (and optionally .pdf)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Prediction raster: prefer full Bono inference, else test region
# Use PROB raster so we can show low probabilities (model often predicts all ~0; prob still shows structure)
PRED_PROB_FULL = os.path.join(REPO_ROOT, "data", "inference_bono_full", "prediction_prob.tif")
PRED_PROB_TEST = os.path.join(REPO_ROOT, "data", "patches_bono_test", "prediction_prob.tif")
OUT_MAP_PATH = os.path.join(REPO_ROOT, "data", "ghana_map_galamsey_bono.png")
# Show red where Mining prob > 0; alpha scaled so even tiny probs are visible
PROB_THRESH = 0.0

# GADM Ghana level 1 (regions) – GeoPackage URL
GADM_GHA_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_GHA.gpkg"


def get_ghana_regions():
    """Load Ghana regions (level 1) from GADM. Uses cache if already downloaded."""
    import geopandas as gpd
    cache_dir = os.path.join(REPO_ROOT, "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    local_gpkg = os.path.join(cache_dir, "gadm41_GHA.gpkg")
    if not os.path.exists(local_gpkg):
        try:
            import urllib.request
            print("Downloading GADM Ghana boundaries (once)...")
            urllib.request.urlretrieve(GADM_GHA_URL, local_gpkg)
        except Exception as e:
            raise FileNotFoundError(
                "Could not download GADM Ghana. Get manually from "
                "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_GHA.gpkg "
                f"and save as {local_gpkg}. Error: {e}"
            ) from e
    # Layer 1 = admin level 1 (regions: Bono, Ashanti, ...)
    try:
        gdf = gpd.read_file(local_gpkg, layer=1)
    except Exception:
        gdf = gpd.read_file(local_gpkg, layer=0)  # fallback: country only
    return gdf


# Max pixels per dimension for map overlay (avoids OOM on large rasters)
MAX_MAP_PX = 2000


def reproject_prob_to_wgs84(prob_path):
    """Read probability raster (UTM) and reproject to WGS84; returns (data, extent).
    Downsampled if larger than MAX_MAP_PX to avoid memory kill."""
    from rasterio.warp import transform_bounds
    with rasterio.open(prob_path) as src:
        dst_crs = "EPSG:4326"
        bounds_wgs84 = transform_bounds(src.crs, dst_crs, *src.bounds)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        # Downsample if too large (full inference raster can be 30k+ px)
        if width > MAX_MAP_PX or height > MAX_MAP_PX:
            scale = min(MAX_MAP_PX / width, MAX_MAP_PX / height)
            width = max(1, int(width * scale))
            height = max(1, int(height * scale))
            transform = rasterio.transform.from_bounds(
                bounds_wgs84[0], bounds_wgs84[1], bounds_wgs84[2], bounds_wgs84[3],
                width, height,
            )
        out = np.zeros((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
        extent = (bounds_wgs84[0], bounds_wgs84[2], bounds_wgs84[1], bounds_wgs84[3])
    return out, extent


def main():
    # 1) Prediction probability raster path
    if os.path.exists(PRED_PROB_FULL):
        pred_path = PRED_PROB_FULL
        title_suffix = "Bono region (full inference)"
    elif os.path.exists(PRED_PROB_TEST):
        pred_path = PRED_PROB_TEST
        title_suffix = "Bono test area (5×5 km)"
    else:
        print("No prediction raster found. Run 04_inference_bono.py or 05_inference_bono_full.py first.")
        return

    # 2) Load Ghana regions (WGS84)
    gdf = get_ghana_regions()
    gdf = gdf.to_crs("EPSG:4326")

    # 3) Reproject Bono probability to WGS84
    prob_raster, extent = reproject_prob_to_wgs84(pred_path)
    p_min, p_max = float(np.nanmin(prob_raster)), float(np.nanmax(prob_raster))
    print(f"Mining probability in raster: min={p_min:.2e}, max={p_max:.2e}")

    # 4) Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 12), facecolor="white")
    ax.set_facecolor("white")

    # Ghana: white fill, black/gray boundaries
    gdf.plot(ax=ax, facecolor="white", edgecolor="black", linewidth=0.7)

    # Region labels (Bono, Ashanti, ...)
    name_col = "NAME_1" if "NAME_1" in gdf.columns else next(
        (c for c in gdf.columns if "name" in c.lower()), None
    )
    if name_col:
        for idx, row in gdf.iterrows():
            centroid = row.geometry.centroid
            if centroid.is_empty:
                continue
            ax.annotate(
                str(row[name_col]),
                (centroid.x, centroid.y),
                fontsize=7,
                ha="center",
                va="center",
                color="gray",
            )

    # Overlay: Mining probability in red. Alpha so even tiny probs are visible
    ref = max(p_max, 1e-9)
    alpha = np.clip(0.25 + 0.65 * (prob_raster / ref), 0, 0.9)
    alpha[prob_raster <= PROB_THRESH] = 0
    rgba = np.zeros((*prob_raster.shape, 4))
    rgba[..., 0] = 1
    rgba[..., 1] = 0
    rgba[..., 2] = 0
    rgba[..., 3] = alpha
    ax.imshow(
        rgba,
        extent=extent,
        origin="upper",
        interpolation="nearest",
        zorder=5,
    )
    # If no red pixels, at least show the prediction area as a red outline
    if p_max <= 1e-9 or np.sum(alpha > 0) == 0:
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (extent[0], extent[2]), extent[1] - extent[0], extent[3] - extent[2],
            linewidth=2, edgecolor="red", facecolor="none", linestyle="--", zorder=6,
        )
        ax.add_patch(rect)
        ax.set_title(ax.get_title() + " (no Mining predicted; area outlined)")

    ax.set_xlim(gdf.total_bounds[0] - 0.5, gdf.total_bounds[2] + 0.5)
    ax.set_ylim(gdf.total_bounds[1] - 0.5, gdf.total_bounds[3] + 0.5)
    ax.set_aspect("equal")
    ax.set_title(f"Ghana – Galamsey (model prediction) in {title_suffix}", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT_MAP_PATH), exist_ok=True)
    plt.savefig(OUT_MAP_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Map saved: {OUT_MAP_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
