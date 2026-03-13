# Progress Log – Mathias' Contribution (BA Thesis)

This document summarizes the steps completed so far as part of the personal contribution
to the replication and extension of the **SmallMinesDS** paper for the Bono/Bono-East region
of Ghana (January 2025).

---

## Step 1 – Data Acquisition via Google Earth Engine

**File:** `00_Mathias_contribution/GEE_data_Export_Bono_Bono-East_Region.js`

A Google Earth Engine (GEE) script was written to export a cloud-free Sentinel-2 mosaic
of the Bono and Bono-East administrative regions.

**Key decisions:**
- **Time window:** January 2025 — Ghana's dry season (Harmattan), which minimizes cloud
  cover and maximizes spectral contrast between bare mining soil and surrounding vegetation.
- **Cloud masking:** Per-pixel cloud filtering using the `QA60` quality band (bits 10 & 11
  for thick clouds and cirrus), retaining only scenes with < 10% cloud coverage.
- **Compositing:** Pixel-wise median across all valid acquisitions in January, eliminating
  remaining outliers (e.g., residual shadows).
- **Band selection:** 6 bands matching the Prithvi-EO model's expected input —
  B2 (Blue), B3 (Green), B4 (Red), B8A (Narrow NIR), B11 (SWIR1), B12 (SWIR2).
- **Normalization:** Values divided by 10,000 in GEE → exported TIFs contain float32 values
  in the range 0–1.
- **Projection:** EPSG:32630 (UTM Zone 30N), appropriate for West Africa.
- **Resolution:** 10 m/pixel (20 m bands B11/B12 resampled to 10 m by GEE).

The export produced 6 GeoTIFF tiles covering the full region
(GEE splits large exports automatically):

| File | Size |
|------|------|
| `Sentinel2_Bono_Januar2025_10m-0000000000-0000000000.tif` | 1.5 GB |
| `Sentinel2_Bono_Januar2025_10m-0000000000-0000013568.tif` | 2.6 GB |
| `Sentinel2_Bono_Januar2025_10m-0000000000-0000027136.tif` | 751 MB |
| `Sentinel2_Bono_Januar2025_10m-0000013568-0000000000.tif` | 1.8 GB |
| `Sentinel2_Bono_Januar2025_10m-0000013568-0000013568.tif` | 267 MB |
| `Sentinel2_Bono_Januar2025_10m-0000013568-0000027136.tif` | 539 MB |

---

## Step 2 – Scene Inventory CSV

**File:** `00_Mathias_contribution/Sentinel2_Bono_Januar2025_Szenen.csv`

As part of the same GEE script, a structured table of all Sentinel-2 scenes that
contributed to the mosaic was exported as a CSV. This provides full transparency about
which satellite acquisitions were used, allowing reproducibility and quality control.

**Columns:**

| Column | Description |
|--------|-------------|
| `scene_id` | Unique Sentinel-2 scene identifier (date + processing time + MGRS tile) |
| `date` | Acquisition date (YYYY-MM-DD) |
| `cloudy_pct` | Cloud pixel percentage reported by ESA for this scene |
| `mgrs_tile` | Military Grid Reference System tile code (e.g., `30NVP`) |
| `sensing_orbit` | Relative orbit number of the satellite pass |
| `spacecraft` | Satellite used (Sentinel-2A or Sentinel-2B) |

The CSV covers **69 scenes** from January 3–30, 2025, acquired by both Sentinel-2A and
Sentinel-2B across multiple MGRS tiles. All scenes passed the < 10% cloud filter.

---

## Step 3 – Repository Documentation

**File:** `00_Mathias_contribution/REPO_OVERVIEW.md`

A comprehensive German-language documentation of the entire repository was created,
covering:
- The scientific objective (binary semantic segmentation of ASGM mining sites)
- All four deep-learning models compared in the paper (Prithvi-EO v2 300M/600M,
  ResNet50 from scratch, ResNet50 ImageNet, SAM2-Hiera-Small)
- A file-by-file walkthrough of every script, config, and dataset detail
- Technical environment setup (two separate Conda environments: `terratorch` and `sam2`)

---

## Step 4 – Mosaic & Visualization Notebook

**File:** `00_Mathias_contribution/Notebook.ipynb`

A Jupyter Notebook was created to merge the 6 GeoTIFF tiles into a single region-wide
mosaic and visualize the result.

**What it does:**

1. **Opens** the 6 raw tiles using `rasterio` and prints basic metadata (CRS, band count,
   data type).
2. **Merges** the tiles into one mosaic using `rasterio.merge` — skipped automatically if
   `Bono_Merged_2025.tif` already exists on disk (the merge produces a ~22 GB file and
   requires ~24 GB RAM; running it twice would be wasteful).
3. **Inspects** the merged file's dimensions and resolution without loading it into RAM.
4. **Visualizes** the full region at reduced resolution by reading a downsampled version
   directly from the 22 GB file using rasterio's `out_shape` parameter (4% of original
   size → ~50 MB RAM instead of 24 GB):
   - **True color** (B4/B3/B2) — natural appearance
   - **False color** (B11/B8A/B4) — highlights bare soil and mining areas in bright
     teal/white, matching the GEE preview layer from Step 1

**Merged output:** `data/raw/Bono_Merged_2025.tif` (22 GB, 36,793 × 26,886 px, 6 bands,
float32, EPSG:32630)

---

## Next Steps (planned)

- Patch extraction from the merged mosaic into 128 × 128 px tiles matching the
  SmallMinesDS format
- Visual inspection and labeling of potential mining sites in the Bono region
- Inference with the trained Prithvi-EO model on the new region
