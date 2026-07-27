"""
Erzeugt für jedes manuelle Bono-Label ein PNG: Satellitenbild | Ground Truth.

Keine Inferenz — nur Visualisierung der Labels auf dem Bono-2025-Mosaik.

Ausgabe: 00_Mathias_contribution/Label_GT_Preview/patches/
"""
import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio import features as rio_features
from shapely.geometry import shape, mapping, box
from shapely.validation import make_valid
import fiona
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Prefer newest export (test.gpkg), fallback to earlier filename
_labels_dir = os.path.join(REPO_ROOT, "00_Mathias_contribution", "labels_incoming")
_candidates = [
    os.path.join(_labels_dir, "test.gpkg"),
    os.path.join(_labels_dir, "galamsey_labels_bono.gpkg"),
]
LABELS_GPKG = next((p for p in _candidates if os.path.isfile(p)), _candidates[0])
MOSAIC_PATH = os.path.join(REPO_ROOT, "data", "raw", "Bono_Merged_2025.tif")
OUT_DIR = os.path.join(REPO_ROOT, "00_Mathias_contribution", "Label_GT_Preview")
PATCHES_DIR = os.path.join(OUT_DIR, "patches")

PATCH_SIZE = 128  # px @ 10 m
R_IDX, G_IDX, B_IDX = 2, 1, 0  # True Color: B4, B3, B2


def fix_geom(geom):
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        geom = polys[0] if polys else geom
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    return geom


def class_int(raw):
    """None → Non-Mining (0), 1 → Mining."""
    return 0 if raw is None else int(raw)


def extract_patch(src, cx, cy, patch_size=PATCH_SIZE):
    res = src.res[0]
    half_m = (patch_size / 2) * res
    window = from_bounds(cx - half_m, cy - half_m, cx + half_m, cy + half_m, transform=src.transform)
    col_off = int(round(window.col_off))
    row_off = int(round(window.row_off))
    window = Window(col_off, row_off, patch_size, patch_size)
    data = src.read(window=window).astype(np.float32)
    transform = src.window_transform(window)
    if data.shape[1] != patch_size or data.shape[2] != patch_size:
        padded = np.full((6, patch_size, patch_size), np.nan, dtype=np.float32)
        h, w = data.shape[1], data.shape[2]
        padded[:, :h, :w] = data
        data = padded
    return data, transform


def truecolor(img6):
    rgb = np.dstack((img6[R_IDX], img6[G_IDX], img6[B_IDX])).astype(np.float32)
    valid = np.isfinite(rgb).all(axis=2)
    if valid.any():
        lo, hi = np.nanpercentile(rgb[valid], (2, 98))
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.nan_to_num(rgb, nan=0.0)


def rasterize_gt(geom, class_val, transform, out_shape):
    mask = np.zeros(out_shape, dtype=np.uint8)
    if class_val == 1 and not geom.is_empty:
        rio_features.rasterize(
            [(mapping(geom), 1)],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            out=mask,
        )
    return mask


def geom_to_pixel_ring(geom, transform):
    if geom is None or geom.is_empty or not hasattr(geom, "exterior"):
        return []
    xs, ys = geom.exterior.xy
    a, c, e, f = transform.a, transform.c, transform.e, transform.f
    return list(zip([(x - c) / a for x in xs], [(y - f) / e for y in ys]))


def render(idx, class_val, rgb, gt, geom, transform, out_path):
    label_name = "Mining" if class_val == 1 else "Non-Mining"
    gt_pct = float(gt.mean() * 100)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    fig.suptitle(
        f"Label {idx:02d}  |  Ground Truth: {label_name}",
        fontsize=12,
        y=1.02,
    )
    cmap_mask = mcolors.ListedColormap(["white", "red"])

    axes[0].imshow(rgb, interpolation="nearest")
    ring = geom_to_pixel_ring(geom, transform)
    if ring:
        axes[0].add_patch(
            MplPolygon(
                ring,
                closed=True,
                fill=False,
                edgecolor="yellow" if class_val == 1 else "cyan",
                linewidth=1.5,
            )
        )
    axes[0].set_xlim(0, PATCH_SIZE - 1)
    axes[0].set_ylim(PATCH_SIZE - 1, 0)
    axes[0].set_title("1. Satellitenbild\n(+ Label-Umriss)", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(gt, cmap=cmap_mask, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(
        f"2. Ground Truth\n{label_name} ({gt_pct:.1f}% Mining-Pixel)",
        fontsize=9,
    )
    axes[1].axis("off")

    handles = [
        mpatches.Patch(color="white", ec="gray", label="Non-Mining"),
        mpatches.Patch(color="red", label="Mining"),
        mpatches.Patch(facecolor="none", ec="yellow", label="Mining-Umriss"),
        mpatches.Patch(facecolor="none", ec="cyan", label="Non-Mining-Umriss"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(PATCHES_DIR, exist_ok=True)

    with fiona.open(LABELS_GPKG, layer="galamsey_bono") as labels, rasterio.open(MOSAIC_PATH) as mosaic:
        print(f"Labels: {len(labels)} | Mosaic: {mosaic.width}x{mosaic.height}")
        results = []
        for idx, feat in enumerate(labels):
            class_val = class_int(feat["properties"].get("class"))
            geom = fix_geom(shape(feat["geometry"]))
            cx, cy = geom.centroid.x, geom.centroid.y

            half_m = (PATCH_SIZE / 2) * mosaic.res[0]
            patch_box = box(cx - half_m, cy - half_m, cx + half_m, cy + half_m)
            try:
                geom_in_patch = geom.intersection(patch_box)
            except Exception:
                geom_in_patch = make_valid(geom).intersection(patch_box)
            geom_in_patch = fix_geom(geom_in_patch) if not geom_in_patch.is_empty else geom_in_patch

            data_01, transform = extract_patch(mosaic, cx, cy)
            rgb = truecolor(data_01)
            gt = rasterize_gt(geom_in_patch, class_val, transform, (PATCH_SIZE, PATCH_SIZE))

            tag = "mining" if class_val == 1 else "nonmining"
            out_name = f"label_{idx:02d}_{tag}.png"
            out_path = os.path.join(PATCHES_DIR, out_name)

            outline = geom_in_patch if not geom_in_patch.is_empty else geom
            outline = fix_geom(outline) if not outline.is_empty else outline
            render(idx, class_val, rgb, gt, outline, transform, out_path)

            results.append((idx, class_val, float(gt.mean() * 100), out_name))
            print(f"  ✓ {out_name}: {('Mining' if class_val==1 else 'Non-Mining')}")

    n_m = sum(1 for r in results if r[1] == 1)
    n_n = sum(1 for r in results if r[1] == 0)
    with open(os.path.join(OUT_DIR, "README.md"), "w") as f:
        f.write("# Label GT Preview (Bono 2025)\n\n")
        f.write("Satellitenbild + Ground Truth für alle manuellen Labels. **Keine Inferenz.**\n\n")
        f.write(f"- Mining: **{n_m}**\n- Non-Mining: **{n_n}**\n- Total: **{len(results)}**\n\n")
        f.write("| Datei | GT | Mining-Pixel im Patch |\n|---|---|---|\n")
        for idx, class_val, gt_pct, name in results:
            gt = "Mining" if class_val == 1 else "Non-Mining"
            f.write(f"| `{name}` | {gt} | {gt_pct:.1f}% |\n")
    print(f"\nFertig: {n_m} Mining + {n_n} Non-Mining → {PATCHES_DIR}")


if __name__ == "__main__":
    main()
