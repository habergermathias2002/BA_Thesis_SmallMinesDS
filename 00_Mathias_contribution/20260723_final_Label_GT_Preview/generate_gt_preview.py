"""
2026-07-23 Final Labels — GT Preview (Satellitenbild | Ground Truth).

Wichtig: Ground Truth enthält ALLE Mining-Polygone, die das Patch-Fenster
schneiden — nicht nur das einzelne Fokus-Polygon. So sind mehrere Polygone
auf demselben Ausschnitt (z.B. Labels 003–008) in der Maske vollständig.

Quelle: labels_incoming/richtige labels 23..gpkg (Layer: richtige_labels_v1)
Attribut: Mining-y-n (1=Mining, 0=Non-Mining)
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
LABELS_GPKG = os.path.join(
    REPO_ROOT, "00_Mathias_contribution", "labels_incoming", "richtige labels 23..gpkg"
)
LABELS_LAYER = "richtige_labels_v1"
CLASS_FIELD = "Mining-y-n"
MOSAIC_PATH = os.path.join(REPO_ROOT, "data", "raw", "Bono_Merged_2025.tif")
PATCHES_DIR = os.path.join(HERE, "patches")

PATCH_SIZE = 128
R_IDX, G_IDX, B_IDX = 2, 1, 0


def fix_geom(geom):
    if geom is None or geom.is_empty:
        return geom
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        geom = polys[0] if polys else geom
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    return geom


def class_int(props):
    raw = (props or {}).get(CLASS_FIELD)
    if raw is None:
        raw = (props or {}).get("class")
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
    return data, transform, half_m


def truecolor(img6):
    rgb = np.dstack((img6[R_IDX], img6[G_IDX], img6[B_IDX])).astype(np.float32)
    valid = np.isfinite(rgb).all(axis=2)
    if valid.any():
        lo, hi = np.nanpercentile(rgb[valid], (2, 98))
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.nan_to_num(rgb, nan=0.0)


def rasterize_all_mining(geoms, transform, out_shape):
    """Burn ALL mining polygons that fall in the patch into one mask."""
    mask = np.zeros(out_shape, dtype=np.uint8)
    shapes = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        g = fix_geom(g)
        if g is None or g.is_empty:
            continue
        shapes.append((mapping(g), 1))
    if shapes:
        rio_features.rasterize(
            shapes,
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


def render(idx, class_val, rgb, gt, focus_geom, all_mining_geoms, transform, out_path, n_polys):
    label_name = "Mining" if class_val == 1 else "Non-Mining"
    gt_pct = float(gt.mean() * 100)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    title = f"Label {idx:03d}  |  Ground Truth: {label_name}"
    if class_val == 1 and n_polys > 1:
        title += f"  |  {n_polys} Polygone im Patch"
    fig.suptitle(title, fontsize=11, y=1.02)
    cmap_mask = mcolors.ListedColormap(["white", "red"])

    axes[0].imshow(rgb, interpolation="nearest")
    # Other mining polygons: orange thin
    for g in all_mining_geoms:
        ring = geom_to_pixel_ring(fix_geom(g), transform)
        if ring:
            axes[0].add_patch(
                MplPolygon(ring, closed=True, fill=False, edgecolor="orange", linewidth=1.0, alpha=0.9)
            )
    # Focus polygon: yellow thick
    focus = fix_geom(focus_geom) if focus_geom is not None else None
    ring = geom_to_pixel_ring(focus, transform)
    if ring:
        color = "yellow" if class_val == 1 else "cyan"
        axes[0].add_patch(
            MplPolygon(ring, closed=True, fill=False, edgecolor=color, linewidth=2.0)
        )
    axes[0].set_xlim(0, PATCH_SIZE - 1)
    axes[0].set_ylim(PATCH_SIZE - 1, 0)
    axes[0].set_title("1. Satellitenbild\n(gelb=Fokus, orange=weitere Mining-Polygone)", fontsize=8)
    axes[0].axis("off")

    axes[1].imshow(gt, cmap=cmap_mask, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(
        f"2. Ground Truth (alle Polygone)\n{label_name} ({gt_pct:.1f}% Mining-Pixel)",
        fontsize=9,
    )
    axes[1].axis("off")

    handles = [
        mpatches.Patch(color="white", ec="gray", label="Non-Mining"),
        mpatches.Patch(color="red", label="Mining"),
        mpatches.Patch(facecolor="none", ec="yellow", label="Fokus-Polygon"),
        mpatches.Patch(facecolor="none", ec="orange", label="Weitere Mining-Polygone"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def load_all_features():
    feats = []
    with fiona.open(LABELS_GPKG, layer=LABELS_LAYER) as src:
        for i, feat in enumerate(src):
            class_val = class_int(feat["properties"])
            geom = fix_geom(shape(feat["geometry"]))
            feats.append({"idx": i, "class": class_val, "geom": geom})
    return feats


def main():
    os.makedirs(PATCHES_DIR, exist_ok=True)
    feats = load_all_features()
    mining_geoms = [f["geom"] for f in feats if f["class"] == 1]
    print(f"Features: {len(feats)} | Mining polygons: {len(mining_geoms)}")

    with rasterio.open(MOSAIC_PATH) as mosaic:
        results = []
        for feat in feats:
            idx = feat["idx"]
            class_val = feat["class"]
            geom = feat["geom"]
            cx, cy = geom.centroid.x, geom.centroid.y

            data_01, transform, half_m = extract_patch(mosaic, cx, cy)
            patch_box = box(cx - half_m, cy - half_m, cx + half_m, cy + half_m)

            # ALL mining polygons intersecting this patch window
            polys_in_patch = [
                g for g in mining_geoms if g is not None and not g.is_empty and g.intersects(patch_box)
            ]
            # clip to patch for cleaner rasterize/outline
            clipped = []
            for g in polys_in_patch:
                try:
                    inter = fix_geom(g.intersection(patch_box))
                except Exception:
                    inter = fix_geom(make_valid(g).intersection(patch_box))
                if inter is not None and not inter.is_empty:
                    clipped.append(inter)

            rgb = truecolor(data_01)
            if class_val == 1:
                gt = rasterize_all_mining(clipped, transform, (PATCH_SIZE, PATCH_SIZE))
            else:
                gt = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)

            tag = "mining" if class_val == 1 else "nonmining"
            out_name = f"label_{idx:03d}_{tag}.png"
            out_path = os.path.join(PATCHES_DIR, out_name)

            try:
                focus_clip = fix_geom(geom.intersection(patch_box))
            except Exception:
                focus_clip = geom
            if focus_clip is None or focus_clip.is_empty:
                focus_clip = geom

            render(
                idx, class_val, rgb, gt, focus_clip, clipped, transform, out_path, len(clipped)
            )
            results.append((idx, class_val, float(gt.mean() * 100), out_name, len(clipped)))
            extra = f", polys_in_patch={len(clipped)}" if class_val == 1 else ""
            print(f"  ✓ {out_name} (GT mining={gt.mean()*100:.1f}%{extra})")

    # Keep random nonmining (122+) untouched; recount
    n_mining_png = len([f for f in os.listdir(PATCHES_DIR) if f.endswith("_mining.png")])
    n_non_png = len([f for f in os.listdir(PATCHES_DIR) if f.endswith("_nonmining.png")])
    multi = sum(1 for r in results if r[1] == 1 and r[4] > 1)

    with open(os.path.join(HERE, "README.md"), "w") as f:
        f.write("# 2026-07-23 Final Label GT Preview\n\n")
        f.write("Finale Labels aus `richtige labels 23..gpkg`.\n\n")
        f.write(
            "**GT-Logik:** Alle Mining-Polygone, die das 128×128-Fenster schneiden, "
            "werden in die Maske geschrieben (nicht nur das Fokus-Polygon).\n\n"
        )
        f.write(f"- Manuelle Mining-Labels: **{sum(1 for r in results if r[1]==1)}**\n")
        f.write(f"- Manuelle Non-Mining-Labels: **{sum(1 for r in results if r[1]==0)}**\n")
        f.write(f"- Mining-Labels mit >1 Polygon im Patch: **{multi}**\n")
        f.write(f"- PNGs gesamt im Ordner: Mining **{n_mining_png}**, Non-Mining **{n_non_png}**\n\n")
        f.write("| Datei | Fokus | Polygone im Patch | Mining-Pixel |\n|---|---|---|---|\n")
        for idx, class_val, gt_pct, name, n_polys in results:
            fokus = "Mining" if class_val == 1 else "Non-Mining"
            f.write(f"| `{name}` | {fokus} | {n_polys} | {gt_pct:.1f}% |\n")

    print(f"\nFertig. Multi-polygon patches: {multi}")
    print(f"PNG counts: mining={n_mining_png}, nonmining={n_non_png}")


if __name__ == "__main__":
    main()
