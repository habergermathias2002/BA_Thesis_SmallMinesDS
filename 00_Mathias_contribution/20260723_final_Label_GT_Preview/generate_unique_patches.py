"""
Unique-Patch GT Preview: ein PNG pro geografischem 128×128-Fenster.

Ground Truth = ALLE Mining-Polygone, die das Fenster schneiden.
Mehrere Polygone auf demselben Ausschnitt → nur noch 1 Preview-Bild.

Zufällige Non-Mining-Patches (label_122+) bleiben erhalten.
"""
import os
import re
import numpy as np
import rasterio
from rasterio.windows import Window
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


def truecolor(img6):
    rgb = np.dstack((img6[R_IDX], img6[G_IDX], img6[B_IDX])).astype(np.float32)
    valid = np.isfinite(rgb).all(axis=2)
    if valid.any():
        lo, hi = np.nanpercentile(rgb[valid], (2, 98))
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.nan_to_num(rgb, nan=0.0)


def geom_to_pixel_ring(geom, transform):
    if geom is None or geom.is_empty or not hasattr(geom, "exterior"):
        return []
    xs, ys = geom.exterior.xy
    a, c, e, f = transform.a, transform.c, transform.e, transform.f
    return list(zip([(x - c) / a for x in xs], [(y - f) / e for y in ys]))


def rasterize_mining(geoms, transform, out_shape):
    mask = np.zeros(out_shape, dtype=np.uint8)
    shapes = []
    for g in geoms:
        g = fix_geom(g)
        if g is None or g.is_empty:
            continue
        shapes.append((mapping(g), 1))
    if shapes:
        rio_features.rasterize(
            shapes, out_shape=out_shape, transform=transform, fill=0, dtype=np.uint8, out=mask
        )
    return mask


def window_for_centroid(transform, cx, cy, patch_size=PATCH_SIZE):
    """Snap centroid-centered patch to integer mosaic window."""
    # col/row of centroid
    # x = c + col*a ; y = f + row*e
    col_c = (cx - transform.c) / transform.a
    row_c = (cy - transform.f) / transform.e
    col_off = int(round(col_c - patch_size / 2))
    row_off = int(round(row_c - patch_size / 2))
    return col_off, row_off


def patch_box_from_window(transform, col_off, row_off, patch_size=PATCH_SIZE):
    x0 = transform.c + col_off * transform.a
    y0 = transform.f + row_off * transform.e
    x1 = transform.c + (col_off + patch_size) * transform.a
    y1 = transform.f + (row_off + patch_size) * transform.e
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)
    return box(xmin, ymin, xmax, ymax)


def render(uid, is_mining, rgb, gt, clipped_geoms, transform, member_ids, out_path):
    label_name = "Mining" if is_mining else "Non-Mining"
    gt_pct = float(gt.mean() * 100)
    members = ",".join(f"{i:03d}" for i in member_ids)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    fig.suptitle(
        f"Patch {uid:03d}  |  {label_name}  |  "
        f"{len(clipped_geoms)} Polygon(e)  |  Labels: {members}",
        fontsize=10,
        y=1.02,
    )
    cmap = mcolors.ListedColormap(["white", "red"])

    axes[0].imshow(rgb, interpolation="nearest")
    for g in clipped_geoms:
        ring = geom_to_pixel_ring(fix_geom(g), transform)
        if ring:
            axes[0].add_patch(
                MplPolygon(
                    ring, closed=True, fill=False,
                    edgecolor="yellow" if is_mining else "cyan",
                    linewidth=1.2,
                )
            )
    axes[0].set_xlim(0, PATCH_SIZE - 1)
    axes[0].set_ylim(PATCH_SIZE - 1, 0)
    axes[0].set_title("1. Satellitenbild\n(+ alle Polygone im Fenster)", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(gt, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(
        f"2. Ground Truth (ALLE Polygone)\n{label_name} ({gt_pct:.1f}% Mining-Pixel)",
        fontsize=9,
    )
    axes[1].axis("off")

    handles = [
        mpatches.Patch(color="white", ec="gray", label="Non-Mining"),
        mpatches.Patch(color="red", label="Mining"),
        mpatches.Patch(facecolor="none", ec="yellow", label="Mining-Polygone"),
        mpatches.Patch(facecolor="none", ec="cyan", label="Non-Mining-Polygone"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(PATCHES_DIR, exist_ok=True)

    # Keep random nonmining (122+)
    random_kept = []
    for fn in os.listdir(PATCHES_DIR):
        m = re.match(r"label_(\d+)_nonmining\.png$", fn)
        if m and int(m.group(1)) >= 122:
            random_kept.append(fn)
        else:
            # remove old per-polygon previews
            os.remove(os.path.join(PATCHES_DIR, fn))
    print(f"Kept {len(random_kept)} random nonmining PNGs; cleared old per-polygon previews")

    feats = []
    with fiona.open(LABELS_GPKG, layer=LABELS_LAYER) as src:
        for i, feat in enumerate(src):
            feats.append(
                {
                    "idx": i,
                    "class": class_int(feat["properties"]),
                    "geom": fix_geom(shape(feat["geometry"])),
                }
            )
    mining_geoms = [(f["idx"], f["geom"]) for f in feats if f["class"] == 1]
    print(f"Manual features: {len(feats)} | Mining polys: {len(mining_geoms)}")

    with rasterio.open(MOSAIC_PATH) as mosaic:
        # 1) Tentative window per feature (centroid-centered, snapped)
        tentative = []  # (col, row, feat)
        for f in feats:
            cx, cy = f["geom"].centroid.x, f["geom"].centroid.y
            col, row = window_for_centroid(mosaic.transform, cx, cy)
            col = max(0, min(col, mosaic.width - PATCH_SIZE))
            row = max(0, min(row, mosaic.height - PATCH_SIZE))
            tentative.append((col, row, f))

        # 2) Merge nearby windows (centroids within ~half patch) so multi-polygon
        #    sites like labels 003–008 become ONE unique patch.
        MERGE_PX = 64  # merge if window origins closer than this
        clusters = []  # list of dicts: col,row,members
        for col, row, f in tentative:
            placed = False
            for cl in clusters:
                if max(abs(cl["col"] - col), abs(cl["row"] - row)) <= MERGE_PX:
                    cl["members"].append(f)
                    # re-center on mean of member centroids
                    cxs = [m["geom"].centroid.x for m in cl["members"]]
                    cys = [m["geom"].centroid.y for m in cl["members"]]
                    ncol, nrow = window_for_centroid(
                        mosaic.transform, float(np.mean(cxs)), float(np.mean(cys))
                    )
                    cl["col"] = max(0, min(ncol, mosaic.width - PATCH_SIZE))
                    cl["row"] = max(0, min(nrow, mosaic.height - PATCH_SIZE))
                    placed = True
                    break
            if not placed:
                clusters.append({"col": col, "row": row, "members": [f]})

        print(
            f"Unique patches after merge: {len(clusters)} "
            f"(from {len(feats)} polygons, merge≤{MERGE_PX}px)"
        )

        results = []
        clusters_sorted = sorted(clusters, key=lambda c: min(m["idx"] for m in c["members"]))
        for uid, cl in enumerate(clusters_sorted):
            col, row = cl["col"], cl["row"]
            members = cl["members"]
            window = Window(col, row, PATCH_SIZE, PATCH_SIZE)
            transform = mosaic.window_transform(window)
            data = mosaic.read(window=window).astype(np.float32)
            if data.shape[1] != PATCH_SIZE or data.shape[2] != PATCH_SIZE:
                padded = np.full((6, PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float32)
                padded[:, : data.shape[1], : data.shape[2]] = data
                data = padded

            pbox = patch_box_from_window(mosaic.transform, col, row)
            # ALWAYS: all mining polygons intersecting this window
            clipped_mining = []
            for mid, g in mining_geoms:
                if g is None or g.is_empty or not g.intersects(pbox):
                    continue
                try:
                    inter = fix_geom(g.intersection(pbox))
                except Exception:
                    inter = fix_geom(make_valid(g).intersection(pbox))
                if inter is not None and not inter.is_empty:
                    clipped_mining.append(inter)

            member_ids = sorted(m["idx"] for m in members)
            is_mining = len(clipped_mining) > 0
            clipped_non = []
            if not is_mining:
                for m in members:
                    if m["class"] != 0:
                        continue
                    try:
                        inter = fix_geom(m["geom"].intersection(pbox))
                    except Exception:
                        inter = m["geom"]
                    if inter is not None and not inter.is_empty:
                        clipped_non.append(inter)

            rgb = truecolor(data)
            gt = rasterize_mining(clipped_mining, transform, (PATCH_SIZE, PATCH_SIZE))
            outlines = clipped_mining if is_mining else clipped_non

            tag = "mining" if is_mining else "nonmining"
            out_name = f"patch_{uid:03d}_{tag}.png"
            out_path = os.path.join(PATCHES_DIR, out_name)
            render(uid, is_mining, rgb, gt, outlines, transform, member_ids, out_path)

            results.append(
                {
                    "uid": uid,
                    "name": out_name,
                    "is_mining": is_mining,
                    "n_polys": len(clipped_mining),
                    "members": member_ids,
                    "gt_pct": float(gt.mean() * 100),
                    "col": col,
                    "row": row,
                }
            )
            print(
                f"  ✓ {out_name}: polys={len(clipped_mining)}, "
                f"labels={member_ids}, GT={gt.mean()*100:.1f}%"
            )

    n_m = sum(1 for r in results if r["is_mining"])
    n_n = sum(1 for r in results if not r["is_mining"])
    multi = sum(1 for r in results if r["n_polys"] > 1)

    with open(os.path.join(HERE, "README.md"), "w") as f:
        f.write("# 2026-07-23 Final Label GT Preview (Unique Patches)\n\n")
        f.write(
            "Ein PNG **pro geografischem 128×128-Fenster**. "
            "Ground Truth enthält **immer alle Mining-Polygone** im Fenster.\n\n"
        )
        f.write(f"- Unique Mining-Patches: **{n_m}**\n")
        f.write(f"- Unique Non-Mining-Patches (manuell): **{n_n}**\n")
        f.write(f"- Patches mit >1 Mining-Polygon: **{multi}**\n")
        f.write(f"- Zufällige Non-Mining (label_122+): **{len(random_kept)}**\n\n")
        f.write("| Patch | Datei | #Mining-Polygone | Original-Label-IDs | Mining-Pixel |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            ids = ",".join(f"{i:03d}" for i in r["members"])
            f.write(
                f"| {r['uid']:03d} | `{r['name']}` | {r['n_polys']} | {ids} | {r['gt_pct']:.1f}% |\n"
            )

    # mapping file for later finetuning selection
    with open(os.path.join(HERE, "patch_to_labels.csv"), "w") as f:
        f.write("patch_id,filename,is_mining,n_mining_polygons,label_ids,col,row,gt_mining_pct\n")
        for r in results:
            ids = ";".join(str(i) for i in r["members"])
            f.write(
                f"{r['uid']},{r['name']},{int(r['is_mining'])},{r['n_polys']},"
                f"\"{ids}\",{r['col']},{r['row']},{r['gt_pct']:.3f}\n"
            )

    print(f"\nDone: {n_m} mining + {n_n} nonmining unique patches ({multi} multi-polygon)")
    print(f"Plus {len(random_kept)} random nonmining kept")


if __name__ == "__main__":
    main()
