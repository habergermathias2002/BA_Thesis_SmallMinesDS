"""
Export der final_selection (65 Patches) als GeoTIFF-Paare für Fine-Tuning.

Output:
  data/GhanaMiningPrithvi_bono/
    training/   BONO_XXXX_IMG.tif + BONO_XXXX_MASK.tif
    validation/ …
    split_manifest.csv

- IMG: 6 Bänder, Float32, DN-Skala (Reflektanz × 10.000)
- MASK: 1 Band, UINT8, 0=Non-Mining, 1=Mining
  (ALLE Mining-Polygone aus dem GPKG, die das Fenster schneiden)
- Split: 80/20, stratifiziert nach Mining / Non-Mining
"""
from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import fiona
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio import features as rio_features
from shapely.geometry import shape, mapping, box
from shapely.ops import unary_union
from shapely.validation import make_valid

REPO = Path(__file__).resolve().parents[2]
PREVIEW = REPO / "00_Mathias_contribution" / "20260723_final_Label_GT_Preview"
MANIFEST = PREVIEW / "final_selection" / "MANIFEST.txt"
PATCH_CSV = PREVIEW / "patch_to_labels.csv"
GPKG = REPO / "00_Mathias_contribution" / "labels_incoming" / "richtige labels 23..gpkg"
LABELS_LAYER = "richtige_labels_v1"
CLASS_FIELD = "Mining-y-n"
MOSAIC = REPO / "data" / "raw" / "Bono_Merged_2025.tif"
OUT_ROOT = REPO / "data" / "GhanaMiningPrithvi_bono"
TRAIN_DIR = OUT_ROOT / "training"
VAL_DIR = OUT_ROOT / "validation"

PATCH_SIZE = 128
SCALE = 10000.0  # GEE 0–1 → DN
VAL_FRAC = 0.20
SEED = 23


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


def load_manifest(path: Path):
    """Return list of (id, class_tag, filename)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Mining") or line.startswith("Non") or line.startswith("Total"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            rows.append((int(parts[0]), parts[1], parts[2]))
    return rows


def load_patch_csv(path: Path):
    """patch_id -> {col,row,is_mining,...}"""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["patch_id"])] = {
                "col": int(row["col"]),
                "row": int(row["row"]),
                "is_mining": int(row["is_mining"]),
                "filename": row["filename"],
            }
    return out


def load_mining_geoms(gpkg: Path):
    geoms = []
    with fiona.open(gpkg, layer=LABELS_LAYER) as src:
        for feat in src:
            props = feat["properties"] or {}
            if int(props.get(CLASS_FIELD) or 0) != 1:
                continue
            g = fix_geom(shape(feat["geometry"]))
            if g is not None and not g.is_empty:
                geoms.append(g)
    return geoms


def reconstruct_random_nonmining_windows(mosaic_path: Path, mining_geoms, n_needed=112, start_idx=122):
    """
    Replay the same sampler used when generating label_122.. label_233
    (seed=23, ≥200 m from mining, ≥95% valid pixels).
    Returns dict: label_id -> (col, row)
    """
    RNG = random.Random(23)
    mining_union = unary_union(mining_geoms).buffer(200)
    coords = {}
    generated = []
    attempts = 0
    max_attempts = n_needed * 40

    with rasterio.open(mosaic_path) as src:
        w, h = src.width, src.height
        transform = src.transform
        while len(generated) < n_needed and attempts < max_attempts:
            attempts += 1
            col = RNG.randint(0, w - PATCH_SIZE - 1)
            row = RNG.randint(0, h - PATCH_SIZE - 1)
            patch_box = box(
                transform.c + col * transform.a,
                transform.f + (row + PATCH_SIZE) * transform.e,
                transform.c + (col + PATCH_SIZE) * transform.a,
                transform.f + row * transform.e,
            )
            if mining_union.intersects(patch_box):
                continue
            data = src.read(1, window=Window(col, row, PATCH_SIZE, PATCH_SIZE)).astype(np.float32)
            valid_frac = float(np.isfinite(data).mean())
            if valid_frac < 0.95:
                continue
            if float(np.nanmean(data)) < 0.02:
                continue
            idx = start_idx + len(generated)
            generated.append(idx)
            coords[idx] = (col, row)
    if len(coords) < n_needed:
        raise RuntimeError(f"Only reconstructed {len(coords)}/{n_needed} random windows")
    return coords


def patch_box_from_window(transform, col, row):
    x0 = transform.c + col * transform.a
    y0 = transform.f + row * transform.e
    x1 = transform.c + (col + PATCH_SIZE) * transform.a
    y1 = transform.f + (row + PATCH_SIZE) * transform.e
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def rasterize_mask(mining_geoms, transform, pbox):
    shapes = []
    for g in mining_geoms:
        if g is None or g.is_empty or not g.intersects(pbox):
            continue
        try:
            inter = fix_geom(g.intersection(pbox))
        except Exception:
            inter = fix_geom(make_valid(g).intersection(pbox))
        if inter is not None and not inter.is_empty:
            shapes.append((mapping(inter), 1))
    mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
    if shapes:
        rio_features.rasterize(
            shapes,
            out_shape=mask.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            out=mask,
        )
    return mask


def stratified_split(items, val_frac=VAL_FRAC, seed=SEED):
    """
    items: list of dicts with key 'is_mining'
    Returns train_list, val_list with ~val_frac per class.
    """
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for it in items:
        by_class[it["is_mining"]].append(it)

    train, val = [], []
    for cls, group in by_class.items():
        group = group[:]
        rng.shuffle(group)
        n_val = max(1, int(round(len(group) * val_frac))) if len(group) >= 5 else max(1, len(group) // 5)
        # ensure at least 1 val if group large enough, and not empty train
        if len(group) == 1:
            train.extend(group)
            continue
        if n_val >= len(group):
            n_val = len(group) - 1
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_pair(out_dir: Path, stem: str, img_dn, mask, transform, crs):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"{stem}_IMG.tif"
    mask_path = out_dir / f"{stem}_MASK.tif"

    profile_img = {
        "driver": "GTiff",
        "height": PATCH_SIZE,
        "width": PATCH_SIZE,
        "count": 6,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(img_path, "w", **profile_img) as dst:
        dst.write(img_dn.astype(np.float32))

    profile_mask = {
        "driver": "GTiff",
        "height": PATCH_SIZE,
        "width": PATCH_SIZE,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
    }
    with rasterio.open(mask_path, "w", **profile_mask) as dst:
        dst.write(mask.astype(np.uint8), 1)

    return img_path, mask_path


def main():
    selected = load_manifest(MANIFEST)
    patch_meta = load_patch_csv(PATCH_CSV)
    mining_geoms = load_mining_geoms(GPKG)
    print(f"Selected: {len(selected)} | Mining polygons in GPKG: {len(mining_geoms)}")

    # Reconstruct random nonmining windows for IDs >= 122
    random_ids = [i for i, _, _ in selected if i >= 122]
    random_coords = {}
    if random_ids:
        print("Reconstructing random Non-Mining windows (seed=23)…")
        all_random = reconstruct_random_nonmining_windows(MOSAIC, mining_geoms, n_needed=112)
        for rid in random_ids:
            if rid not in all_random:
                raise KeyError(f"Random label {rid} not found in reconstructed set")
            random_coords[rid] = all_random[rid]
        print(f"  Resolved {len(random_coords)} random windows")

    items = []
    for pid, tag, fname in selected:
        is_mining = 1 if tag == "mining" else 0
        if pid < 122:
            if pid not in patch_meta:
                raise KeyError(f"patch_id {pid} missing in patch_to_labels.csv")
            col, row = patch_meta[pid]["col"], patch_meta[pid]["row"]
            source = "unique_patch"
        else:
            col, row = random_coords[pid]
            source = "random_nonmining"
        items.append(
            {
                "id": pid,
                "tag": tag,
                "is_mining": is_mining,
                "filename": fname,
                "col": col,
                "row": row,
                "source": source,
            }
        )

    train_items, val_items = stratified_split(items)
    print(
        f"Split: train={len(train_items)} "
        f"(M={sum(i['is_mining'] for i in train_items)}, "
        f"N={sum(1-i['is_mining'] for i in train_items)}) | "
        f"val={len(val_items)} "
        f"(M={sum(i['is_mining'] for i in val_items)}, "
        f"N={sum(1-i['is_mining'] for i in val_items)})"
    )

    # clean output dirs
    for d in (TRAIN_DIR, VAL_DIR):
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("BONO_*"):
            p.unlink()

    manifest_rows = []
    with rasterio.open(MOSAIC) as src:
        crs = src.crs
        for split_name, split_items in (("training", train_items), ("validation", val_items)):
            out_dir = TRAIN_DIR if split_name == "training" else VAL_DIR
            for rank, it in enumerate(sorted(split_items, key=lambda x: x["id"])):
                col, row = it["col"], it["row"]
                window = Window(col, row, PATCH_SIZE, PATCH_SIZE)
                transform = src.window_transform(window)
                data01 = src.read(window=window).astype(np.float32)
                if data01.shape[1] != PATCH_SIZE or data01.shape[2] != PATCH_SIZE:
                    padded = np.full((6, PATCH_SIZE, PATCH_SIZE), np.nan, dtype=np.float32)
                    padded[:, : data01.shape[1], : data01.shape[2]] = data01
                    data01 = padded
                data01 = np.nan_to_num(data01, nan=0.0)
                img_dn = data01 * SCALE

                pbox = patch_box_from_window(src.transform, col, row)
                mask = rasterize_mask(mining_geoms, transform, pbox)
                # For explicitly nonmining selection with empty mask: keep zeros
                # (already the case if no mining polys intersect)

                stem = f"BONO_{it['id']:04d}"
                write_pair(out_dir, stem, img_dn, mask, transform, crs)
                mining_pct = float(mask.mean() * 100)
                manifest_rows.append(
                    {
                        "stem": stem,
                        "split": split_name,
                        "selection_id": it["id"],
                        "class": it["tag"],
                        "is_mining": it["is_mining"],
                        "source": it["source"],
                        "col": col,
                        "row": row,
                        "mining_pct": round(mining_pct, 3),
                        "preview": it["filename"],
                    }
                )
                print(
                    f"  [{split_name}] {stem}: {it['tag']}, "
                    f"mask={mining_pct:.1f}%, col={col}, row={row}"
                )

    man_path = OUT_ROOT / "split_manifest.csv"
    with open(man_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    # README
    n_tr_m = sum(1 for r in manifest_rows if r["split"] == "training" and r["is_mining"] == 1)
    n_tr_n = sum(1 for r in manifest_rows if r["split"] == "training" and r["is_mining"] == 0)
    n_va_m = sum(1 for r in manifest_rows if r["split"] == "validation" and r["is_mining"] == 1)
    n_va_n = sum(1 for r in manifest_rows if r["split"] == "validation" and r["is_mining"] == 0)
    readme = OUT_ROOT / "README.md"
    readme.write_text(
        f"""# GhanaMiningPrithvi_bono — Fine-Tuning Dataset

Exported from final selection (2026-07-23).

## Counts

| Split | Mining | Non-Mining | Total |
|---|---|---|---|
| training | {n_tr_m} | {n_tr_n} | {n_tr_m + n_tr_n} |
| validation | {n_va_m} | {n_va_n} | {n_va_m + n_va_n} |
| **total** | **{n_tr_m+n_va_m}** | **{n_tr_n+n_va_n}** | **{len(manifest_rows)}** |

Stratified 80/20 by class (seed={SEED}).

## Format

- `BONO_XXXX_IMG.tif`: 6 bands (B2,B3,B4,B8A,B11,B12), float32, DN = reflectance × 10 000
- `BONO_XXXX_MASK.tif`: 1 band uint8, 0=Non-Mining, 1=Mining  
  Mask includes **all** mining polygons intersecting the 128×128 window.

## Files

- `split_manifest.csv` — full mapping selection_id → split / class / pixel coords
- Generated by: `00_Mathias_contribution/scripts/export_bono_finetune_dataset.py`
"""
    )
    print(f"\nDone → {OUT_ROOT}")
    print(f"Manifest: {man_path}")
    print(f"Train M/N = {n_tr_m}/{n_tr_n} | Val M/N = {n_va_m}/{n_va_n}")


if __name__ == "__main__":
    main()
