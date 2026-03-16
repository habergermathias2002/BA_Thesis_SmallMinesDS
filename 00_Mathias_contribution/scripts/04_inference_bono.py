"""
04_inference_bono.py
=====================

Kurz: Führt mit dem trainierten Prithvi-Checkpoint Inferenz auf den Bono-Testpatches
(aus Skript 02) aus: lädt jede 128×128-Patch, normalisiert mit SmallMinesDS-Statistik,
berechnet die Mining-Wahrscheinlichkeit pro Pixel und setzt die Vorhersagen zu einem
georeferenzierten Raster zusammen. Ausgabe: prediction_binary.tif, prediction_prob.tif
und eine Visualisierungs-PNG in data/patches_bono_test/.

What it does:
  1. Loads the trained checkpoint (from 03_train_colab.py)
  2. For each patch: normalizes values using SmallMinesDS statistics,
     runs a forward pass, extracts per-pixel Mining probability (0–1)
  3. Reassembles individual patch predictions into a single raster
  4. Saves two GeoTIFFs:
       - prediction_binary.tif   (0 = Non-Mining, 1 = Mining)
       - prediction_prob.tif     (float32, 0.0–1.0 Mining probability)
  5. Saves a visualization PNG (false-color + probability heatmap)

Normalization:
  The model was trained with z-score normalization using SmallMinesDS statistics.
  Patches from Bono were already rescaled × 10,000 (by script 02).
  Here we apply: normalized = (value - mean) / std

Output:
  data/patches_bono_test/prediction_binary.tif
  data/patches_bono_test/prediction_prob.tif
  data/patches_bono_test/prediction_visualization.png

Usage:
  python 00_Mathias_contribution/scripts/04_inference_bono.py
  (run from repo root)
  Can also run on Google Colab (set CHECKPOINT_PATH below).
"""

import os
import csv
import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Paths — checkpoint: erste .ckpt-Datei im Ordner models/ ───────────────────
REPO_ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATCHES_DIR  = os.path.join(REPO_ROOT, "data", "patches_bono_test")
MODELS_DIR   = os.path.join(REPO_ROOT, "models")
_ckpts       = [f for f in os.listdir(MODELS_DIR) if f.endswith(".ckpt")]
CHECKPOINT_PATH = os.path.join(MODELS_DIR, _ckpts[0]) if _ckpts else os.path.join(MODELS_DIR, "prithvi-v2-300-best.ckpt")
# ↑ Verwendet automatisch die erste .ckpt-Datei in models/ (z.B. prithvi-v2-300-epoch=16-val_loss=0.0000.ckpt)

# ── Normalization statistics ───────────────────────────────────────────────────
# TEST A: Bono-specific statistics (computed from all 16 test patches)
# These replace SmallMinesDS stats to check if domain-shift in value range is
# the reason the model predicts 0% everywhere.
# SmallMinesDS stats (original): [1473.81, 1703.35, 1696.68, 3832.40, 3156.11, 2226.07]
MEANS = np.array([ 583.63,  851.72, 1241.71, 2411.21, 3027.37, 2290.58], dtype=np.float32)
STDS  = np.array([ 157.83,  227.45,  348.81,  717.34,  828.90,  609.14], dtype=np.float32)

PATCH_SIZE     = 128
MINING_THRESH  = 0.5   # probability threshold for binary mask
# Suffix für Ausgabedateien, damit alte Ergebnisse nicht überschrieben werden (z. B. "_6band"; "" = überschreiben)
OUTPUT_SUFFIX  = "_6band"


def load_model(checkpoint_path):
    """Load the fine-tuned SemanticSegmentationTask from a Lightning checkpoint."""
    from terratorch.tasks import SemanticSegmentationTask
    from terratorch.models import PrithviModelFactory

    ghana_mining_bands = ["BLUE", "GREEN", "RED", "VNIR_5", "SWIR_1", "SWIR_2"]
    model_args = {
        "backbone":              "prithvi_eo_v2_300",
        "bands":                 ghana_mining_bands,
        "in_channels":           6,
        "num_classes":           2,
        "pretrained":            False,  # weights come from checkpoint, not HuggingFace
        "decoder":               "UperNetDecoder",
        "rescale":               True,
        "backbone_num_frames":   1,
        "head_dropout":          0.1,
        "decoder_scale_modules": True,
    }

    task = SemanticSegmentationTask.load_from_checkpoint(
        checkpoint_path,
        model_args=model_args,
        model_factory="PrithviModelFactory",
        loss="ce",
        lr=1e-3,
        ignore_index=-1,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": 0.05},
        freeze_backbone=True,
        class_names=["Non_mining", "Mining"],
    )
    task.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = task.to(device)
    print(f"Model loaded on: {device}")
    return task, device


def normalize(patch):
    """Apply SmallMinesDS z-score normalization. patch shape: (6, H, W)."""
    return (patch - MEANS[:, None, None]) / STDS[:, None, None]


def run_inference(task, device, patches_dir):
    """
    Reads patch_index.csv, runs inference on each patch,
    returns prob_map and binary_map as 2D numpy arrays.
    """
    index_path = os.path.join(patches_dir, "patch_index.csv")
    with open(index_path) as f:
        records = list(csv.DictReader(f))

    # Determine grid size
    max_row = max(int(r["row"]) for r in records)
    max_col = max(int(r["col"]) for r in records)
    n_rows  = max_row + 1
    n_cols  = max_col + 1

    prob_map   = np.zeros((n_rows * PATCH_SIZE, n_cols * PATCH_SIZE), dtype=np.float32)
    binary_map = np.zeros((n_rows * PATCH_SIZE, n_cols * PATCH_SIZE), dtype=np.uint8)

    print(f"Running inference on {len(records)} patches ({n_rows}×{n_cols} grid)...")

    for i, rec in enumerate(records):
        patch_path = os.path.join(patches_dir, rec["patch_file"])
        with rasterio.open(patch_path) as src:
            patch = src.read().astype(np.float32)  # shape: (6, 128, 128)

        # Normalize
        patch_norm = normalize(patch)

        # Forward pass
        tensor = torch.FloatTensor(patch_norm).unsqueeze(0).to(device)  # (1, 6, 128, 128)
        with torch.no_grad():
            out = task.model(tensor)
            # terratorch ModelOutput stores the logit tensor in .output
            logits = out.output if hasattr(out, "output") else out
            probs  = torch.softmax(logits, dim=1)   # (1, 2, 128, 128)
            mining_prob = probs[0, 1].cpu().numpy() # (128, 128)  — channel 1 = Mining

        # Place in map
        r = int(rec["row"])
        c = int(rec["col"])
        row_s = r * PATCH_SIZE
        col_s = c * PATCH_SIZE
        prob_map  [row_s:row_s+PATCH_SIZE, col_s:col_s+PATCH_SIZE] = mining_prob
        binary_map[row_s:row_s+PATCH_SIZE, col_s:col_s+PATCH_SIZE] = (mining_prob >= MINING_THRESH).astype(np.uint8)

        if (i + 1) % 4 == 0 or (i + 1) == len(records):
            print(f"  {i+1}/{len(records)} patches processed")

    return prob_map, binary_map, records


def get_full_transform(records):
    """Reconstruct the top-left GeoTransform of the reassembled raster."""
    top_left = [r for r in records if int(r["row"]) == 0 and int(r["col"]) == 0]
    rec = top_left[0]
    utm_left = float(rec["utm_left"])
    utm_top  = float(rec["utm_top"])
    return rasterio.transform.Affine(10.0, 0.0, utm_left, 0.0, -10.0, utm_top)


def save_geotiff(path, data, transform, count=1, dtype="float32"):
    """Save a numpy array as a single-band GeoTIFF."""
    with rasterio.open(
        path, "w",
        driver="GTiff", height=data.shape[0], width=data.shape[1],
        count=count, dtype=dtype, crs="EPSG:32630", transform=transform,
    ) as dst:
        dst.write(data[np.newaxis, :, :] if data.ndim == 2 else data)


def save_visualization(patches_dir, prob_map, binary_map, suffix=""):
    """
    Saves a side-by-side visualization:
      Left:  Mining probability heatmap (yellow-red = high probability)
      Right: Binary prediction (red = Mining, green = Non-Mining)
    """
    # Load a representative false-color image from one patch (top-left)
    first_patch_path = os.path.join(patches_dir, "patch_0000_r0_c0.tif")
    with rasterio.open(first_patch_path) as src:
        patch = src.read().astype(np.float32)
    # False color: SWIR1(4)/NIR(3)/Red(2) — normalized for display
    fc = np.dstack((patch[4], patch[3], patch[2]))
    fc = np.clip(fc / np.nanpercentile(fc, 98) * 0.8, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: False color (first patch only as reference)
    axes[0].imshow(fc)
    axes[0].set_title("False Color (SWIR/NIR/Red)\nTop-left patch", fontsize=12)
    axes[0].axis("off")

    # Panel 2: Probability heatmap
    im = axes[1].imshow(prob_map, cmap="YlOrRd", vmin=0, vmax=1)
    axes[1].set_title(f"Mining Probability\n(full {prob_map.shape[1]//100*10:.1f}×{prob_map.shape[0]//100*10:.1f} km area)", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="P(Mining)")

    # Panel 3: Binary mask
    colors = ["#2d6a4f", "#d62828"]  # green = Non-Mining, red = Mining
    cmap_binary = mcolors.ListedColormap(colors)
    axes[2].imshow(binary_map, cmap=cmap_binary, vmin=0, vmax=1)
    axes[2].set_title(f"Binary Prediction (threshold={MINING_THRESH})\nRed = Mining", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(patches_dir, f"prediction_visualization{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved: {out_path}")


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        print("Please set CHECKPOINT_PATH to your downloaded .ckpt file.")
        return

    # Load model
    task, device = load_model(CHECKPOINT_PATH)

    # Run inference
    prob_map, binary_map, records = run_inference(task, device, PATCHES_DIR)

    # Get georeferencing
    transform = get_full_transform(records)

    # Save prediction GeoTIFFs (OUTPUT_SUFFIX z. B. "_6band" → neue Dateien, keine Überschreibung)
    prob_path   = os.path.join(PATCHES_DIR, f"prediction_prob{OUTPUT_SUFFIX}.tif")
    binary_path = os.path.join(PATCHES_DIR, f"prediction_binary{OUTPUT_SUFFIX}.tif")

    save_geotiff(prob_path,   prob_map,   transform, dtype="float32")
    save_geotiff(binary_path, binary_map, transform, dtype="uint8")

    mining_pct = binary_map.mean() * 100
    print(f"\nPrediction summary:")
    print(f"  Mining pixels: {binary_map.sum():,} / {binary_map.size:,} ({mining_pct:.2f}%)")
    print(f"  Mean mining probability: {prob_map.mean():.3f}")
    print(f"\nSaved:")
    print(f"  {prob_path}")
    print(f"  {binary_path}")

    # Save visualization
    save_visualization(PATCHES_DIR, prob_map, binary_map, suffix=OUTPUT_SUFFIX)

    print("\nDone. Open the GeoTIFFs in QGIS to overlay with satellite imagery.")


if __name__ == "__main__":
    main()
