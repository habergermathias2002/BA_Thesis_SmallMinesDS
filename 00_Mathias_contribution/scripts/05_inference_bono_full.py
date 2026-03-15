"""
05_inference_bono_full.py
==========================

Kurz: Inferenz auf dem gesamten Bono-Mosaik (ohne vorher Patches zu extrahieren).
Liest das Raster fensterweise in 128×128-Blöcken, wendet das Modell an und schreibt
die Vorhersagen (Wahrscheinlichkeit und binär) in data/inference_bono_full/. Für
große Raster speichersparend; Laufzeit z. B. viele Stunden auf CPU. LIMIT_PATCHES
auf z. B. 100 setzen für einen schnellen Test.

Outputs (in data/inference_bono_full/):
  - prediction_prob.tif   (float32, Mining probability 0–1)
  - prediction_binary.tif (uint8, 0 = Non-Mining, 1 = Mining)

Runtime: Depends on mosaic size. Example: 20,000×20,000 px → ~24,400 patches;
  on CPU ~2–5 s/patch → ~14–34 h. Use GPU (Colab or local) to speed up.

Usage:
  python 00_Mathias_contribution/scripts/05_inference_bono_full.py
  (run from repo root; conda env smallmines)

Optional env or edit:
  MOSAIC_PATH  = path to Bono_Merged_2025.tif
  OUT_DIR      = directory for output rasters
  LIMIT_PATCHES = 0 = process all; set to e.g. 100 for a quick test
"""

import os
import numpy as np
import torch
import rasterio
from rasterio.windows import Window

# Reuse paths and constants from 04
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
MOSAIC_PATH = os.path.join(REPO_ROOT, "data", "raw", "Bono_Merged_2025.tif")
OUT_DIR = os.path.join(REPO_ROOT, "data", "inference_bono_full")

_ckpts = [f for f in os.listdir(MODELS_DIR) if f.endswith(".ckpt")]
CHECKPOINT_PATH = os.path.join(MODELS_DIR, _ckpts[0]) if _ckpts else os.path.join(MODELS_DIR, "prithvi-v2-300-best.ckpt")

MEANS = np.array([1473.81, 1703.35, 1696.68, 3832.40, 3156.11, 2226.07], dtype=np.float32)
STDS  = np.array([ 223.44,  285.54,  413.82,  389.61,  451.50,  468.27], dtype=np.float32)
PATCH_SIZE = 128
MINING_THRESH = 0.5
# Mosaic band order (1-based): B2,B3,B4,B8A,B11,B12 → BLUE,GREEN,RED,VNIR_5,SWIR_1,SWIR_2
BAND_INDICES = list(range(1, 7))

# Set to a positive number to process only that many patches (for testing)
LIMIT_PATCHES = 100  # 0 = all; 100 = quick test (~few min)


def load_model(checkpoint_path):
    from terratorch.tasks import SemanticSegmentationTask
    ghana_mining_bands = ["BLUE", "GREEN", "RED", "VNIR_5", "SWIR_1", "SWIR_2"]
    model_args = {
        "backbone": "prithvi_eo_v2_300",
        "bands": ghana_mining_bands,
        "in_channels": 6,
        "num_classes": 2,
        "pretrained": False,
        "decoder": "UperNetDecoder",
        "rescale": True,
        "backbone_num_frames": 1,
        "head_dropout": 0.1,
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
    # Prefer CUDA; avoid MPS (Apple Metal) due to adaptive_avg_pool2d bug on MPS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = task.to(device)
    print(f"Model loaded on: {device}")
    return task, device


def normalize(patch):
    """patch shape (6, H, W). Values must already be on 0–10,000 scale (GEE ×10000 undone)."""
    return (patch.astype(np.float32) - MEANS[:, None, None]) / STDS[:, None, None]


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        return
    if not os.path.exists(MOSAIC_PATH):
        print(f"[ERROR] Mosaic not found: {MOSAIC_PATH}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    task, device = load_model(CHECKPOINT_PATH)

    with rasterio.open(MOSAIC_PATH) as src:
        width, height = src.width, src.height
        out_h = (height // PATCH_SIZE) * PATCH_SIZE
        out_w = (width // PATCH_SIZE) * PATCH_SIZE
        n_rows = out_h // PATCH_SIZE
        n_cols = out_w // PATCH_SIZE
        total = n_rows * n_cols
        if LIMIT_PATCHES > 0:
            total = min(total, LIMIT_PATCHES)
        print(f"Mosaic: {width}×{height} px → processing {out_w}×{out_h} px")
        print(f"Patches: {n_rows}×{n_cols} = {n_rows * n_cols} (processing {total})")

        # Output transform: same as source (upper-left aligned)
        out_transform = src.transform
        crs = src.crs
        profile_prob = {
            "driver": "GTiff",
            "width": out_w,
            "height": out_h,
            "count": 1,
            "dtype": "float32",
            "crs": crs,
            "transform": out_transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 128,
            "blockysize": 128,
        }
        profile_binary = {**profile_prob, "dtype": "uint8"}

        prob_path = os.path.join(OUT_DIR, "prediction_prob.tif")
        binary_path = os.path.join(OUT_DIR, "prediction_binary.tif")

        processed = 0
        with rasterio.open(prob_path, "w", **profile_prob) as dst_prob, \
             rasterio.open(binary_path, "w", **profile_binary) as dst_binary:
            for ri in range(n_rows):
                for ci in range(n_cols):
                    if LIMIT_PATCHES > 0 and processed >= LIMIT_PATCHES:
                        break
                    row_off = ri * PATCH_SIZE
                    col_off = ci * PATCH_SIZE
                    window = Window(col_off, row_off, PATCH_SIZE, PATCH_SIZE)
                    # Read 6 bands; GEE export is 0–1 → rescale to 0–10,000
                    patch = src.read(BAND_INDICES, window=window).astype(np.float32) * 10000.0
                    patch_norm = normalize(patch)
                    tensor = torch.FloatTensor(patch_norm).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = task.model(tensor)
                        logits = out.output if hasattr(out, "output") else out
                        probs = torch.softmax(logits, dim=1)
                        mining_prob = probs[0, 1].cpu().numpy()
                    binary = (mining_prob >= MINING_THRESH).astype(np.uint8)
                    dst_prob.write(mining_prob.astype(np.float32), 1, window=window)
                    dst_binary.write(binary, 1, window=window)
                    processed += 1
                    if processed % 100 == 0 or processed == total:
                        print(f"  {processed}/{total} patches")
                if LIMIT_PATCHES > 0 and processed >= LIMIT_PATCHES:
                    break

        print(f"Done. Outputs: {prob_path}, {binary_path}")


if __name__ == "__main__":
    main()
