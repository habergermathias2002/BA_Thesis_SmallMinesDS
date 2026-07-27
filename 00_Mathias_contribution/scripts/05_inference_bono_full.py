"""
05_inference_bono_full.py
==========================

Fensterweise Inferenz auf dem Bono-Mosaik (128×128) mit dem Fine-Tuned Checkpoint.

Outputs (default: data/inference_bono_full_ft/):
  - prediction_prob.tif   (float32, P(Mining) 0–1)
  - prediction_binary.tif (uint8, 0 = Non-Mining, 1 = Mining)

Usage (Repo-Root, env smallmines):
  LIMIT_PATCHES=100 python 00_Mathias_contribution/scripts/05_inference_bono_full.py   # Smoke-Test
  LIMIT_PATCHES=0   python 00_Mathias_contribution/scripts/05_inference_bono_full.py   # Voll-Lauf

Env-Overrides:
  MOSAIC_PATH, CHECKPOINT_PATH, OUT_DIR, LIMIT_PATCHES, MINING_THRESH
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import rasterio
from rasterio.windows import Window

REPO_ROOT = Path(__file__).resolve().parents[2]

MOSAIC_PATH = Path(
    os.environ.get("MOSAIC_PATH", REPO_ROOT / "data" / "raw" / "Bono_Merged_2025.tif")
)
CHECKPOINT_PATH = Path(
    os.environ.get(
        "CHECKPOINT_PATH",
        REPO_ROOT / "models" / "prithvi-v2-300-finetuned.ckpt",
    )
)
OUT_DIR = Path(
    os.environ.get("OUT_DIR", REPO_ROOT / "data" / "inference_bono_full_ft")
)
REPORT_DIR = REPO_ROOT / "reports" / "05_Full_Bono_Inference"

MEANS = np.array(
    [1473.81, 1703.35, 1696.68, 3832.40, 3156.11, 2226.07], dtype=np.float32
)
STDS = np.array(
    [223.44, 285.54, 413.82, 389.61, 451.50, 468.27], dtype=np.float32
)
PATCH_SIZE = 128
MINING_THRESH = float(os.environ.get("MINING_THRESH", "0.5"))
BAND_INDICES = list(range(1, 7))  # B2,B3,B4,B8A,B11,B12
LIMIT_PATCHES = int(os.environ.get("LIMIT_PATCHES", "100"))  # 0 = all


def load_model(checkpoint_path: Path):
    from terratorch.tasks import SemanticSegmentationTask

    bands = ["BLUE", "GREEN", "RED", "VNIR_5", "SWIR_1", "SWIR_2"]
    model_args = {
        "backbone": "prithvi_eo_v2_300",
        "bands": bands,
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
        str(checkpoint_path),
        model_args=model_args,
        model_factory="PrithviModelFactory",
        loss="ce",
        lr=1e-3,
        ignore_index=-1,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": 0.05},
        freeze_backbone=True,
        class_names=["Non_mining", "Mining"],
        strict=False,
    )
    task.eval()
    # CUDA preferred; avoid MPS (adaptive_avg_pool2d bug)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = task.to(device)
    print(f"Model loaded: {checkpoint_path.name} on {device}")
    return task, device


def normalize(patch: np.ndarray) -> np.ndarray:
    """patch (6, H, W) on DN scale 0–10000; NaNs → 0 before z-score."""
    patch = np.nan_to_num(patch.astype(np.float32), nan=0.0)
    return (patch - MEANS[:, None, None]) / STDS[:, None, None]


def main():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint fehlt: {CHECKPOINT_PATH}")
    if not MOSAIC_PATH.is_file():
        raise FileNotFoundError(f"Mosaik fehlt: {MOSAIC_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "tables").mkdir(exist_ok=True)
    (REPORT_DIR / "figures").mkdir(exist_ok=True)

    task, device = load_model(CHECKPOINT_PATH)

    with rasterio.open(MOSAIC_PATH) as src:
        width, height = src.width, src.height
        full_h = (height // PATCH_SIZE) * PATCH_SIZE
        full_w = (width // PATCH_SIZE) * PATCH_SIZE
        n_rows = full_h // PATCH_SIZE
        n_cols = full_w // PATCH_SIZE
        n_full = n_rows * n_cols

        # Smoke-Test: nur so viele Patches wie LIMIT; Output-Raster auf benötigte Rows begrenzen
        if LIMIT_PATCHES > 0:
            n_proc = min(LIMIT_PATCHES, n_full)
            use_rows = (n_proc + n_cols - 1) // n_cols
            use_cols = n_cols if n_proc >= n_cols else n_proc
            # For partial last row, still write full row width for simpler geotransform
            out_h = use_rows * PATCH_SIZE
            out_w = full_w
            total = n_proc
        else:
            out_h, out_w = full_h, full_w
            total = n_full

        print(f"Mosaic: {width}×{height} px")
        print(f"Grid:   {n_rows}×{n_cols} = {n_full} patches")
        print(f"Run:    {total} patches → output {out_w}×{out_h} px")
        print(f"Out:    {OUT_DIR}")

        profile_prob = {
            "driver": "GTiff",
            "width": out_w,
            "height": out_h,
            "count": 1,
            "dtype": "float32",
            "crs": src.crs,
            "transform": src.transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 128,
            "blockysize": 128,
            "BIGTIFF": "YES",  # full mosaic output can exceed classic TIFF's 4 GB limit
        }
        profile_binary = {**profile_prob, "dtype": "uint8"}

        prob_path = OUT_DIR / "prediction_prob.tif"
        binary_path = OUT_DIR / "prediction_binary.tif"

        processed = 0
        mining_pixels = 0
        valid_pixels = 0

        with rasterio.open(prob_path, "w", **profile_prob) as dst_prob, rasterio.open(
            binary_path, "w", **profile_binary
        ) as dst_binary:
            for ri in range(n_rows):
                for ci in range(n_cols):
                    if LIMIT_PATCHES > 0 and processed >= LIMIT_PATCHES:
                        break
                    row_off = ri * PATCH_SIZE
                    col_off = ci * PATCH_SIZE
                    if row_off + PATCH_SIZE > out_h:
                        break

                    window = Window(col_off, row_off, PATCH_SIZE, PATCH_SIZE)
                    # GEE reflectance 0–1 → DN 0–10000
                    patch = src.read(BAND_INDICES, window=window).astype(np.float32) * 10000.0
                    patch_norm = normalize(patch)
                    tensor = torch.from_numpy(patch_norm).float().unsqueeze(0).to(device)

                    with torch.no_grad():
                        out = task.model(tensor)
                        logits = out.output if hasattr(out, "output") else out
                        probs = torch.softmax(logits, dim=1)
                        mining_prob = probs[0, 1].detach().cpu().numpy().astype(np.float32)

                    binary = (mining_prob >= MINING_THRESH).astype(np.uint8)
                    dst_prob.write(mining_prob, 1, window=window)
                    dst_binary.write(binary, 1, window=window)

                    mining_pixels += int(binary.sum())
                    valid_pixels += binary.size
                    processed += 1
                    if processed % 50 == 0 or processed == total:
                        print(f"  {processed}/{total} patches")

                if LIMIT_PATCHES > 0 and processed >= LIMIT_PATCHES:
                    break

        share = mining_pixels / max(valid_pixels, 1)
        stats_path = REPORT_DIR / "tables" / "inference_stats.txt"
        stats = (
            f"checkpoint={CHECKPOINT_PATH}\n"
            f"mosaic={MOSAIC_PATH}\n"
            f"limit_patches={LIMIT_PATCHES}\n"
            f"processed={processed}\n"
            f"output_size={out_w}x{out_h}\n"
            f"mining_pixels={mining_pixels}\n"
            f"valid_pixels={valid_pixels}\n"
            f"mining_share={share:.6f}\n"
            f"threshold={MINING_THRESH}\n"
            f"prob={prob_path}\n"
            f"binary={binary_path}\n"
        )
        stats_path.write_text(stats)
        print(f"Done. Mining share (processed pixels): {share:.4%}")
        print(f"Outputs: {prob_path}\n         {binary_path}")
        print(f"Stats:   {stats_path}")


if __name__ == "__main__":
    main()
