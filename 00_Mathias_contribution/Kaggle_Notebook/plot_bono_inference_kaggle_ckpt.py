"""
Bono-Inferenz mit Kaggle-Checkpoint.

1. Führt Inferenz auf allen 16 Bono-Test-Patches durch (5×5 km Testgebiet).
2. Zeigt ein 2-spaltiges Vergleichsbild mit 6 zufällig ausgewählten Patches:
     Links:  True-Color-Satellitenbild (B4/B3/B2)
     Rechts: Mining-Wahrscheinlichkeit (weiß = kein Mining, rot = Mining)

Checkpoint: Erste .ckpt-Datei in 00_Mathias_contribution/Kaggle_Notebook/
Normalisierung: SmallMinesDS-Statistiken (identisch zu Training)
Ausgabe: 00_Mathias_contribution/Kaggle_Notebook/bono_inference_kaggle_ckpt.png
"""

import os
import csv
import random
import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATCHES_DIR = os.path.join(REPO_ROOT, "data", "patches_bono_test")
CKPT_DIR    = os.path.dirname(os.path.abspath(__file__))
OUT_PATH    = os.path.join(CKPT_DIR, "bono_inference_kaggle_ckpt.png")

# ── Checkpoint automatisch finden ────────────────────────────────────────────
_ckpts = sorted([f for f in os.listdir(CKPT_DIR) if f.endswith(".ckpt")])
if not _ckpts:
    raise FileNotFoundError(
        f"Keine .ckpt-Datei in {CKPT_DIR}\n"
        "Kaggle-Checkpoint dort ablegen (z.B. last.ckpt)."
    )
CKPT_PATH = os.path.join(CKPT_DIR, _ckpts[0])
print(f"Verwende Checkpoint: {CKPT_PATH}")

# ── Normalisierung: SmallMinesDS-Statistiken (B2,B3,B4,B8A,B11,B12) ─────────
# Identisch zu Training – Bono-Patches wurden in 02_extract_bono_test_patches.py
# bereits mit ×10000 auf die gleiche Größenordnung gebracht.
MEANS = np.array(
    [1473.81388377, 1703.35249650, 1696.67685941, 3832.39764247, 3156.11122121, 2226.06822112],
    dtype=np.float32,
)
STDS = np.array(
    [223.43533204, 285.53613398, 413.82320306, 389.61483882, 451.49534791, 468.26765909],
    dtype=np.float32,
)

PATCH_SIZE   = 128
MINING_THRESH = 0.5
R_IDX, G_IDX, B_IDX = 2, 1, 0   # True Color: B4, B3, B2

RANDOM_SEED  = 42                 # für Reproduzierbarkeit
N_SHOW       = 6                  # Anzahl Patches im Vergleichsbild


def load_model():
    from terratorch.tasks import SemanticSegmentationTask
    task = SemanticSegmentationTask.load_from_checkpoint(
        CKPT_PATH,
        model_args={
            "backbone": "prithvi_eo_v2_300",
            "bands": ["BLUE", "GREEN", "RED", "VNIR_5", "SWIR_1", "SWIR_2"],
            "in_channels": 6,
            "num_classes": 2,
            "pretrained": False,
            "decoder": "UperNetDecoder",
            "rescale": True,
            "backbone_num_frames": 1,
            "head_dropout": 0.1,
            "decoder_scale_modules": True,
        },
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
    return task


def predict_patch(task, img6: np.ndarray) -> np.ndarray:
    norm = (img6.astype(np.float32) - MEANS[:, None, None]) / STDS[:, None, None]
    t = torch.from_numpy(norm).float().unsqueeze(0)
    with torch.no_grad():
        out = task.model(t)
        logits = out.output if hasattr(out, "output") else out
        probs = torch.softmax(logits, dim=1)
    return probs[0, 1].cpu().numpy()


def truecolor(img6: np.ndarray) -> np.ndarray:
    rgb = np.dstack((img6[R_IDX], img6[G_IDX], img6[B_IDX])).astype(np.float32)
    lo, hi = np.nanpercentile(rgb, (2, 98))
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    # ── 1. Alle Patches aus patch_index.csv laden ────────────────────────────
    index_path = os.path.join(PATCHES_DIR, "patch_index.csv")
    with open(index_path) as f:
        records = list(csv.DictReader(f))

    print(f"\nBono-Test-Region: {len(records)} Patches gefunden.")
    task = load_model()

    # ── 2. Inferenz auf allen Patches ────────────────────────────────────────
    results = []   # (patch_file, img6, prob_map)
    for rec in records:
        patch_path = os.path.join(PATCHES_DIR, rec["patch_file"])
        with rasterio.open(patch_path) as src:
            img6 = src.read().astype(np.float32)
        prob = predict_patch(task, img6)
        results.append((rec["patch_file"], img6, prob))
        print(f"  {rec['patch_file']:30s}  ⌀ Mining: {prob.mean()*100:.1f}%")

    # ── 3. 6 zufällige Patches für Vergleichsbild auswählen ──────────────────
    random.seed(RANDOM_SEED)
    selected = random.sample(results, min(N_SHOW, len(results)))

    cmap = mcolors.LinearSegmentedColormap.from_list("wp", ["white", "red"])
    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3.5))
    fig.suptitle(
        f"Bono-Inferenz mit Kaggle-Checkpoint ({os.path.basename(CKPT_PATH)})\n"
        "Links: True-Color-Satellitenbild  |  Rechts: Mining-Wahrscheinlichkeit",
        fontsize=11,
        y=1.01,
    )

    for row, (fn, img6, prob) in enumerate(selected):
        ax_sat  = axes[row, 0]
        ax_pred = axes[row, 1]
        pred_pct = prob.mean() * 100.0

        ax_sat.imshow(truecolor(img6), interpolation="nearest")
        ax_sat.set_title(f"{fn}", fontsize=7, loc="left")
        ax_sat.axis("off")

        im = ax_pred.imshow(prob, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_pred.set_title(
            f"⌀ {pred_pct:.1f}% Mining-Wahrscheinlichkeit",
            fontsize=7,
            loc="left",
        )
        ax_pred.axis("off")
        plt.colorbar(im, ax=ax_pred, fraction=0.046, pad=0.04, label="P(Mining)")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nVergleichsbild gespeichert: {OUT_PATH}")

    # ── 4. Gesamtzusammenfassung ─────────────────────────────────────────────
    all_probs = np.concatenate([p.flatten() for _, _, p in results])
    print(f"\nGesamt-Statistik Bono-Test-Region ({len(results)} Patches):")
    print(f"  ⌀ Mining-Wahrscheinlichkeit: {all_probs.mean()*100:.1f}%")
    print(f"  Anteil Pixel > {MINING_THRESH:.1f}: {(all_probs > MINING_THRESH).mean()*100:.1f}%")
    print(f"  Min: {all_probs.min():.3f}  Max: {all_probs.max():.3f}")


if __name__ == "__main__":
    main()
