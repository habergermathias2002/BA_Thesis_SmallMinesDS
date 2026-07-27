"""
Erzeugt 10 Proof-Bilder: je 1 PNG pro SmallMinesDS-Trainingspatch mit 4 Panels.

Panels (von links nach rechts):
  1. Satellitenbild (True Color, 10 m/px)
  2. Ground Truth (Label-Maske)
  3. Modell P(Mining) – Wahrscheinlichkeitskarte
  4. Binäre Vorhersage (Threshold 0.5)

Ausgabe: 00_Mathias_contribution/Model_Proof_Training/patches/*.png

Voraussetzung:
  - 6-Band-Training-Daten in data/GhanaMiningPrithvi/training/
  - Kaggle-Checkpoint in 00_Mathias_contribution/Kaggle_Notebook/*.ckpt
    (Fallback: models/*.ckpt)
"""
import os
import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIR = os.path.join(REPO_ROOT, "data", "GhanaMiningPrithvi", "training")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches")

MEANS = np.array(
    [1473.81388377, 1703.35249650, 1696.67685941, 3832.39764247, 3156.11122121, 2226.06822112],
    dtype=np.float32,
)
STDS = np.array(
    [223.43533204, 285.53613398, 413.82320306, 389.61483882, 451.49534791, 468.26765909],
    dtype=np.float32,
)

R_IDX, G_IDX, B_IDX = 2, 1, 0
THRESHOLD = 0.5

# 10 diverse Patches: Non-Mining, wenig Mining, mittel, viel Mining; 2016 + 2022
PATCHES = [
    "GH_0001_2016",  # 0.0%  – reines Non-Mining
    "GH_0002_2016",  # 0.0%  – reines Non-Mining
    "GH_0004_2016",  # 0.0%  – reines Non-Mining
    "GH_0354_2022",  # 0.1%  – minimal Mining
    "GH_1173_2022",  # 5.0%  – wenig Mining
    "GH_1952_2016",  # 30.0% – mittlerer Mining-Anteil
    "GH_0080_2022",  # 30.0% – mittlerer Mining-Anteil
    "GH_0079_2016",  # 68.9% – viel Mining
    "GH_0122_2022",  # 79.6% – viel Mining
    "GH_0865_2016",  # 70.9% – viel Mining
]


def find_checkpoint():
    candidates = [
        os.path.join(REPO_ROOT, "00_Mathias_contribution", "Kaggle_Notebook"),
        os.path.join(REPO_ROOT, "models"),
    ]
    for folder in candidates:
        if not os.path.isdir(folder):
            continue
        ckpts = sorted(f for f in os.listdir(folder) if f.endswith(".ckpt"))
        if ckpts:
            path = os.path.join(folder, ckpts[0])
            print(f"Checkpoint: {path}")
            return path
    raise FileNotFoundError("Kein .ckpt in Kaggle_Notebook/ oder models/ gefunden.")


def load_model(ckpt_path):
    from terratorch.tasks import SemanticSegmentationTask

    task = SemanticSegmentationTask.load_from_checkpoint(
        ckpt_path,
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


def predict(task, img6):
    norm = (img6.astype(np.float32) - MEANS[:, None, None]) / STDS[:, None, None]
    t = torch.from_numpy(norm).float().unsqueeze(0)
    with torch.no_grad():
        out = task.model(t)
        logits = out.output if hasattr(out, "output") else out
        probs = torch.softmax(logits, dim=1)
    return probs[0, 1].cpu().numpy()


def truecolor(img6):
    rgb = np.dstack((img6[R_IDX], img6[G_IDX], img6[B_IDX])).astype(np.float32)
    lo, hi = np.nanpercentile(rgb, (2, 98))
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def mining_pct(mask):
    return float((mask > 0).mean() * 100)


def iou_binary(gt, pred):
    gt_b = gt > 0
    pred_b = pred > 0
    inter = np.logical_and(gt_b, pred_b).sum()
    union = np.logical_or(gt_b, pred_b).sum()
    return float(inter / union) if union > 0 else 1.0


def render_patch(task, patch_id, cmap_prob, cmap_mask):
    img_path = os.path.join(TRAIN_DIR, f"{patch_id}_IMG.tif")
    mask_path = os.path.join(TRAIN_DIR, f"{patch_id}_MASK.tif")

    with rasterio.open(img_path) as src:
        img = src.read().astype(np.float32)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    prob = predict(task, img)
    pred_bin = (prob >= THRESHOLD).astype(np.uint8)
    gt_pct = mining_pct(mask)
    pred_pct = float(prob.mean() * 100)
    patch_iou = iou_binary(mask, pred_bin)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    fig.suptitle(
        f"{patch_id}  |  GT: {gt_pct:.1f}% Mining  |  Modell: {pred_pct:.1f}%  |  IoU: {patch_iou:.2f}",
        fontsize=11,
        y=1.02,
    )

    axes[0].imshow(truecolor(img), interpolation="nearest")
    axes[0].set_title("1. Satellitenbild\n(True Color, 10 m/px)", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(mask, cmap=cmap_mask, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"2. Ground Truth\n{gt_pct:.1f}% Mining", fontsize=9)
    axes[1].axis("off")

    im = axes[2].imshow(prob, cmap=cmap_prob, vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title(f"3. Modell P(Mining)\nØ {pred_pct:.1f}%", fontsize=9)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="P(Mining)")

    axes[3].imshow(pred_bin, cmap=cmap_mask, vmin=0, vmax=1, interpolation="nearest")
    axes[3].set_title(f"4. Binäre Vorhersage\n(Threshold {THRESHOLD})", fontsize=9)
    axes[3].axis("off")

    legend_handles = [
        mpatches.Patch(color="white", ec="gray", label="Non-Mining"),
        mpatches.Patch(color="red", label="Mining"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=9, frameon=False)

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    out_path = os.path.join(OUT_DIR, f"{patch_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path, gt_pct, pred_pct, patch_iou


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt = find_checkpoint()
    task = load_model(ckpt)

    cmap_prob = mcolors.LinearSegmentedColormap.from_list("wp", ["white", "red"])
    cmap_mask = mcolors.ListedColormap(["white", "red"])

    print(f"Erzeuge {len(PATCHES)} Proof-Bilder in {OUT_DIR}\n")
    results = []
    for patch_id in PATCHES:
        out_path, gt_pct, pred_pct, patch_iou = render_patch(
            task, patch_id, cmap_prob, cmap_mask
        )
        results.append((patch_id, gt_pct, pred_pct, patch_iou, out_path))
        print(f"  ✓ {patch_id}: GT={gt_pct:.1f}%, Pred={pred_pct:.1f}%, IoU={patch_iou:.2f}")

    mean_iou = np.mean([r[3] for r in results])
    print(f"\nFertig. Mittleres IoU über {len(results)} Patches: {mean_iou:.3f}")
    print(f"Ausgabeordner: {OUT_DIR}")


if __name__ == "__main__":
    main()
