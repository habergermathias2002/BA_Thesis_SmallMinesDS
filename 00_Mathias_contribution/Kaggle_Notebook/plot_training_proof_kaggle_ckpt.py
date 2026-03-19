"""
Vergleichsbild: Modell-Check mit Kaggle-Checkpoint (6 Beispiel-Patches).

Erstellt ein 2-spaltiges Bild mit 6 Zeilen:
  Links:  True-Color-Satellitenbild (10 m/px, B4/B3/B2)
  Rechts: Mining-Wahrscheinlichkeit des Kaggle-Trainings-Checkpoints

Eingaben:
- SmallMinesDS-Training-Patches (bereits als 6-Band-TIFs vorliegend)
- Kaggle-Checkpoint-Datei im Ordner Kaggle_Notebook (Pfad siehe CKPT_PATH)

Ausgabe:
- PNG im Ordner data/, z.B. data/model_proof_on_training_patches_kaggle.png
"""

import os
import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Training-Patches (6-Band nach Band-Fix)
TRAIN_DIR = os.path.join(REPO_ROOT, "data", "GhanaMiningPrithvi", "training")

# Pfad zum Kaggle-Checkpoint.
# Wir suchen automatisch nach der ersten .ckpt-Datei im Ordner Kaggle_Notebook,
# damit du den Dateinamen nicht manuell anpassen musst.
CKPT_DIR = os.path.join(REPO_ROOT, "Kaggle_Notebook")
_ckpts = [f for f in os.listdir(CKPT_DIR) if f.endswith(".ckpt")]
if not _ckpts:
    raise FileNotFoundError(
        f"Keine .ckpt-Datei im Ordner gefunden: {CKPT_DIR}\n"
        f"Bitte einen Kaggle-Checkpoint (z.B. best_checkpoint.ckpt) dort ablegen."
    )
# Falls mehrere vorhanden sind, nimm den ersten in sortierter Reihenfolge.
_ckpts.sort()
CKPT_FILENAME = _ckpts[0]
CKPT_PATH = os.path.join(CKPT_DIR, CKPT_FILENAME)
print(f"Verwende Checkpoint: {CKPT_PATH}")

# Suffix für Ausgabe, damit alte Proof-Bilder nicht überschrieben werden
OUTPUT_SUFFIX = "_kaggle_ckpt"
OUT_PATH = os.path.join(
    REPO_ROOT, "data", f"model_proof_on_training_patches{OUTPUT_SUFFIX}.png"
)

# Normalisierung (6 Bänder: B2,B3,B4,B8A,B11,B12) – identisch zu Training
MEANS = np.array(
    [1473.81388377, 1703.35249650, 1696.67685941, 3832.39764247, 3156.11122121, 2226.06822112],
    dtype=np.float32,
)
STDS = np.array(
    [223.43533204, 285.53613398, 413.82320306, 389.61483882, 451.49534791, 468.26765909],
    dtype=np.float32,
)

# 6-Band-Reihenfolge nach Fix: 0=Blue(B2), 1=Green(B3), 2=Red(B4), 3=B8A, 4=SWIR1, 5=SWIR2
R_IDX, G_IDX, B_IDX = 2, 1, 0  # True Color: Red(B4), Green(B3), Blue(B2)

# Beispiel-Patches: 3 Mining, 3 Non-Mining (wie im Original-Skript)
PATCHES = [
    ("GH_0122_2022_IMG.tif", 79.6),  # Mining
    ("GH_0079_2016_IMG.tif", 68.9),  # Mining
    ("GH_0105_2022_IMG.tif", 66.1),  # Mining
    ("GH_0001_2016_IMG.tif", 0.0),   # Non-Mining
    ("GH_0002_2016_IMG.tif", 0.0),   # Non-Mining
    ("GH_0004_2016_IMG.tif", 0.0),   # Non-Mining
]


def load_model():
    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {CKPT_PATH}")

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


def predict(task, img6: np.ndarray) -> np.ndarray:
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
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    return rgb


def main():
    task = load_model()
    n = len(PATCHES)
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3.5))
    fig.suptitle(
        "Modell-Check mit Kaggle-Checkpoint (SmallMinesDS Training-Patches)\n"
        "Links: Satellitenbild (10 m/px, True Color) | Rechts: Mining-Wahrscheinlichkeit",
        fontsize=12,
        y=1.01,
    )

    cmap_prob = mcolors.LinearSegmentedColormap.from_list("wp", ["white", "red"])

    for row, (fn, true_pct) in enumerate(PATCHES):
        ax_sat = axes[row, 0]
        ax_pred = axes[row, 1]

        img_path = os.path.join(TRAIN_DIR, fn)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Patch nicht gefunden: {img_path}")

        with rasterio.open(img_path) as s:
            img = s.read().astype(np.float32)  # (6, H, W)

        prob = predict(task, img)
        pred_pct = prob.mean() * 100.0

        ax_sat.imshow(truecolor(img), interpolation="nearest")
        ax_sat.set_title(
            f"{fn}\nGrund-Wahrheit: {true_pct:.1f}% Mining",
            fontsize=8,
            loc="left",
        )
        ax_sat.axis("off")

        im = ax_pred.imshow(
            prob, cmap=cmap_prob, vmin=0.0, vmax=1.0, interpolation="nearest"
        )
        ax_pred.set_title(
            f"Modell-Ausgabe (Kaggle-Checkpoint)\n⌀ {pred_pct:.1f}% Mining-Wahrscheinlichkeit",
            fontsize=8,
            loc="left",
        )
        ax_pred.axis("off")
        plt.colorbar(im, ax=ax_pred, fraction=0.046, pad=0.04, label="P(Mining)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()

