"""
Proof (alter Checkpoint): Modell funktioniert auf SmallMinesDS-Patches.
Liest die originalen 13-Band-TIFs aus der HuggingFace-Struktur und nimmt
die ersten 6 Bänder (0–5 = B2, B3, B4, B5, B6, B7), so wie der alte
Checkpoint (epoch=16) trainiert wurde.

Ausgabe: data/model_proof_on_training_patches_old_ckpt_13band.png
"""
import os, torch, rasterio, numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HF_ROOT    = os.path.join(REPO_ROOT, "Hugging_Face_Input", "SmallMinesDS")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
OUT_PATH   = os.path.join(REPO_ROOT, "data", "model_proof_on_training_patches_old_ckpt_13band.png")

OLD_CKPT   = "prithvi-v2-300-epoch=16-val_loss=0.0000.ckpt"

MEANS = np.array([1473.81, 1703.35, 1696.68, 3832.40, 3156.11, 2226.07], dtype=np.float32)
STDS  = np.array([ 223.44,  285.54,  413.82,  389.61,  451.50,  468.27], dtype=np.float32)

# 13-Band-Reihenfolge: 0=Blue, 1=Green, 2=Red, 3=RE1, 4=RE2, 5=RE3, 6=NIR, 7=B8A, ...
# Alter Checkpoint wurde auf Bändern 0–5 trainiert → True Color: R=2, G=1, B=0
R_IDX, G_IDX, B_IDX = 2, 1, 0

# Patches: (HuggingFace-Dateiname, Jahr, Grundwahrheit Mining-%)
PATCHES = [
    ("IMG_GH_0122_2022.tif", "2022", 79.6),
    ("IMG_GH_0079_2016.tif", "2016", 68.9),
    ("IMG_GH_0105_2022.tif", "2022", 66.1),
    ("IMG_GH_0001_2016.tif", "2016",  0.0),
    ("IMG_GH_0002_2016.tif", "2016",  0.0),
]


def load_model():
    from terratorch.tasks import SemanticSegmentationTask
    task = SemanticSegmentationTask.load_from_checkpoint(
        os.path.join(MODELS_DIR, OLD_CKPT),
        model_args={
            "backbone": "prithvi_eo_v2_300",
            "bands": ["BLUE", "GREEN", "RED", "VNIR_5", "SWIR_1", "SWIR_2"],
            "in_channels": 6, "num_classes": 2, "pretrained": False,
            "decoder": "UperNetDecoder", "rescale": True,
            "backbone_num_frames": 1, "head_dropout": 0.1,
            "decoder_scale_modules": True,
        },
        model_factory="PrithviModelFactory",
        loss="ce", lr=1e-3, ignore_index=-1,
        optimizer="AdamW", optimizer_hparams={"weight_decay": 0.05},
        freeze_backbone=True, class_names=["Non_mining", "Mining"],
    )
    task.eval()
    return task


def predict(task, img6):
    norm = (img6.astype(np.float32) - MEANS[:, None, None]) / STDS[:, None, None]
    t = torch.FloatTensor(norm).unsqueeze(0)
    with torch.no_grad():
        out = task.model(t)
        logits = out.output if hasattr(out, "output") else out
        probs = torch.softmax(logits, dim=1)
    return probs[0, 1].numpy()


def truecolor(img6):
    rgb = np.dstack((img6[R_IDX], img6[G_IDX], img6[B_IDX])).astype(np.float32)
    lo, hi = np.nanpercentile(rgb, (2, 98))
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    return rgb


def main():
    task = load_model()
    n = len(PATCHES)
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3.5))
    fig.suptitle(
        "Modell-Check (alter Checkpoint, 13-Band → [:6])\n"
        "Links: Satellitenbild (True Color) | Rechts: Mining-Wahrscheinlichkeit",
        fontsize=12, y=1.01,
    )

    cmap_prob = mcolors.LinearSegmentedColormap.from_list("wp", ["white", "red"])

    for row, (fn, year, true_pct) in enumerate(PATCHES):
        ax_sat = axes[row, 0]
        ax_pred = axes[row, 1]

        src_path = os.path.join(HF_ROOT, year, "IMAGE", fn)
        with rasterio.open(src_path) as s:
            img13 = s.read().astype(np.float32)  # (13, 128, 128)

        img6 = img13[:6]  # Bänder 0–5: B2, B3, B4, B5, B6, B7

        prob = predict(task, img6)
        pred_pct = prob.mean() * 100

        ax_sat.imshow(truecolor(img6), interpolation="nearest")
        ax_sat.set_title(
            f"{fn}\nGrund-Wahrheit: {true_pct:.0f}% Mining",
            fontsize=8, loc="left",
        )
        ax_sat.axis("off")

        im = ax_pred.imshow(prob, cmap=cmap_prob, vmin=0, vmax=1, interpolation="nearest")
        ax_pred.set_title(
            f"Modell-Ausgabe\n⌀ {pred_pct:.1f}% Mining-Wahrscheinlichkeit",
            fontsize=8, loc="left",
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
