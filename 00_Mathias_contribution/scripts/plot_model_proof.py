"""
Proof: Modell funktioniert auf SmallMinesDS-Patches.
Zeigt für 5 Patches jeweils:
  Links:  10×10 m Satellitenbild (True Color)
  Rechts: Mining-Wahrscheinlichkeit (weiß = kein Mining, rot = Mining)
Ausgabe: data/model_proof_on_training_patches.png

HINWEIS: Erwartet 6-Band-GeoTIFFs (nach dem Band-Fix in 01_prepare_dataset.py).
  Band 0=Blue(B2), 1=Green(B3), 2=Red(B4), 3=B8A, 4=SWIR1(B11), 5=SWIR2(B12)
"""
import os, torch, rasterio, numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_DIR  = os.path.join(REPO_ROOT, "data", "GhanaMiningPrithvi", "training")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
# Suffix für Ausgabe, damit alte Proof-Bilder nicht überschrieben werden (z. B. "_6band"; "" = überschreiben)
OUTPUT_SUFFIX = "_6band"
OUT_PATH   = os.path.join(REPO_ROOT, "data", f"model_proof_on_training_patches{OUTPUT_SUFFIX}.png")

MEANS = np.array([1473.81, 1703.35, 1696.68, 3832.40, 3156.11, 2226.07], dtype=np.float32)
STDS  = np.array([ 223.44,  285.54,  413.82,  389.61,  451.50,  468.27], dtype=np.float32)

# 6-Band-Reihenfolge nach Fix: 0=Blue(B2), 1=Green(B3), 2=Red(B4), 3=B8A, 4=SWIR1, 5=SWIR2
R_IDX, G_IDX, B_IDX = 2, 1, 0  # True Color: Red(B4), Green(B3), Blue(B2)

# Test-Patches: je ein Mining-Patch und ein Non-Mining-Patch
PATCHES = [
    ("GH_0122_2022_IMG.tif", 79.6),  # Mining
    ("GH_0079_2016_IMG.tif", 68.9),  # Mining
    ("GH_0105_2022_IMG.tif", 66.1),  # Mining
    ("GH_0001_2016_IMG.tif",  0.0),  # Non-Mining
    ("GH_0002_2016_IMG.tif",  0.0),  # Non-Mining
]


def load_model():
    ckpt = [f for f in os.listdir(MODELS_DIR) if f.endswith(".ckpt")][0]
    from terratorch.tasks import SemanticSegmentationTask
    task = SemanticSegmentationTask.load_from_checkpoint(
        os.path.join(MODELS_DIR, ckpt),
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
        "Modell-Check: SmallMinesDS Training-Patches\n"
        "Links: Satellitenbild (10 m/px, True Color) | Rechts: Mining-Wahrscheinlichkeit",
        fontsize=12, y=1.01,
    )

    cmap_prob = mcolors.LinearSegmentedColormap.from_list("wp", ["white", "red"])

    for row, (fn, true_pct) in enumerate(PATCHES):
        ax_sat = axes[row, 0]
        ax_pred = axes[row, 1]

        with rasterio.open(os.path.join(TRAIN_DIR, fn)) as s:
            img = s.read().astype(np.float32)  # (6, H, W) after band fix
        mask_fn = fn.replace("_IMG.tif", "_MASK.tif")
        with rasterio.open(os.path.join(TRAIN_DIR, mask_fn)) as s:
            mask = s.read(1)

        prob = predict(task, img)
        pred_pct = prob.mean() * 100

        ax_sat.imshow(truecolor(img), interpolation="nearest")
        ax_sat.set_title(
            f"{fn}\nGrund-Wahrheit: {true_pct:.0f}% Mining",
            fontsize=8, loc="left",
        )
        ax_sat.axis("off")

        # Probability map
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
