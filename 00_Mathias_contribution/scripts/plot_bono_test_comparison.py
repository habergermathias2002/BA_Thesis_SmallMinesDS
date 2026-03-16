"""
Vergleichsbild Bono-Testregion (wie Proof, aber für Test-Patches).
Links: 10×10 m Satellitenbild der 5×5 km Bono-Testfläche (True Color).
Rechts: Mining-Wahrscheinlichkeit des Modells (weiß = kein Mining, rot = Mining).
Zeigt im gleichen Layout wie model_proof, dass die Vorhersage hier nicht plausibel ist (Domain Shift).

Voraussetzung: 02_extract_bono_test_patches.py und 04_inference_bono.py ausgeführt.
Ausgabe: data/patches_bono_test/bono_test_comparison.png
"""
import os
import csv
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATCHES_DIR = os.path.join(REPO_ROOT, "data", "patches_bono_test")
PATCH_SIZE = 128
OUT_SIZE = 4 * PATCH_SIZE  # 512
# Suffix muss zu 04_inference_bono.py passen (dort OUTPUT_SUFFIX); "" = prediction_prob.tif
OUTPUT_SUFFIX = "_6band"
PROB_PATH = os.path.join(PATCHES_DIR, f"prediction_prob{OUTPUT_SUFFIX}.tif")
OUT_PATH = os.path.join(PATCHES_DIR, f"bono_test_comparison{OUTPUT_SUFFIX}.png")

# Band indices 0-based: 0=Blue, 1=Green, 2=Red → true color
R, G, B = 2, 1, 0


def load_truecolor():
    """Stitch 16 Bono patches to 512×512 true-color (0–1 float)."""
    index_path = os.path.join(PATCHES_DIR, "patch_index.csv")
    with open(index_path) as f:
        records = list(csv.DictReader(f))
    rgb = np.zeros((OUT_SIZE, OUT_SIZE, 3), dtype=np.float32)
    for rec in records:
        path = os.path.join(PATCHES_DIR, rec["patch_file"])
        row_s = int(rec["row_px_start"])
        col_s = int(rec["col_px_start"])
        with rasterio.open(path) as src:
            patch = src.read()
        rgb[row_s : row_s + PATCH_SIZE, col_s : col_s + PATCH_SIZE, 0] = patch[R]
        rgb[row_s : row_s + PATCH_SIZE, col_s : col_s + PATCH_SIZE, 1] = patch[G]
        rgb[row_s : row_s + PATCH_SIZE, col_s : col_s + PATCH_SIZE, 2] = patch[B]
    lo, hi = np.nanpercentile(rgb, (2, 98))
    rgb = np.clip((rgb.astype(np.float64) - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    return rgb


def main():
    if not os.path.exists(PROB_PATH):
        print(f"Fehler: {PROB_PATH} nicht gefunden. Zuerst 04_inference_bono.py ausführen.")
        return

    rgb = load_truecolor()
    with rasterio.open(PROB_PATH) as src:
        prob = src.read(1)

    p_min, p_max = float(np.nanmin(prob)), float(np.nanmax(prob))
    p_mean = float(np.nanmean(prob)) * 100
    print(f"Bono Testregion – P(Mining): min={p_min:.2e}, max={p_max:.2e}, ⌀={p_mean:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        "Bono-Testregion (5×5 km) – gleiches Layout wie Proof\n"
        "Links: Satellitenbild (10 m/px, True Color) | Rechts: Mining-Wahrscheinlichkeit (Domain Shift → nicht plausibel)",
        fontsize=11, y=1.02,
    )

    cmap_prob = mcolors.LinearSegmentedColormap.from_list("wp", ["white", "red"])

    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("Bono-Testgebiet\n(bekannter Galamsey-Standort, Januar 2025)", fontsize=10, loc="left")
    axes[0].axis("off")

    im = axes[1].imshow(prob, cmap=cmap_prob, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(
        f"Modell-Ausgabe\n⌀ {p_mean:.1f}% Mining-Wahrscheinlichkeit (nicht plausibel)",
        fontsize=10, loc="left",
    )
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="P(Mining)")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
