"""
Zeigt die Galamsey-Wahrscheinlichkeit (P(Mining)) pro Pixel als eine Grafik
mit Farbskala. Nutzt prediction_prob.tif aus patches_bono_test; die Skala
wird an den tatsächlichen Wertebereich angepasst, damit auch sehr kleine
Wahrscheinlichkeiten (z. B. 1e-6) räumlich sichtbar sind.

Output: data/patches_bono_test/galamsey_probability_map.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATCHES_DIR = os.path.join(REPO_ROOT, "data", "patches_bono_test")
PROB_PATH = os.path.join(PATCHES_DIR, "prediction_prob.tif")
OUT_PATH = os.path.join(PATCHES_DIR, "galamsey_probability_map.png")


def main():
    if not os.path.exists(PROB_PATH):
        print(f"Fehler: {PROB_PATH} nicht gefunden. Zuerst 04_inference_bono.py ausführen.")
        return

    with rasterio.open(PROB_PATH) as src:
        prob = src.read(1)  # (H, W)

    p_min, p_max = float(np.nanmin(prob)), float(np.nanmax(prob))
    print(f"Galamsey-Wahrscheinlichkeit: min = {p_min:.2e}, max = {p_max:.2e}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Skala an echten Wertebereich anpassen, damit räumliche Struktur sichtbar wird
    vmax = max(p_max, 1e-9)  # mind. 1e-9, damit keine div by zero
    im = ax.imshow(prob, cmap="YlOrRd", vmin=0, vmax=vmax, interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label("Galamsey-Wahrscheinlichkeit P(Mining)", fontsize=12)
    ax.set_title("Modell-Ausgabe: Mining-Wahrscheinlichkeit pro Pixel\n(Testgebiet 5×5 km, 10 m/px)", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gespeichert: {OUT_PATH}")


if __name__ == "__main__":
    main()
