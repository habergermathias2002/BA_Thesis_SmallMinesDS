"""
04_inference_bono_2.0.py
========================

PROJEKT: Mapping von Artisanal and Small-Scale Gold Mines (Galamsey)
BESCHREIBUNG: Inferenz-Script unter Verwendung des Prithvi-v2-300 Foundation Models.

METHODIK (Domain Adaptation):
Dieses Script implementiert ein "Z-Score Domain Alignment". Da die Inferenz-Daten (Bono 2025) 
eine signifikant andere radiometrische Verteilung (Domain Shift) aufweisen als die 
Original-Trainingsdaten (SmallMinesDS 2016/2022), führt eine standardmäßige 
Normalisierung zu Fehlklassifikationen. 

Der Fix transformiert die Pixelwerte wie folgt:
1. Lokale Zentrierung: (Pixel - Bono_Mean) / Bono_Std
   Dies bringt die Daten in einen "neutralen" statistischen Raum (Mittelwert 0, Varianz 1).
2. Da der Prithvi-Encoder auf ebendiese Z-Scores trainiert wurde, wird so die 
   Kompatibilität zwischen den Zeitpunkten (2022 vs. 2025) wiederhergestellt.

OUTPUT:
  - prediction_binary_aligned.tif (Binäre Maske: 0=Kein Mining, 1=Mining)
  - prediction_prob_aligned.tif   (Wahrscheinlichkeitskarte 0.0-1.0)
  - prediction_visualization_aligned.png (Visueller Vergleich)
"""

import os
import csv
import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── PFADE & KONFIGURATION ────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(_script_dir))
PATCHES_DIR = os.path.join(REPO_ROOT, "data", "patches_bono_test")
MODELS_DIR  = os.path.join(REPO_ROOT, "models")

# Automatische Wahl des Checkpoints
if not os.path.isdir(MODELS_DIR):
    raise FileNotFoundError(f"Models-Ordner nicht gefunden: {MODELS_DIR}")
_ckpts = [f for f in os.listdir(MODELS_DIR) if f.endswith(".ckpt")]
CHECKPOINT_PATH = os.path.join(MODELS_DIR, _ckpts[0]) if _ckpts else os.path.join(MODELS_DIR, "prithvi-v2-300-best.ckpt")

# ── RADIOMETRISCHE STATISTIKEN (WICHTIG FÜR DOMAIN ALIGNMENT) ────────────────
# Referenz-Statistiken aus dem Training (SmallMinesDS)
TRAIN_MEANS = np.array([1473.81, 1703.35, 1696.68, 3832.40, 3156.11, 2226.07], dtype=np.float32)
TRAIN_STDS  = np.array([ 223.44,  285.54,  413.82,  389.61,  451.50,  468.27], dtype=np.float32)

# Beobachtete Statistiken der Bono-Region (Inferenz-Daten 2025)
# Diese Werte korrigieren den Helligkeitsunterschied zwischen den Datensätzen.
BONO_MEANS  = np.array([ 583.63,  851.72, 1241.71, 2411.21, 3027.37, 2290.58], dtype=np.float32)
BONO_STDS   = np.array([ 157.83,  227.45,  348.81,  717.34,  828.90,  609.14], dtype=np.float32)

PATCH_SIZE     = 128
MINING_THRESH  = 0.5   # Schwellenwert für die binäre Maske

def load_model(checkpoint_path):
    """Lädt das feinjustierte Prithvi-Modell aus dem Lightning Checkpoint."""
    from terratorch.tasks import SemanticSegmentationTask
    from terratorch.models import PrithviModelFactory

    ghana_mining_bands = ["BLUE", "GREEN", "RED", "VNIR_5", "SWIR_1", "SWIR_2"]
    model_args = {
        "backbone":              "prithvi_eo_v2_300",
        "bands":                 ghana_mining_bands,
        "in_channels":           6,
        "num_classes":           2,
        "pretrained":            False,
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
    print(f"Modell erfolgreich geladen auf: {device}")
    return task, device

def align_and_normalize(patch):
    """
    Führt das Z-Score Domain Alignment durch. 
    Bringt das Bono-Patch in den statistischen Raum, den das Modell im Training gelernt hat.
    """
    # Schritt: Zentrierung auf Basis der Bono-Statistik
    # Formel: (x - mean_local) / std_local
    neutral = (patch - BONO_MEANS[:, None, None]) / BONO_STDS[:, None, None]
    return neutral

def run_inference(task, device, patches_dir):
    """Führt Inferenz für alle Patches im Verzeichnis aus."""
    index_path = os.path.join(patches_dir, "patch_index.csv")
    with open(index_path) as f:
        records = list(csv.DictReader(f))

    max_row = max(int(r["row"]) for r in records)
    max_col = max(int(r["col"]) for r in records)
    n_rows, n_cols = max_row + 1, max_col + 1

    prob_map   = np.zeros((n_rows * PATCH_SIZE, n_cols * PATCH_SIZE), dtype=np.float32)
    binary_map = np.zeros((n_rows * PATCH_SIZE, n_cols * PATCH_SIZE), dtype=np.uint8)

    print(f"Starte Aligned Inference auf {len(records)} Patches ({n_rows}x{n_cols} Gitter)...")

    for i, rec in enumerate(records):
        patch_path = os.path.join(patches_dir, rec["patch_file"])
        with rasterio.open(patch_path) as src:
            patch = src.read().astype(np.float32) 

        # Domain Alignment anwenden
        patch_norm = align_and_normalize(patch)

        # Vorhersage
        tensor = torch.FloatTensor(patch_norm).unsqueeze(0).to(device)
        with torch.no_grad():
            out = task.model(tensor)
            logits = out.output if hasattr(out, "output") else out
            probs  = torch.softmax(logits, dim=1)
            mining_prob = probs[0, 1].cpu().numpy()

        # In die Gesamtkarte einfügen
        r, c = int(rec["row"]), int(rec["col"])
        rs, cs = r * PATCH_SIZE, c * PATCH_SIZE
        prob_map[rs:rs+PATCH_SIZE, cs:cs+PATCH_SIZE] = mining_prob
        binary_map[rs:rs+PATCH_SIZE, cs:cs+PATCH_SIZE] = (mining_prob >= MINING_THRESH).astype(np.uint8)

        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            print(f"  {i+1}/{len(records)} Patches verarbeitet...")

    return prob_map, binary_map, records

def get_full_transform(records):
    """Rekonstruiert die Georeferenzierung."""
    tl = [r for r in records if int(r["row"]) == 0 and int(r["col"]) == 0][0]
    return rasterio.transform.Affine(10.0, 0.0, float(tl["utm_left"]), 0.0, -10.0, float(tl["utm_top"]))

def save_geotiff(path, data, transform, count=1, dtype="float32"):
    """Speichert das Ergebnis als GeoTIFF."""
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=count, dtype=dtype, crs="EPSG:32630", transform=transform) as dst:
        dst.write(data[np.newaxis, :, :] if data.ndim == 2 else data)

def save_visualization(patches_dir, prob_map, binary_map):
    """Erstellt ein side-by-side Vergleichsbild."""
    # Lade Referenz-Patch für Farbdarstellung (SWIR1/NIR/Red)
    ref_path = os.path.join(patches_dir, "patch_0000_r0_c0.tif")
    with rasterio.open(ref_path) as src:
        p = src.read().astype(np.float32)
    fc = np.dstack((p[4], p[3], p[2]))
    fc = np.clip(fc / np.nanpercentile(fc, 98) * 0.8, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(fc); axes[0].set_title("Echtfarben (Referenz)"); axes[0].axis("off")
    im = axes[1].imshow(prob_map, cmap="YlOrRd", vmin=0, vmax=1)
    axes[1].set_title("Mining Wahrscheinlichkeit (Aligned)"); axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cmap_bin = mcolors.ListedColormap(["#2d6a4f", "#d62828"])
    axes[2].imshow(binary_map, cmap=cmap_bin); axes[2].set_title("Binäre Vorhersage (Rot=Mining)"); axes[2].axis("off")
    
    out_path = os.path.join(patches_dir, "prediction_visualization_aligned.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Visualisierung gespeichert: {out_path}")

def main():
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"FEHLER: Checkpoint nicht gefunden.")
        return

    task, device = load_model(CHECKPOINT_PATH)
    prob_map, binary_map, records = run_inference(task, device, PATCHES_DIR)
    transform = get_full_transform(records)

    save_geotiff(os.path.join(PATCHES_DIR, "prediction_prob_aligned.tif"), prob_map, transform)
    save_geotiff(os.path.join(PATCHES_DIR, "prediction_binary_aligned.tif"), binary_map, transform, dtype="uint8")
    save_visualization(PATCHES_DIR, prob_map, binary_map)

    print(f"\nInferenz abgeschlossen. Mining-Anteil in der Region: {binary_map.mean()*100:.2f}%")

if __name__ == "__main__":
    main()
    