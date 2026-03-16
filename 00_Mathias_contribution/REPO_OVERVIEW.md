# Repository-Übersicht: BA_Thesis_SmallMinesDS

## Projektbeschreibung

Dieses Repository enthält den Code und die Dokumentation zur Forschungsarbeit
**"SmallMinesDS: A Multi-Modal Satellite Image Dataset for Artisanal and Small-Scale Gold Mining Mapping"**,
veröffentlicht im *IEEE Geoscience and Remote Sensing Letters*
([DOI: 10.1109/LGRS.2025.3566356](https://ieeexplore.ieee.org/document/10982207)).

Das Projekt wurde an der Technischen Universität München (TUM) in Zusammenarbeit mit dem
Deutschen Zentrum für Luft- und Raumfahrt (DLR) durchgeführt.

### Wissenschaftliche Aufgabe

Der Kern der Arbeit ist **binäre semantische Segmentierung** von Satellitenbildern:
Jeder Pixel eines Bildausschnitts wird als entweder **Mining** (Artisanaler und
Kleinbergbau, ASGM) oder **Non-Mining** klassifiziert.

Untersucht werden dazu vier verschiedene Deep-Learning-Ansätze:

| Modell | Architektur | Spektralbänder | Vortraining |
|---|---|---|---|
| Prithvi-EO v2 300M | ViT-Backbone + UperNet | 6 (multispektral) | NASA/IBM geospatial |
| Prithvi-EO v2 600M | ViT-Backbone + UperNet | 6 (multispektral) | NASA/IBM geospatial |
| ResNet50 (from scratch) | ResNet50 + UNet | 6 (multispektral) | keines |
| ResNet50 (ImageNet) | ResNet50 + UNet | 3 (RGB) | ImageNet |
| SAM2-Hiera-Small | Hiera-ViT + Mask Decoder | 3 (RGB) | SA-1B (Meta SAM2) |

---

## Fähigkeiten des Repos

- **Daten-Laden und -Vorverarbeitung**: GeoTIFF-Patches lesen, normalisieren und mit
  Augmentierungen (horizontale/vertikale Spiegelungen) versehen
- **Modelltraining**: Fine-Tuning und Training von vier verschiedenen Segmentierungsmodellen
  auf dem SmallMinesDS-Datensatz
- **Evaluation**: Berechnung von mIoU (mean Intersection over Union) für Mining- und
  Non-Mining-Klassen auf dem Validierungsdatensatz
- **Experiment-Tracking**: Logging von Trainingsmetriken via TensorBoard
- **Modell-Checkpointing**: Speicherung der besten Modell-Gewichte während des Trainings

---

## Erwarteter Input

### Datensatz: SmallMinesDS

Verfügbar auf [HuggingFace](https://huggingface.co/datasets/ellaampy/SmallMinesDS).

**Abdeckung**: 5 Verwaltungsbezirke in Südwestghana (~3.200 km²),
zwei Zeitpunkte: Januar 2016 und Januar 2022.

**Gesamtanzahl Patches**: 4.270 (je 2.175 pro Jahrgang)

#### Dateiformat und Benennung

```
<name>_IMG.tif    ← Satellitenbildpatch (GeoTIFF)
<name>_MASK.tif   ← Binäre Segmentierungsmaske (GeoTIFF)
```

#### Bildstruktur

| Eigenschaft | Wert |
|---|---|
| Bildgröße | `13 × 128 × 128` Pixel (13 Bänder, 128×128 Pixel spatial) |
| Masken-Shape | `1 × 128 × 128` (binär: 0 = Non-Mining, 1 = Mining) |
| Dateiformat | GeoTIFF (`.tif`), gelesen mit `rasterio` |

#### Spektralbänder (6-Band-Modelle)

| Band | Name | Sensor |
|---|---|---|
| 1 | Blue | Sentinel-2 (optisch) |
| 2 | Green | Sentinel-2 |
| 3 | Red | Sentinel-2 |
| 4 | VNIR_5 (schmales NIR) | Sentinel-2 |
| 5 | SWIR_1 | Sentinel-2 |
| 6 | SWIR_2 | Sentinel-2 |

Die verbleibenden 7 Bänder im vollständigen 13-Band-Stack umfassen Radar-Bänder
(Sentinel-1 SAR) sowie weitere optische Bänder. SAM2 und der ImageNet-vortrainierte
ResNet50 verwenden ausschließlich die 3 RGB-Bänder (Red, Green, Blue).

#### Normalisierungsstatistiken (6-Band-Stack)

| Band | Mittelwert | Standardabweichung |
|---|---|---|
| Blue | 1473,81 | 223,44 |
| Green | 1703,35 | 285,54 |
| Red | 1696,68 | 413,82 |
| VNIR_5 | 3832,40 | 389,61 |
| SWIR_1 | 3156,11 | 451,50 |
| SWIR_2 | 2226,07 | 468,27 |

#### Erwartete Verzeichnisstruktur auf dem HPC-Cluster

```
GhanaMiningPrithvi/          ← 6-Band-Modelle (Prithvi, ResNet50-scratch)
    training/
        *_IMG.tif
        *_MASK.tif
    validation/
        *_IMG.tif
        *_MASK.tif

GhanaMiningRGB/              ← RGB-Subset für ft-resnet50
    training/
    validation/

GhanaMining3bands_final/     ← RGB-Subset für SAM2
    train_imgs/
    train_masks/
    val_imgs/
    val_masks/
```

---

## Generierter Output

| Output | Beschreibung |
|---|---|
| `output*/` Verzeichnisse | Modell-Checkpoints der besten Epoche (via PyTorch Lightning) |
| TensorBoard-Logs | Trainings- und Validierungsmetriken (Loss, mIoU) pro Epoche |
| `model_cocoa_10epochs.torch` | Gespeichertes SAM2-Modell (state dict) nach 10 Epochen |
| Konsolenausgabe | Test-mIoU-Werte für Mining- und Non-Mining-Klassen nach dem Training |

> Hinweis: Das `.gitignore` schließt alle `output*`-Verzeichnisse vom Versionieren aus.

---

## Datei-für-Datei-Erklärung

### Wurzelverzeichnis

---

#### `README.md`

Haupt-Dokumentationsdatei des Projekts. Enthält:
- Beschreibung des SmallMinesDS-Datensatzes und des Untersuchungsgebiets in Ghana
- Erklärung der Datenstruktur (Patches, Bänder, Masken)
- Setup-Anleitung für zwei Conda-Umgebungen (`terratorch` und `sam2`)
- Befehle zum Starten jedes der 5 Trainingsskripte
- BibTeX-Zitation für den IEEE-Artikel

---

#### `requirements.txt`

Vollständige, gepinnte Abhängigkeitsliste für die **`terratorch`**-Conda-Umgebung
(Python 3.11). Enthält alle nötigen Pakete für Training, Evaluation und
Datenverarbeitung, darunter:

| Paket | Version | Zweck |
|---|---|---|
| `terratorch` | 0.99.7 | IBM/NASA Geospatial-ML-Framework (Prithvi-Modelle) |
| `torch` / `torchvision` | 2.5.0 / 0.20.0 | Deep-Learning-Basis |
| `lightning` / `pytorch-lightning` | 2.4.0 | Training-Orchestrierung |
| `segmentation-models-pytorch` | 0.3.4 | ResNet50+UNet-Architektur |
| `rasterio` | 1.3.11 | GeoTIFF-Lesen |
| `albumentations` | 1.4.10 | Daten-Augmentierung |
| `mlflow` / `tensorboard` | 2.19.0 / 2.18.0 | Experiment-Tracking |

---

#### `install_sam2.sh`

Shell-Skript zur Einrichtung der separaten **SAM2-Umgebung**. Führt folgende Schritte aus:

1. Klont das offizielle SAM2-Repository von Meta (`https://github.com/facebookresearch/sam2`)
2. Installiert SAM2 im Editable-Mode (`pip install -e .`)
3. Installiert zusätzliche Abhängigkeiten: `rasterio` und `opencv-python`
4. Enthält einen kommentierten Hinweis, `ft-sam2.py` in das `sam2/`-Verzeichnis zu
   verschieben, da SAM2-Imports nur innerhalb des Paketverzeichnisses funktionieren

---

#### `.gitignore`

Enthält eine einzige Regel: `output*`

Verhindert, dass alle Trainings-Ausgabeverzeichnisse (Checkpoints, Logs) versioniert werden.

---

### `scripts/` — Trainingsskripte

---

#### `scripts/train-prithvi-v2-300.py`

**Zweck**: Fine-Tuning des **Prithvi-EO v2 300M** Geospatial-Foundation-Models auf dem
6-Band-SmallMinesDS-Datensatz.

**Technische Details**:
- Framework: TerraTorch (`PrithviModelFactory`)
- Backbone: `prithvi_eo_v2_300` (Vision Transformer, 300M Parameter), **eingefroren**
  (`freeze_backbone=True`) — nur der Decoder-Kopf wird trainiert
- Decoder: `UperNetDecoder` (Standard-Dense-Prediction-Head)
- Eingabe: 6 Spektralbänder (`in_channels=6`)
- Ausgabe: 2 Klassen (`Non_mining`, `Mining`)
- Loss: Cross-Entropy
- Optimizer: AdamW (lr=1e-3, weight_decay=0.05)
- Augmentierungen: horizontale + vertikale Spiegelungen (Albumentations)
- Training: bis zu 100 Epochen, Mixed Precision (fp16), 1 GPU, Batch-Size 4
- Output-Verzeichnis: `output-check-num-params/prithvi-v2-300-check-num-params`
- Abschließender `trainer.test()`-Aufruf zur Ausgabe der mIoU auf dem Validierungsset

---

#### `scripts/train-prithvi-v2-600.py`

**Zweck**: Identisch zu `train-prithvi-v2-300.py`, verwendet aber den größeren
**Prithvi-EO v2 600M** Backbone (`prithvi_eo_v2_600`).

Alle Hyperparameter, Augmentierungen, Loss-Funktion und Evaluations-Logik sind
identisch zum 300M-Skript. Output-Verzeichnis:
`output-check-params/prithvi-v2-600-check-params`.

---

#### `scripts/train-resnet50-6bands.py`

**Zweck**: Trainiert einen **ResNet50 + UNet von Grund auf** (ohne vortrainierte Gewichte)
auf dem 6-Band-Datensatz als Nicht-Foundation-Model-Baseline.

**Unterschiede zu den Prithvi-Skripten**:
- Framework: TerraTorch `SMPModelFactory` (wraps segmentation-models-pytorch)
- Backbone: `resnet50`, zufällige Initialisierung (kein `encoder_weights`)
- Architektur: Standard `Unet`
- **Klassen-Gewichte**: `[0.1, 0.9]` zur Behandlung des Klassenungleichgewichts
  (Mining-Flächen sind deutlich unterrepräsentiert)
- Niedrigere Lernrate: `lr=1e-4`
- Output-Verzeichnis: `output-scratch-6bands/resnet50-scratch-6bands`

---

#### `scripts/ft-resnet50.py`

**Zweck**: **Fine-Tuning eines ImageNet-vortrainierten ResNet50** als RGB-Baseline,
um einen fairen Vergleich mit SAM2 zu ermöglichen (beide nutzen nur RGB-Bänder).

**Unterschiede**:
- Eingabe: 3 Bänder (RED, GREEN, BLUE), `in_channels=3`
- Bandstatistiken in RGB-Reihenfolge umsortiert (Red, Green, Blue)
- Datenpfad: `GhanaMiningRGB` (vorab extrahiertes RGB-Subset)
- `encoder_weights: "imagenet"` — Initialisierung mit ImageNet-Gewichten
- Gleiche UNet-Architektur, gleiche Klassen-Gewichte `[0.1, 0.9]`
- Enthält Kommentar mit Google-Drive-Link zum besten Checkpoint (Epoche 57)
- Output-Verzeichnis: `output/resnet50`

---

#### `scripts/ft-sam2.py`

**Zweck**: **Fine-Tuning von Meta's Segment Anything Model v2 (SAM2)** — einem
prompt-basierten Vision-Foundation-Model — auf den RGB-Satellitenbildern.

**Architektur und Trainingsansatz**:
- Modell: `SAM2ImagePredictor` mit dem `sam2_hiera_small`-Checkpoint
- Nur **Mask Decoder** und **Prompt Encoder** werden trainiert;
  der Image Encoder (Backbone) bleibt eingefroren
- **Prompt-basierter Ansatz**: Für jede Klasse in einem Bild wird ein zufälliger Punkt
  aus der entsprechenden Klassenregion als Prompt-Punkt gesampelt
- **Loss**: Binäre Cross-Entropy-Segmentierungs-Loss + IoU-Score-Loss
  (gewichtet mit 0,05): `loss = seg_loss + score_loss * 0.05`
- Optimizer: AdamW (lr=1e-5, weight_decay=4e-5), Mixed Precision via `GradScaler`
- 10 Trainingsepochen; ein Sample pro Schritt (kein DataLoader mit Batches)
- Gespeichertes Modell: `model_cocoa_10epochs.torch`

**Wichtige Funktionen**:

| Funktion | Beschreibung |
|---|---|
| `load_data(images_dir, masks_dir)` | Paart `*_IMG.tif`- mit `*_MASK.tif`-Dateien |
| `read_batch(data, idx)` | Liest ein GeoTIFF-Bild (3 Bänder), extrahiert Masken und sampelt Prompt-Punkte |
| `read_batch_test(data)` | Wie `read_batch`, wählt aber ein zufälliges Sample |
| `evaluate_model(predictor, test_data)` | Führt vollständige SAM2-Inferenz aus und berechnet IoU |

---

### `figs/` — Abbildungen

---

#### `figs/study_area.png`

Karte der 5 Verwaltungsbezirke in Südwestghana, die vom Datensatz abgedeckt werden.
Dient als geografischer Überblick über das Untersuchungsgebiet.

---

#### `figs/multilayer_sample_patches.png`

Visuelle Beispiele der multimodalen Patches: zeigt mehrere Spektralschichten eines
Beispiel-Patches nebeneinander sowie die zugehörige binäre Mining-Maske.

---

#### `figs/order_of_bands.png`

Diagramm, das die Reihenfolge der 13 Spektralbänder in jedem Patch erklärt
(optische Bänder aus Sentinel-2 + Radar-Bänder aus Sentinel-1 SAR).

---

#### `figs/naming_convention.png`

Diagramm zur Datei-Benennungskonvention des Datensatzes:
`<name>_IMG.tif` für Bilder, `<name>_MASK.tif` für Masken.

---

### `00_Mathias_contribution/`

Verzeichnis für die eigenen Beiträge im Rahmen der Bachelorarbeit.

- **Skripte** (`scripts/`): Datenvorbereitung (01), Bono-Patch-Extraktion (02), Colab-Training (03), Inferenz Testregion (04) und gesamte Bono-Region (05), Ghana-Karte mit Galamsey-Overlay (06), True-Color-Mosaic (`make_bono_test_mosaic_png.py`), Galamsey-Heatmap (`plot_galamsey_probability_map.py`), **Modell-Check/Proof** (`plot_model_proof.py` – Satellitenbild + Weiß-Rot-Grafik auf SmallMinesDS-Patches), **Bono-Vergleichsbild** (`plot_bono_test_comparison.py` – gleiches Layout für Bono-Testregion, zeigt nicht plausible Vorhersage).
- **Colab Notebook:** `Colab Notebook/BA_Thesis_Model_Training_SmallMinesDS_data.ipynb` – Training Prithvi-EO v2 auf SmallMinesDS.
- **Dokumentation:** `Daily Progress/20260314_Documentation.md` – Pipeline, alle Skripte, Diagnose. **`Inference_warum_kein_Galamsey.md`** – Ursachenanalyse (Domain Shift, Normalisierung, Backbone, etc.) und empfohlene nächste Schritte.

---

## Technische Umgebungen

Das Projekt verwendet zwei separate Conda-Umgebungen:

| Umgebung | Skripte | Setup |
|---|---|---|
| `terratorch` | alle `scripts/*.py` außer `ft-sam2.py` | `pip install -r requirements.txt` |
| `sam2` | `scripts/ft-sam2.py` | `bash install_sam2.sh` |

---

## Referenz

```bibtex
@article{SmallMinesDS2025,
  title   = {SmallMinesDS: A Multi-Modal Satellite Image Dataset
             for Artisanal and Small-Scale Gold Mining Mapping},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  year    = {2025},
  doi     = {10.1109/LGRS.2025.3566356}
}
```
