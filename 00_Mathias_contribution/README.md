# BA_Thesis_SmallMinesDS — Vollständige Repository-Dokumentation

**Bachelorarbeit:** Fernerkundungsbasierte Kartierung von Galamsey (Artisanal and Small-Scale Gold Mining, ASGM) in den Regionen Bono/Bono East (Ghana) via Fine-Tuning eines Geospatial Foundation Models (Prithvi-EO v2), verknüpft mit Haushalts-Mikrodaten von Cashew-Farmern für eine ökonometrische Wirkungsanalyse.

Dieses Dokument beschreibt **das gesamte Repository** — sowohl die wissenschaftliche Basis (das Open-Source-Paper *SmallMinesDS*) als auch **alle eigenen Beiträge** (`00_Mathias_contribution/`) im Detail: Training, Zero-Shot-Test, Fine-Tuning, flächige Inferenz, Verknüpfung mit Mikrodaten und ökonometrische Analyse.

---

## Inhaltsverzeichnis

1. [Big Picture: Was macht dieses Projekt?](#1-big-picture-was-macht-dieses-projekt)
2. [Teil A — Basis: SmallMinesDS (Paper, fremdes Vorwissen)](#2-teil-a--basis-smallminesds-paper-fremdes-vorwissen)
3. [Teil B — Eigener Beitrag: Die Pipeline im Überblick](#3-teil-b--eigener-beitrag-die-pipeline-im-überblick)
4. [Schritt 1: Trainingsdaten vorbereiten & Base-Modell trainieren](#4-schritt-1-trainingsdaten-vorbereiten--base-modell-trainieren)
5. [Schritt 2: Neue Region beschaffen (Google Earth Engine Export)](#5-schritt-2-neue-region-beschaffen-google-earth-engine-export)
6. [Schritt 3: Zero-Shot-Test des Base-Modells auf Bono](#6-schritt-3-zero-shot-test-des-base-modells-auf-bono)
7. [Schritt 4: Manuelles Labeling der Bono-Region](#7-schritt-4-manuelles-labeling-der-bono-region)
8. [Schritt 5: Fine-Tuning auf Bono-Labels](#8-schritt-5-fine-tuning-auf-bono-labels)
9. [Schritt 6: Evaluation — Base vs. Fine-Tuned](#9-schritt-6-evaluation--base-vs-fine-tuned)
10. [Schritt 7: Flächige Inferenz über ganz Bono/Bono East](#10-schritt-7-flächige-inferenz-über-ganz-bonobono-east)
11. [Schritt 8: Verknüpfung mit Mikrodaten (Spatial Linkage)](#11-schritt-8-verknüpfung-mit-mikrodaten-spatial-linkage)
12. [Schritt 9: Ökonometrische Analyse](#12-schritt-9-ökonometrische-analyse)
13. [Vollständige Ordnerstruktur](#13-vollständige-ordnerstruktur)
14. [Technische Umgebungen (Conda-Environments)](#14-technische-umgebungen-conda-environments)
15. [Große / sensible Dateien (nicht versioniert)](#15-große--sensible-dateien-nicht-versioniert)
16. [Reproduktion Schritt für Schritt (Kurzfassung)](#16-reproduktion-schritt-für-schritt-kurzfassung)

---

## 1. Big Picture: Was macht dieses Projekt?

Die Arbeit beantwortet zwei aufeinander aufbauende Fragen:

1. **Remote-Sensing-Frage:** Kann ein vortrainiertes Satellitenbild-Foundation-Model (Prithvi-EO v2), das ursprünglich auf Südwest-Ghana (2016/2022) trainiert wurde, per Fine-Tuning auf eine neue Region (Bono/Bono East, Januar 2025) übertragen werden, um Galamsey (illegalen Kleinbergbau) automatisiert zu kartieren?
2. **Ökonomische Frage:** Hängt die räumliche Nähe/Exposition von Cashew-Farmern zu kartiertem Galamsey mit ihrem wirtschaftlichen Verhalten zusammen (Einkommensanteil aus Cashew, Umweltbesorgnis, Teilnahme an Nachhaltigkeitsprogrammen)?

Der Workflow ist daher zweigeteilt:

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│   TEIL 1: REMOTE SENSING     │        │   TEIL 2: ÖKONOMETRIE         │
│   (Bildanalyse, Deep Learning)│  ───▶  │   (Mikrodaten-Regression)     │
│                              │  Karte  │                              │
│  Training → Zero-Shot →      │  als    │  Distanz/Fläche zu Galamsey  │
│  Labeling → Fine-Tuning →    │  Input  │  je Farmer berechnen  →      │
│  Flächige Inferenz           │         │  OLS-Regressionen            │
└─────────────────────────────┘        └──────────────────────────────┘
```

---

## 2. Teil A — Basis: SmallMinesDS (Paper, fremdes Vorwissen)

Das Wurzelverzeichnis des Repos enthält den **öffentlichen Code und Datensatz** aus:

> Ofori-Ampofo, Zappacosta, Kuzu, Schauer, Willberg, Zhu (2025): *SmallMinesDS: A Multi-Modal Dataset for Mapping Artisanal and Small-Scale Gold Mines*, IEEE Geoscience and Remote Sensing Letters. [DOI: 10.1109/LGRS.2025.3566356](https://ieeexplore.ieee.org/document/10982207)

Das ist **nicht mein eigener Beitrag**, sondern die wissenschaftliche/technische Grundlage, auf der die Bachelorarbeit aufbaut (Foundation Model + Trainingscode + Original-Trainingsgebiet).

### Was der Datensatz enthält

- **Untersuchungsgebiet (Original):** 5 Verwaltungsbezirke in Südwest-Ghana, ca. 3.200 km², **nicht** Bono/Bono East (das ist die *neue* Region, auf die in dieser Arbeit übertragen wird)
- **Zeitpunkte:** Januar 2016 und Januar 2022 (Trockenzeit, wolkenfrei)
- **4.270 Patches** (2.175 je Jahrgang), je `13 × 128 × 128` Pixel (Bild) + `1 × 128 × 128` (binäre Mining-Maske)
- **13 Bänder** pro Patch: optische Sentinel-2-Bänder (Blue, Green, Red, RE1–3, NIR, B8A, SWIR1/2) + Sentinel-1-SAR (VV, VH) + DEM
- Download: [HuggingFace](https://huggingface.co/datasets/ellaampy/SmallMinesDS)

### Vier verglichene Deep-Learning-Modelle (aus dem Paper)

| Modell | Architektur | Bänder | Vortraining | Trainingsskript |
|---|---|---|---|---|
| **Prithvi-EO v2 300M** | ViT-Backbone + UperNet-Decoder | 6 (multispektral) | NASA/IBM Geospatial Foundation Model | `scripts/train-prithvi-v2-300.py` |
| Prithvi-EO v2 600M | ViT-Backbone + UperNet-Decoder | 6 (multispektral) | NASA/IBM Geospatial Foundation Model | `scripts/train-prithvi-v2-600.py` |
| ResNet50 (from scratch) | ResNet50 + U-Net | 6 (multispektral) | keines | `scripts/train-resnet50-6bands.py` |
| ResNet50 (ImageNet) | ResNet50 + U-Net | 3 (RGB) | ImageNet | `scripts/ft-resnet50.py` |
| SAM2-Hiera-Small | Hiera-ViT + Mask Decoder | 3 (RGB) | SA-1B (Meta) | `scripts/ft-sam2.py` |

**Für diese Bachelorarbeit relevant ist ausschließlich Prithvi-EO v2 300M**, da es im Paper die beste Performance zeigte und als einziges Modell für Transfer-Learning auf eine neue Region weiterverwendet wurde.

### Prithvi-EO v2 300M im Detail (Basistraining, `scripts/train-prithvi-v2-300.py`)

- Backbone: `prithvi_eo_v2_300` (Vision Transformer, ~300M Parameter), **eingefroren** — nur der Decoder-Kopf wird trainiert
- Decoder: `UperNetDecoder`
- Input: 6 Bänder (Blue, Green, Red, VNIR_5/B8A, SWIR_1, SWIR_2), Output: 2 Klassen (Non-Mining/Mining)
- Loss: Cross-Entropy; Optimizer: AdamW (lr=1e-3, weight_decay=0.05)
- Augmentierung: horizontale + vertikale Spiegelung
- Training: bis 100 Epochen, Mixed Precision, Batch-Size 4
- Framework: **TerraTorch** (IBM/NASA Geospatial-ML-Bibliothek), orchestriert über **PyTorch Lightning**

### Normalisierungsstatistiken (6-Band-Stack, aus SmallMinesDS berechnet)

| Band | Mittelwert | Std |
|---|---|---|
| Blue (B2) | 1473,81 | 223,44 |
| Green (B3) | 1703,35 | 285,54 |
| Red (B4) | 1696,68 | 413,82 |
| VNIR_5 (B8A) | 3832,40 | 389,61 |
| SWIR_1 (B11) | 3156,11 | 451,50 |
| SWIR_2 (B12) | 2226,07 | 468,27 |

Diese Werte tauchen in **jedem** späteren Trainings-/Inferenzskript wieder auf (Training, Fine-Tuning, Zero-Shot, Voll-Inferenz) — sie sind die "Sprache", in der das Modell Pixelwerte versteht.

---

## 3. Teil B — Eigener Beitrag: Die Pipeline im Überblick

Alles unter `00_Mathias_contribution/` ist die eigene Arbeit. Die Kernfrage: *Funktioniert das auf SmallMinesDS trainierte Modell auch in einer komplett anderen Region (Bono/Bono East, 2025), die es nie gesehen hat?*

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Base-Training auf SmallMinesDS (Kaggle)                             │
│    → models/prithvi-v2-300-*.ckpt                                      │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Neue Region beschaffen: GEE-Export Bono/Bono-East, Januar 2025      │
│    → Bono_Merged_2025.tif (Sentinel-2, 6 Bänder, 10m)                  │
├────────────────────────────────────────────────────────────────────────┤
│ 3. ZERO-SHOT-TEST: Base-Modell direkt auf Bono anwenden (ohne Fine-Tune)│
│    → Ergebnis: Modell "sieht" kein Mining (Domain Shift)                │
├────────────────────────────────────────────────────────────────────────┤
│ 4. Manuelles Labeling: 233 Bono-Kandidatenpatches → 65 final ausgewählt│
│    → QGIS-Polygone (Mining-y-n)                                        │
├────────────────────────────────────────────────────────────────────────┤
│ 5. FINE-TUNING: Base-Checkpoint auf 65 Bono-Labels nachtrainieren       │
│    (partial unfreeze, Kaggle GPU)                                      │
│    → models/prithvi-v2-300-bono-ep13-iou0.7155.ckpt                    │
├────────────────────────────────────────────────────────────────────────┤
│ 6. EVALUATION: Base vs. Fine-Tuned auf Validation-Set vergleichen       │
│    → Mining IoU: 0.00 → 0.49  (dramatische Verbesserung)               │
├────────────────────────────────────────────────────────────────────────┤
│ 7. FLÄCHIGE INFERENZ: Fine-tuned Modell auf gesamte Bono/Bono-East-     │
│    Region anwenden (Kaggle GPU, ~60.000 Patches)                       │
│    → prediction_prob.tif, prediction_binary.tif                        │
├────────────────────────────────────────────────────────────────────────┤
│ 8. SPATIAL LINKAGE: Distanz/Fläche zu Galamsey je Cashew-Farmer         │
│    berechnen und an Mikrodaten anhängen                                │
├────────────────────────────────────────────────────────────────────────┤
│ 9. ÖKONOMETRIE: OLS-Regressionen (mit District-FE, geclusterten SE)     │
│    Exposition → Cashew-Einkommen / Umweltbesorgnis / Nachhaltigkeit    │
└────────────────────────────────────────────────────────────────────────┘
```

Die folgenden Abschnitte gehen jeden dieser 9 Schritte im Detail durch.

---

## 4. Schritt 1: Trainingsdaten vorbereiten & Base-Modell trainieren

### 4.1 Datenvorbereitung — `scripts/01_prepare_dataset.py`

SmallMinesDS liefert `.tif`-Dateien mit **13 Bändern**, das Prithvi-Modell erwartet aber nur **6**. Das Skript:

1. Liest die offiziellen Train/Test-Split-CSVs aus `Hugging_Face_Input/`
2. Extrahiert aus jedem 13-Band-Stack die **6 richtigen** Bänder per Index `[0, 1, 2, 7, 8, 9]` (= B2, B3, B4, B8A, B11, B12)
3. Speichert die Ergebnisse als schlanke 6-Band-GeoTIFFs in `data/GhanaMiningPrithvi/training/` bzw. `validation/`

> **Wichtiger Bugfix (v2):** Eine frühere Version kopierte die 13-Band-Dateien unverändert, sodass TerraTorch versehentlich die *falschen* Bänder (0–5 statt 0,1,2,7,8,9) nahm → SWIR fehlte komplett, was später zu Fehlklassifikationen führte. Diese Version behebt das, indem die Extraktion explizit lokal passiert.

### 4.2 Base-Training auf Kaggle — `Kaggle_Notebook/BA_Thesis_01_Training_SmallMinesDS.ipynb`

Weil das Training GPU-Ressourcen braucht, die lokal nicht verfügbar waren, läuft das Training auf **Kaggle** (kostenlose P100/T4-GPU). Ablauf des Notebooks (5 Zellen):

| Zelle | Inhalt |
|---|---|
| 1 | Pakete installieren (`terratorch==0.99.7`, gepinntes `numpy`, `torchgeo`) |
| 2 | Datensatz automatisch unter `/kaggle/input` finden, 6-Band-Check |
| 3 | Konfiguration: DataModule (Means/Stds s.o.), `SemanticSegmentationTask` mit `freeze_backbone=True` |
| 4 | Training: `AdamW`, lr=1e-3, max. 50 Epochen, `EarlyStopping` (patience=10), Mixed Precision |
| 5 | Evaluation (`trainer.test()`) + Checkpoint-Übersicht |

**Ergebnis:** `models/prithvi-v2-300-epoch=16-val_loss=0.0000.ckpt` (und weitere Checkpoints) — das "Base-Modell", das nur die Original-SmallMinesDS-Region kennt.

**Verifikation ("Model Proof"):** `Model_Proof_Training/generate_proof_images.py` erzeugt 10 Vier-Panel-Bilder (Satellitenbild | Ground Truth | P(Mining) | binäre Vorhersage) auf **Trainingspatches** unterschiedlicher Mining-Anteile (0 % bis ~80 %), um zu zeigen, dass das Modell auf bekannten Daten sauber funktioniert, bevor es auf eine neue Region losgelassen wird.

---

## 5. Schritt 2: Neue Region beschaffen (Google Earth Engine Export)

**Skript:** `GEE_data_Export_Bono_Bono-East_Region.js` (läuft im [Google Earth Engine Code Editor](https://code.earthengine.google.com/), nicht lokal)

Zweck: Ein analoges Sentinel-2-Mosaik für die **Bono / Bono East**-Regionen erzeugen — dieselben Bänder, dieselbe Skalierung wie bei SmallMinesDS, damit das Modell die Daten "wiedererkennt".

| Schritt | Details |
|---|---|
| Gebietsgrenzen | FAO/GAUL Level-1-Grenzen, gefiltert nach `Bono`, `Bono East` **und** dem historischen Namen `Brong Ahafo` (da die Region 2018 administrativ geteilt wurde) |
| Zeitraum | Januar 2025 (Trockenzeit / Harmattan → kaum Wolken, kontrastreicher Boden → Galamsey-Flächen heben sich optisch ab) |
| Quelle | `COPERNICUS/S2_SR_HARMONIZED` (atmosphärisch korrigiert, Surface Reflectance) |
| Wolkenfilter | `CLOUDY_PIXEL_PERCENTAGE < 10` + QA60-Bitmask (Bit 10/11 = Wolken/Zirrus) |
| Kompositierung | **Median** aller wolkenfreien Aufnahmen im Monat → robust gegen Restartefakte |
| Bänder | B2, B3, B4, B8A, B11, B12 (identisch zu SmallMinesDS) |
| Skalierung | `/10000` (GEE-Standard: Reflektanz 0–1 statt 0–10.000 DN) |
| Auflösung/CRS | 10 m, EPSG:32630 (UTM Zone 30N) |

**Output:** `Bono_Merged_2025.tif` (großes Mosaik, ca. 24 GB, liegt lokal unter `data/raw/`, **nicht versioniert**).

> Wichtig für spätere Schritte: Die GEE-Skalierung (`/10000`) muss vor der Inferenz **rückgängig gemacht** werden (`× 10.000`), weil das Modell auf den rohen DN-Werten von SmallMinesDS trainiert wurde.

---

## 6. Schritt 3: Zero-Shot-Test des Base-Modells auf Bono

**Ziel:** Testen, ob das SmallMinesDS-Modell *ohne jegliches Nachtrainieren* (also "zero-shot") auch in Bono Galamsey erkennt.

### 6.1 Testgebiet ausschneiden — `scripts/02_extract_bono_test_patches.py`

- Wählt ein bekanntes Galamsey-Gebiet als 5×5 km-Testfläche (Zentrum: lat 8.054635, lon -2.025502)
- Wandelt WGS84 → UTM Zone 30N um
- Liest das Fenster aus dem großen Mosaik (ohne die komplette 24-GB-Datei zu laden)
- Skaliert Pixelwerte `× 10.000` (GEE-Normalisierung rückgängig machen)
- Padded auf ein Vielfaches von 128 px, zerschneidet in `128×128`-Patches
- Speichert `patch_index.csv` zur späteren Wiederzusammensetzung

### 6.2 Naive Inferenz — `scripts/04_inference_bono.py`

Wendet das Base-Modell direkt (mit den SmallMinesDS-Means/Stds) auf die Bono-Patches an. **Ergebnis:** Das Modell erkennt praktisch **kein** Mining — massive Unterschätzung.

### 6.3 Diagnose: Domain Shift

Der Grund: Bono/Bono East (2025) hat eine **andere radiometrische Verteilung** als das Originaltrainingsgebiet (2016/2022, andere Region). Gemessene Mittelwerte/Std weichen stark ab:

| Band | Train-Mean (SmallMinesDS) | Bono-Mean (2025) |
|---|---|---|
| Blue | 1473,81 | 583,63 |
| Green | 1703,35 | 851,72 |
| Red | 1696,68 | 1241,71 |
| VNIR_5 | 3832,40 | 2411,21 |
| SWIR_1 | 3156,11 | 3027,37 |
| SWIR_2 | 2226,07 | 2290,58 |

Bono-Bilder sind im Schnitt **deutlich dunkler** (vor allem in Blue/Green/Red) — ein klassischer Sensor-/Atmosphären-/Landbedeckungs-Domain-Shift.

### 6.4 Domain-Alignment-Versuch — `scripts/04_inference_bono_2.0.py`

Als Zwischenschritt (vor dem eigentlichen Fine-Tuning) wurde ein rein statistischer Fix ausprobiert: **Z-Score Domain Alignment**.

```
1. z = (Pixel − Bono_Mean) / Bono_Std        # in "neutralen" Statistikraum bringen
2. aligned = z × Train_Std + Train_Mean       # in Trainingsverteilung projizieren
```

Das verbessert die Situation etwas, löst das Problem aber nicht grundsätzlich — die eigentliche Lösung ist **Fine-Tuning mit echten Bono-Labels** (Schritt 5), weil rein statistisches Alignment keine inhaltlichen (spektralen) Unterschiede in der Landbedeckung ausgleichen kann.

**Fazit Zero-Shot-Test:** Direkter Transfer ohne Anpassung funktioniert nicht zuverlässig → Notwendigkeit von Fine-Tuning auf echten, manuell erstellten Bono-Labels.

---

## 7. Schritt 4: Manuelles Labeling der Bono-Region

Da für Bono/Bono East keine offiziellen Ground-Truth-Labels existieren, wurden diese **manuell in QGIS erstellt**.

### 7.1 Rohdaten — `labels_incoming/`

- `20260722_Galamsey Bono - manual labelling.qgz` — QGIS-Projektdatei
- `richtige labels 23..gpkg` (Layer `richtige_labels_v1`, Feld `Mining-y-n`) — finale, händisch digitalisierte Mining-Polygone (visuell anhand des Bono-Mosaiks identifiziert: offene, unbewachsene Erdflächen mit typischer Galamsey-Form — Gruben, Absetzbecken, unregelmäßige Rodungen)

### 7.2 Kandidaten-Patches generieren — `20260723_final_Label_GT_Preview/`

- `generate_unique_patches.py`: Erzeugt **ein PNG pro geografischem 128×128-Fenster**, das mindestens ein Mining-Polygon enthält (Ground Truth zeigt *alle* Mining-Polygone im Fenster, nicht nur eines) — ergibt **39 eindeutige Mining-Patches** (davon 33 mit mehr als einem Polygon) + zusätzlich zufällig gesampelte Non-Mining-Fenster (mind. 200 m von jeder Mine entfernt, ≥95 % valide Pixel)
- `generate_gt_preview.py`: Rendert die Vorschaubilder (Satellitenbild + Polygon-Overlay) zur visuellen Qualitätskontrolle
- `patch_to_labels.csv`: Mapping Patch-ID → Pixel-Koordinaten (`col`, `row`) → Original-Label-IDs
- `README.md`: Tabelle aller 43 unique Mining/Non-Mining-Kandidaten mit Polygonanzahl und Mining-Pixel-Anteil je Patch

### 7.3 Finale Auswahl — `final_selection/` (65 Patches)

Aus den Kandidaten wurden **65 Patches final für das Fine-Tuning ausgewählt**:

| Klasse | Anzahl |
|---|---|
| Mining | 35 |
| Non-Mining | 30 |
| **Total** | **65** |

Dokumentiert in `final_selection/MANIFEST.txt`. Diese 65 Patches sind der **einzige gelabelte Datensatz für Bono** und damit die Grundlage des gesamten Fine-Tunings.

---

## 8. Schritt 5: Fine-Tuning auf Bono-Labels

### 8.1 Export als Trainingsdaten — `scripts/export_bono_finetune_dataset.py`

Wandelt die 65 finalen Auswahl-Patches in das gleiche Format wie SmallMinesDS um:

- **IMG:** 6 Bänder, Float32, DN-Skala (Reflektanz × 10.000) — direkt aus dem Bono-Mosaik ausgeschnitten
- **MASK:** 1 Band, UINT8 (0/1) — **alle** Mining-Polygone rasterisiert, die das 128×128-Fenster schneiden (nicht nur das ursprünglich auswählende Polygon)
- **Split:** 80/20, **stratifiziert nach Mining/Non-Mining** (Seed=23), damit beide Splits eine ähnliche Klassenbalance haben
- Für zufällig generierte Non-Mining-Fenster (IDs ≥122) wird der ursprüngliche Zufallssampler (Seed=23, Mindestabstand 200 m) exakt reproduziert, um konsistente Koordinaten zu garantieren

**Output:** `data/GhanaMiningPrithvi_bono/{training,validation}/BONO_XXXX_{IMG,MASK}.tif` + `split_manifest.csv`

### 8.2 Fine-Tuning auf Kaggle — `Kaggle_Notebook/BA_Thesis_02_Finetuning_Bono.ipynb`

Mit nur 65 Patches besteht ein hohes Overfitting-Risiko. Die Strategie dagegen: **Partial Unfreezing** (nur ein kleiner Teil des Netzes wird angepasst):

| Komponente | Status |
|---|---|
| Encoder-Blöcke 0–19 (von 24) | **eingefroren** |
| Encoder-Blöcke 20–23 (letzte 4) + finales LayerNorm | **trainierbar** |
| UperNet-Decoder + Segmentation-Head | **trainierbar** |
| Lernrate | 5×10⁻⁴ (niedriger als beim reinen Decoder-Training) |
| Class Weights | `[0.2, 0.8]` (Non-Mining/Mining) — kompensiert, dass Mining-Pixel in den Masken die Minderheit sind |
| Monitor für Checkpointing | `val/Multiclass_Jaccard_Index` (Gesamt-IoU), zusätzlich **F1** und klassenweise IoU/F1 geloggt |
| Early Stopping | Patience = 8 Epochen |
| Max. Epochen | 40 |

**Warum nur die letzten 4 von 24 Blöcken?** Die frühen Encoder-Blöcke lernen generische, übertragbare Bildmerkmale (Kanten, Texturen), die letzten Blöcke sind spezifischer für die ursprüngliche Bildverteilung. Nur die spätesten Schichten + der Decoder werden an die neue Region angepasst, um mit nur 65 Beispielen nicht das gesamte Modell zu "verwirren" (Overfitting-Schutz).

### 8.3 Trainingsverlauf (`reports/02_Finetuning_Bono/tables/metrics.csv`)

Beste Validation-Ergebnisse bei **Epoche 13** (danach kein weiterer Fortschritt → Early Stopping):

| Metrik | Wert |
|---|---|
| Val Overall IoU (`Multiclass_Jaccard_Index`) | **0,7155** |
| Val Overall F1 | 0,9477 |
| Val Mining IoU | 0,4861 |
| Val Mining Accuracy (Recall-artig) | 0,6685 |
| Val Non-Mining IoU | 0,9449 |

**Finaler Checkpoint:** `models/prithvi-v2-300-bono-ep13-iou0.7155.ckpt`

---

## 9. Schritt 6: Evaluation — Base vs. Fine-Tuned

**Notebook:** `Notebooks/03_Inference_and_Evaluation_Comparison.ipynb`
**Ergebnisse:** `reports/03_Inference_and_Evaluation_Comparison/`

Direkter Vergleich beider Modelle auf demselben Bono-Validation-Set (13 Patches):

| Metrik | Base-Modell (Zero-Shot) | Fine-Tuned | Δ absolut | Δ relativ |
|---|---|---|---|---|
| **Mining IoU** | 0,000 | **0,486** | +0,486 | — (von 0 ausgehend) |
| Mining Accuracy | 0,000 | 0,668 | +0,668 | — |
| Overall IoU | 0,463 | 0,716 | +0,253 | **+54,6 %** |
| Overall F1 | 0,926 | 0,948 | +0,022 | +2,3 % |
| Test Loss | 0,800 | 0,203 | −0,597 | −74,6 % |

**Interpretation:** Das Base-Modell erkennt in der neuen Region **buchstäblich keine einzige Mining-Fläche** (IoU = 0) — reiner Domain-Shift-Effekt (siehe Schritt 3). Nach dem Fine-Tuning mit nur 65 Labels erreicht das Modell eine **Mining-IoU von 0,49**, was für ASGM-Kartierung (notorisch schwierige, kleinflächige, heterogene Klasse) ein solides Ergebnis ist.

Visualisierung: `figures/bono_visual_comparison.png` (4-Panel: RGB | Ground Truth | Base-Vorhersage | Fine-Tuned-Vorhersage).

---

## 10. Schritt 7: Flächige Inferenz über ganz Bono/Bono East

**Ziel:** Das fine-getunte Modell nicht nur auf Testpatches, sondern auf das **komplette Bono/Bono-East-Mosaik** anwenden, um eine vollständige Galamsey-Karte zu erzeugen.

### 10.1 Kaggle-Vollinferenz — `Kaggle_Notebook/BA_Thesis_03_Full_Bono_Inference.ipynb`

- Läuft blockweise über das gesamte ~24-GB-Mosaik (Sliding-Window, 128×128-Patches, kein Laden der ganzen Datei in den RAM)
- Für jeden Patch: Normalisierung mit SmallMinesDS-Means/Stds, Modellvorhersage, Softmax → `P(Mining)`
- Schreibt zwei georeferenzierte GeoTIFFs blockweise direkt auf Platte:
  - `prediction_prob.tif` (Float32, P(Mining) ∈ [0,1])
  - `prediction_binary.tif` (UINT8, Schwelle 0,5)
- **Ergebnis (voller Lauf):** 60.270 Patches verarbeitet, **Mining-Anteil ≈ 0,28 %** der Gesamtfläche (`reports/05_Full_Bono_Inference/tables/inference_stats_full.txt`)

Lokales Pendant/Smoke-Test: `scripts/05_inference_bono_full.py` (gleiche Logik, mit `LIMIT_PATCHES` für kleine Testläufe ohne GPU).

### 10.2 Übersichtskarten — `scripts/06_ghana_map_galamsey_bono.py` & `scripts/07_regional_maps_bono.py`

- **Skript 06:** Ganz-Ghana-Karte mit Verwaltungsgrenzen (GADM) und der Bono-Vorhersage als rote Überlagerung → `data/ghana_map_galamsey_bono.png`
- **Skript 07:** Detailliertere regionale Karten (verschiedene Konfidenzschwellen 50 %/75 %, kombinierte und getrennte Bono/Bono-East-Ansichten) → `reports/05_Full_Bono_Inference/figures/regional_map_bono_*.png`

### 10.3 Post-Processing für die Ökonometrie (wichtig!)

Für die spätere Verknüpfung mit Mikrodaten wurde aus `prediction_prob.tif` eine **konservativere** binäre Mining-Maske erzeugt (im Spatial-Linkage-Notebook, siehe Schritt 8):

- Schwellenwert **P(Mining) ≥ 90 %** (statt 50 %) — reduziert False Positives
- **Sieve-Filter** (< 10 Pixel) — entfernt einzelne Rausch-Pixel ohne räumlichen Zusammenhang
- Ergebnis: `prediction_mining_conf90_sieve.tif` (liegt im Mikrodaten-Analyseordner, da nur dort verwendet)

---

## 11. Schritt 8: Verknüpfung mit Mikrodaten (Spatial Linkage)

**Ordner:** `01_Microdata/20260724_Analysis/` (kompletter Ordner `01_Microdata/` ist **gitignored** — enthält personenbezogene Farmer-Daten)
**Notebook:** `04_Spatial_Linkage.ipynb`

### 11.1 Ausgangsdaten

- **Mikrodaten:** Haushaltsbefragung von **411 Cashew-Farmern** in Bono/Bono East (Carbon-Farming-Projekt-Survey), ~290 Spalten (soziodemografische Angaben, Farmcharakteristika, Einkommen, Umweltwahrnehmung, Nachhaltigkeitsverhalten)
- **Räumliche Daten:** GPS-Koordinaten je Farmer + die Mining-Wahrscheinlichkeitskarte aus Schritt 7

### 11.2 Erzeugte Mining-Maske

`P(Mining) ≥ 90 %` aus `prediction_prob.tif`, gefiltert mit Sieve (< 10 zusammenhängende Pixel entfernt) → binäre Rastermaske bei 10 m Auflösung, gespeichert als `prediction_mining_conf90_sieve.tif`.

### 11.3 Neue Exposure-Variablen (an die Mikrodaten angehängt)

Für jeden der 411 Farmer wurden vier räumliche Kennzahlen berechnet (mittels `scipy.spatial.cKDTree` für Nearest-Neighbor-Distanzen und Pufferzonen-Flächenberechnung):

| Variable | Bedeutung | Verwendung |
|---|---|---|
| `dist_to_galamsey_km` | Distanz zum **nächsten einzelnen Mining-Pixel** (km) | **Baseline**-Distanzmaß |
| `dist_to_galamsey_min50px_km` | Distanz zur nächsten **zusammenhängenden** Mining-Fläche mit **≥ 50 Pixeln** (= 0,5 ha) | **Robustheitscheck**: schließt einzelne Rauschpixel als "nächste Mine" aus |
| `galamsey_area_5km_ha` | Summe der Mining-Fläche in einem **5-km-Radius** um den Farmer (ha) | Lokale Flächen-Exposition |
| `galamsey_area_20km_ha` | Summe der Mining-Fläche in einem **20-km-Radius** (ha) | Regionale (breitere) Flächen-Exposition |

Ablauf im Notebook: (1) CRS-Transformation der Farmer-Koordinaten in das Raster-CRS (UTM), (2) Erzeugung der 90 %+Sieve-Maske, (3) Distanzberechnung via KD-Tree auf Mining-Pixel-Zentroiden, (4) Flächenberechnung über Pufferzonen-Masken-Intersektion.

**Output:** `Data_CarbonFarming_Linked.csv` / `.xlsx` — die ursprünglichen 411×~290 Mikrodaten **plus die 4 neuen Exposure-Spalten** (→ 294 Spalten).

> Hinweis: Es gibt **keine qualitative/Freitext-Erwähnung** von Gold/Galamsey in der Befragung selbst — die Exposition kommt ausschließlich aus der räumlichen Verknüpfung mit der Fernerkundungskarte.

---

## 12. Schritt 9: Ökonometrische Analyse

**Ordner:** `01_Microdata/20260724_Analysis/` — drei eigenständige Python-Skripte (keine Notebooks), die alle demselben Analyse-Design folgen, aber unterschiedliche Exposure-Definitionen als Treatment nutzen:

| Skript | Treatment-Variable | Output |
|---|---|---|
| `05_Econometric_Analysis.py` | `log(dist_to_galamsey_km)` | `reports/06_Econometric_Analysis/` |
| `06_Econometric_Analysis_min50px.py` | `log(dist_to_galamsey_min50px_km)` | `reports/07_Econometric_Analysis_min50px/` |
| `07_Econometric_Analysis_area.py` | `galamsey_area_5km_ha` **und** `galamsey_area_20km_ha` | `reports/08_.../` bzw. `reports/09_.../` |

### 12.1 Modell — was wird geschätzt?

Für jede der drei Outcome-Variablen wird eine **lineare Regression (OLS)** geschätzt:

```
Y_i = β₀ + β₁·Exposition_i + β·Controls_i + (Distrikt-FE) + ε_i
```

- **Y (abhängige Variablen, 3 Modelle je Skript):**
  1. `perc_cashew_income` — Anteil des Cashew-Einkommens am Gesamteinkommen
  2. `worry_env` — selbstberichtete Umweltbesorgnis (Skala 0–10)
  3. `sust_participation` — Teilnahme an Nachhaltigkeits-/Carbon-Aktivitäten (0/1)
  (Nicht verwendet: `carbon_interest` — fast keine Varianz in der Stichprobe)
- **Exposition (Treatment):** je nach Skript eine der 4 oben beschriebenen Variablen
- **Controls:** Alter, Geschlecht, Bildung, `num_plots` (Anzahl Plots als Proxy für Betriebsgröße — `farm_size` fehlt bei ~80 % der Befragten), Landbesitz, landwirtschaftliche Erfahrung
- **Zwei Spezifikationen je Modell:**
  - **Spec A:** mit **District Fixed Effects** (Distrikt-Dummies) — Vergleich *innerhalb* desselben Distrikts
  - **Spec B:** ohne Fixed Effects — nutzt auch Unterschiede *zwischen* Communities/Distrikten
- **Standardfehler:** Community-geclustert (9 Communities) — erlaubt Korrelation der Fehler innerhalb eines Dorfes

> **Wichtige Einschränkung:** Community und Distrikt sind in dieser Stichprobe **1:1 verschachtelt** (jede Community liegt in genau einem Distrikt). Spec A mit Distrikt-FE ist daher eine reine Within-Schätzung, Spec B nutzt zusätzlich Between-Variation. Mit nur 9 Clustern sind alle Ergebnisse mit Vorsicht zu interpretieren (Cluster-Robust-SE-Theorie verlangt eigentlich mehr Cluster).

### 12.2 Kernergebnis (Flächen-Exposition, 5 km, Spec A mit Distrikt-FE)

| Outcome | Koeffizient | Signifikanz | Interpretation |
|---|---|---|---|
| Cashew-Einkommensanteil | **+0,019** | *** (p<0,01) | Mehr Galamsey-Fläche im 5-km-Umkreis ist mit einem **höheren** Cashew-Einkommensanteil assoziiert |
| Umweltbesorgnis | +0,011 | n.s. | Kein signifikanter Zusammenhang |
| Nachhaltigkeits-Teilnahme | **−0,014** | ** (p<0,05) | Mehr Galamsey-Fläche ist mit **geringerer** Teilnahme an Nachhaltigkeitsprogrammen assoziiert |

(Die anderen Treatment-Definitionen — Distanz, ≥50-Pixel-Distanz, 20-km-Fläche — liefern die vollständigen Robustheitschecks; Details in den jeweiligen `reports/0X_.../tables/regression_results_summary.csv`.)

### 12.3 Publikationsreife Outputs

Jedes Skript exportiert automatisch:

- `regression_results_summary.csv` / `.tex` — kompakte Koeffiziententabelle (alle 6 Modell-Spec-Kombinationen)
- `regression_coefficients_full.csv` — vollständige Regressionsoutputs inkl. aller Kontrollvariablen und Distrikt-Dummies
- `regression_table_publication.csv` / `.md` / `.tex` — AER-Stil-Tabelle (Koeffizient + Standardfehler in Klammern, Sternchen-Signifikanz)
- `figures/regression_coefficients.png` — Koeffizientenplot mit Konfidenzintervallen

Zusätzlich existiert eine **konsolidierte Word-Datei** `reports/tables/all_regressions.docx` mit allen 8 Haupttabellen (Flächen 5/20 km × Distanz baseline/≥50px, je mit/ohne FE) **plus** unter jeder Tabelle einer ausführlichen Erklärung in Alltagssprache: was Fixed Effects, Controls, N, R² und die Sternchen bedeuten, und eine konkrete Interpretation jedes Koeffizienten ("Variable X hat einen signifikant positiven/negativen Einfluss auf Y").

---

## 13. Vollständige Ordnerstruktur

```
BA_Thesis_SmallMinesDS/
│
├── README.md                          ← Original SmallMinesDS-Paper-README (Zitation, Setup)
├── requirements.txt                   ← Python-Abhängigkeiten (Conda-Env "terratorch")
├── install_sam2.sh                    ← Setup-Skript für SAM2-Environment
├── LICENSE
├── figs/                              ← Abbildungen aus dem Paper (Study Area, Bänder, etc.)
├── scripts/                           ← Original-Trainingsskripte (Prithvi 300M/600M, ResNet50, SAM2)
│
├── Hugging_Face_Input/                ← SmallMinesDS-Rohdaten (13-Band-Patches, Train/Test-Splits)
├── data/
│   ├── GhanaMiningPrithvi/            ← 6-Band-Trainingsdaten (Original-Region, aus Schritt 1)
│   ├── GhanaMiningPrithvi_bono/       ← 6-Band-Fine-Tuning-Daten (65 Bono-Patches, aus Schritt 5)
│   ├── raw/                           ← Bono_Merged_2025.tif (großes GEE-Mosaik, gitignored)
│   ├── patches_bono_test/             ← 5×5km-Testpatches für Zero-Shot (Schritt 3)
│   ├── inference_bono_full_ft/        ← Vollflächige Vorhersage-GeoTIFFs (Schritt 7)
│   └── cache/                         ← Zwischencache (Flüsse etc. für Kartenskripte)
├── models/                            ← Alle Checkpoints (Base + Fine-Tuned), gitignored
│
└── 00_Mathias_contribution/           ← EIGENER BEITRAG (dieses Dokument)
    ├── README.md                      ← DIESE DATEI
    ├── REPO_OVERVIEW.md               ← Ältere, technischere Übersicht (Fokus: Basis-Trainingsskripte)
    ├── GEE_data_Export_Bono_Bono-East_Region.js   ← Schritt 2
    │
    ├── scripts/                       ← Eigene Pipeline-Skripte (Schritte 1–3, 5, 7, 8)
    │   ├── 01_prepare_dataset.py
    │   ├── 02_extract_bono_test_patches.py
    │   ├── 04_inference_bono.py
    │   ├── 04_inference_bono_2.0.py    (Domain-Alignment-Experiment)
    │   ├── 05_inference_bono_full.py
    │   ├── 06_ghana_map_galamsey_bono.py
    │   ├── 07_regional_maps_bono.py
    │   ├── export_bono_finetune_dataset.py
    │   └── ... (Plot-/Hilfsskripte)
    │
    ├── labels_incoming/                ← QGIS-Rohlabels (Schritt 4)
    ├── 20260723_final_Label_GT_Preview/← Label-Kandidaten & finale 65er-Auswahl (Schritt 4)
    │
    ├── Kaggle_Notebook/                ← Alle GPU-Trainingsnotebooks (Schritte 1, 5, 7)
    │   ├── BA_Thesis_01_Training_SmallMinesDS.ipynb
    │   ├── BA_Thesis_02_Finetuning_Bono.ipynb
    │   └── BA_Thesis_03_Full_Bono_Inference.ipynb
    ├── Notebooks/                      ← Evaluationsnotebook (Schritt 6)
    │   └── 03_Inference_and_Evaluation_Comparison.ipynb
    ├── Model_Proof_Training/           ← Sanity-Check auf Original-Trainingsdaten
    │
    ├── reports/                        ← Alle generierten Tabellen/Figuren, nummeriert nach Analyseschritt
    │   ├── 02_Finetuning_Bono/
    │   ├── 03_Inference_and_Evaluation_Comparison/
    │   ├── 05_Full_Bono_Inference/
    │   ├── 06_Econometric_Analysis/            (Distanz, baseline)
    │   ├── 07_Econometric_Analysis_min50px/    (Distanz, ≥50px)
    │   ├── 08_Econometric_Analysis_area5km/    (Fläche, 5km)
    │   ├── 09_Econometric_Analysis_area20km/   (Fläche, 20km)
    │   └── tables/all_regressions.docx         (konsolidierte Wortdatei)
    │
    ├── 01_Microdata/                   ← GITIGNORED (sensible Farmer-Mikrodaten)
    │   └── 20260724_Analysis/
    │       ├── 04_Spatial_Linkage.ipynb        (Schritt 8)
    │       ├── 05_Econometric_Analysis.py      (Schritt 9)
    │       ├── 06_Econometric_Analysis_min50px.py
    │       ├── 07_Econometric_Analysis_area.py
    │       ├── Data_CarbonFarming_Linked.csv/.xlsx
    │       └── prediction_mining_conf90_sieve.tif
    │
    └── 99_old/                         ← Archiv früherer Colab-Notebooks & Tagesnotizen
```

---

## 14. Technische Umgebungen (Conda-Environments)

Für den eigenen Beitrag werden **zwei Umgebungen** genutzt (zusätzlich zu den beiden im Paper-README beschriebenen `terratorch`/`sam2`-Envs, die für das Original-Training gebraucht werden):

| Environment | Zweck | Wo genutzt |
|---|---|---|
| `terratorch` (lokal, `requirements.txt`) | GeoTIFF-Verarbeitung, Modell-Inferenz lokal, Kartenskripte | `scripts/01_...` bis `07_...` (außer Training) |
| **Kaggle-Kernel** (`terratorch==0.99.7`, gepinntes `numpy`, `torchgeo>=0.6,<0.7`) | GPU-Training & -Inferenz (Prithvi ist rechenintensiv) | Alle 3 Kaggle-Notebooks (Training, Fine-Tuning, Voll-Inferenz) |
| `smallmines` (lokal, Conda) | Ökonometrie/Statistik (`pandas`, `statsmodels`, `python-docx`, `geopandas`, `scipy`) | `01_Microdata/20260724_Analysis/*.py`, `04_Spatial_Linkage.ipynb` |

**Warum Kaggle statt lokal für Training/Inferenz?** Prithvi-EO v2 300M ist ein Vision Transformer mit ~300M Parametern; Training und Voll-Inferenz über ein 24-GB-Mosaik erfordern eine GPU mit ausreichend VRAM — lokal nicht verfügbar, daher Kaggle-P100/T4-Kernel mit hochgeladenen Datasets/Checkpoints.

---

## 15. Große / sensible Dateien (nicht versioniert)

Aus `.gitignore` (Auszug, projektspezifische Regeln):

```gitignore
data/raw/                          # Bono_Merged_2025.tif (~24 GB)
data/patches_bono_test/
data/inference_bono_full/
data/GhanaMiningPrithvi/
data/cache/
models/                            # alle .ckpt-Checkpoints
*.ckpt / *.pt / *.pth / *.torch
GhanaMiningPrithvi.zip
01_Microdata/                      # komplette Mikrodaten (personenbezogen!)
```

**Grund für `01_Microdata/`:** Der Ordner enthält personenbezogene Umfragedaten von 411 Cashew-Farmern (Namen indirekt über GPS-Koordinaten rekonstruierbar, Einkommensangaben etc.) und wird daher bewusst **nicht** ins Repository committed, auch nicht in Kopien innerhalb von `00_Mathias_contribution/`.

---

## 16. Reproduktion Schritt für Schritt (Kurzfassung)

```bash
# 0. Environments einrichten (siehe Abschnitt 14 / Paper-README)
conda create -n terratorch python=3.11 && conda activate terratorch
pip install -r requirements.txt

# 1. SmallMinesDS-Rohdaten von HuggingFace laden → Hugging_Face_Input/
# 2. 6-Band-Trainingsdaten vorbereiten
python 00_Mathias_contribution/scripts/01_prepare_dataset.py

# 3. Base-Training (auf Kaggle, GPU nötig)
#    → Kaggle_Notebook/BA_Thesis_01_Training_SmallMinesDS.ipynb hochladen & ausführen
#    → Checkpoint lokal speichern: models/prithvi-v2-300-base.ckpt

# 4. Neues Gebiet exportieren (in Google Earth Engine Code Editor)
#    → GEE_data_Export_Bono_Bono-East_Region.js ausführen → Bono_Merged_2025.tif
#      lokal ablegen unter data/raw/

# 5. Zero-Shot-Test
python 00_Mathias_contribution/scripts/02_extract_bono_test_patches.py
python 00_Mathias_contribution/scripts/04_inference_bono.py   # zeigt: kein Mining erkannt

# 6. Manuelles Labeling (QGIS) → labels_incoming/, dann Kandidaten/Auswahl generieren
python 00_Mathias_contribution/20260723_final_Label_GT_Preview/generate_unique_patches.py
python 00_Mathias_contribution/20260723_final_Label_GT_Preview/generate_gt_preview.py

# 7. Fine-Tuning-Datensatz exportieren
python 00_Mathias_contribution/scripts/export_bono_finetune_dataset.py

# 8. Fine-Tuning (auf Kaggle, GPU nötig)
#    → Kaggle_Notebook/BA_Thesis_02_Finetuning_Bono.ipynb
#    → Checkpoint lokal speichern: models/prithvi-v2-300-bono-ep13-iou0.7155.ckpt

# 9. Evaluation Base vs. Fine-Tuned
#    → Notebooks/03_Inference_and_Evaluation_Comparison.ipynb ausführen

# 10. Flächige Inferenz (auf Kaggle, GPU nötig)
#     → Kaggle_Notebook/BA_Thesis_03_Full_Bono_Inference.ipynb
#     → Ergebnisse lokal nach data/inference_bono_full_ft/ kopieren
python 00_Mathias_contribution/scripts/06_ghana_map_galamsey_bono.py
python 00_Mathias_contribution/scripts/07_regional_maps_bono.py

# 11. Spatial Linkage (Mikrodaten benötigt, nicht im Repo enthalten)
conda activate smallmines
jupyter notebook 00_Mathias_contribution/01_Microdata/20260724_Analysis/04_Spatial_Linkage.ipynb

# 12. Ökonometrische Analyse
python 00_Mathias_contribution/01_Microdata/20260724_Analysis/05_Econometric_Analysis.py
python 00_Mathias_contribution/01_Microdata/20260724_Analysis/06_Econometric_Analysis_min50px.py
python 00_Mathias_contribution/01_Microdata/20260724_Analysis/07_Econometric_Analysis_area.py
```

---

## Referenzen

```bibtex
@ARTICLE{10982207,
  author={Ofori-Ampofo, Stella and Zappacosta, Antony and Kuzu, Rıdvan Salih and Schauer, Peter and Willberg, Martin and Zhu, Xiao Xiang},
  journal={IEEE Geoscience and Remote Sensing Letters},
  title={SmallMinesDS: A Multi-Modal Dataset for Mapping Artisanal and Small-Scale Gold Mines},
  year={2025},
  doi={10.1109/LGRS.2025.3566356}
}
```

- SmallMinesDS-Datensatz: <https://huggingface.co/datasets/ellaampy/SmallMinesDS>
- TerraTorch: <https://github.com/IBM/terratorch>
- Prithvi-EO v2: NASA/IBM Geospatial Foundation Model
