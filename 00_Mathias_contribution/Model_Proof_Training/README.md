# Model Proof – SmallMinesDS Training Data

Dieser Ordner dient als **klarer Proof Point**, dass das trainierte Prithvi-EO v2 Modell auf den **Original-Trainingsdaten (SmallMinesDS)** funktioniert.

## Inhalt

| Datei/Ordner | Beschreibung |
|---|---|
| `generate_proof_images.py` | Skript zur Erzeugung der Proof-Bilder |
| `patches/` | 10 PNG-Dateien, je ein Trainingspatch |

## Layout pro Bild (4 Panels)

Jede PNG zeigt **einen** SmallMinesDS-Trainingspatch (128×128 px, 10 m Auflösung):

| Panel | Inhalt |
|---|---|
| 1 | Satellitenbild (True Color: B4/B3/B2) |
| 2 | Ground Truth (Label-Maske) |
| 3 | Modell-Ausgabe P(Mining) |
| 4 | Binäre Vorhersage (Threshold 0.5) |

## Ausgewählte Patches

10 diverse Beispiele aus dem Trainingsset:

- **3× Non-Mining** (0 %): `GH_0001_2016`, `GH_0002_2016`, `GH_0004_2016`
- **2× wenig Mining** (0.1–5 %): `GH_0354_2022`, `GH_1173_2022`
- **2× mittlerer Anteil** (~30 %): `GH_1952_2016`, `GH_0080_2022`
- **3× viel Mining** (~70–80 %): `GH_0079_2016`, `GH_0122_2022`, `GH_0865_2016`

## Ausführung

```bash
conda activate smallmines   # oder terratorch
python 00_Mathias_contribution/Model_Proof_Training/generate_proof_images.py
```

**Voraussetzungen:**
- `data/GhanaMiningPrithvi/training/` (6-Band-Patches nach Band-Fix)
- Checkpoint in `00_Mathias_contribution/Kaggle_Notebook/` oder `models/`

## Interpretation

- **Non-Mining-Patches:** Panel 3 sollte überwiegend weiß (niedrige P(Mining)) sein, Panel 4 fast komplett weiß.
- **Mining-Patches:** Panel 3 sollte rote Bereiche zeigen, Panel 4 sollte GT (Panel 2) annähernd widerspiegeln.
- **IoU** in der Überschrift: Übereinstimmung zwischen GT und binärer Vorhersage pro Patch.
