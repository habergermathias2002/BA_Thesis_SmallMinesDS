# Daily Progress – Tuesday, March 17, 2026

**Ziel:** Training auf Kaggle mit korrekten 6-Band-Daten durchführen, Checkpoints sichern, Proof-Bilder erzeugen und Repo-Struktur (.gitignore, Ordner) für Code/Bilder vs. Daten/Checkpoints klären.

---

## Überblick: Schritte seit letzter Doku (14.03.2026)

| Schritt | Status | Kurzbeschreibung |
|--------|--------|------------------|
| 1. Training auf Kaggle statt Colab | ✅ | Notebook `BA_Thesis_01_Training_SmallMinesDS.ipynb` in Kaggle ausgeführt |
| 2. Kaggle-Umgebungsprobleme beheben | ✅ | numpy-Pinning, torchgeo-Version, ein einziger pip-Aufruf |
| 3. Training erfolgreich | ✅ | 2983 Train / 1287 Val, 6 Bänder, P100, ~97.98 % Accuracy |
| 4. Checkpoints von Kaggle laden | ✅ | epoch=08, epoch=10, last.ckpt (lokal in `Kaggle_Notebook/`) |
| 5. Proof-Bilder mit Kaggle-Checkpoints | ✅ | Skript `plot_training_proof_kaggle_ckpt.py`, 2 PNGs (kaggle_ckpt, last_ckpt) |
| 6. Git: Nur Daten & Checkpoints ignorieren | ✅ | Proof-Bilder, Skripte, MDs werden getrackt; *.ckpt, *.zip, große Daten ignoriert |
| 7. Kaggle_Notebook in 00_Mathias_contribution | ✅ | Alles unter `00_Mathias_contribution/Kaggle_Notebook/`; Commit & Push |

---

## 1. Training auf Kaggle

- **Notebook:** `00_Mathias_contribution/Kaggle_Notebook/BA_Thesis_01_Training_SmallMinesDS.ipynb`
- **Daten:** Dataset `ghanaminingprithvi-ashanti-smds-dataset` (6-Band SmallMinesDS nach Band-Fix) als Input hinzugefügt.
- **Ablauf:** Zelle 1 (Pakete), Zelle 2 (Imports & Pfade), Zelle 3 (Konfiguration), Zelle 4 (Training), Zelle 5 (Evaluation).
- **Hardware:** GPU P100 auf Kaggle.

---

## 2. Kaggle-Umgebungsprobleme und Fixes

### Problem 1: numpy – „dtype size changed“ / „cannot load module more than once“

- **Ursache:** Transitive Dependencies (z. B. über `segmentation-models-pytorch`/`timm`) haben numpy auf 1.26.4 gedowngradet; Kaggle-C-Extensions (rasterio, numpy) sind für numpy 2.x gebaut → binäre Inkompatibilität.
- **Fix:** Vor dem Install die **exakte** Kaggle-numpy-Version ermitteln und in **einem** pip-Aufruf mit terratorch installieren:  
  `numpy==<exakte Version>` (z. B. 2.0.2), sodass pip numpy gar nicht ändert.

### Problem 2: torchgeo – „AugmentationSequential“ nicht gefunden

- **Ursache:** Kaggle hat eine neuere torchgeo-Version (0.7+) vorinstalliert; darin wurde `AugmentationSequential` entfernt. `terratorch==0.99.7` erwartet torchgeo 0.6.x.
- **Fix:** Explizit pinnen:  
  `"torchgeo>=0.6.0,<0.7.0"` im gleichen pip-Install wie terratorch.

### Problem 3: Zwei pip-Aufrufe → zwei numpy-Installationen

- **Ursache:** Erst `pip install terratorch`, dann `pip install numpy>=2.0` → zwei numpy-Installationen im Pfad, rasterio-C-Extension crasht mit „cannot load module more than once“.
- **Fix:** Alles in **einem** Aufruf:  
  `pip install terratorch==0.99.7 "torchgeo>=0.6.0,<0.7.0" numpy==<exakte Version>`.

### Pfad zum Dataset auf Kaggle

- Dataset-Pfad kann je nach Mount variieren. Im Notebook wird automatisch nach dem Ordner `GhanaMiningPrithvi` unter `/kaggle/input` gesucht (`find`), Fallback auf einen festen Pfad mit Verzeichnisliste bei Fehler.

---

## 3. Trainingsergebnis (Kaggle-Run)

- **Training:** 2983 Patches, **Validation:** 1287 Patches, **6 Bänder** bestätigt.
- **Metriken (Validation/Test):**
  - Multiclass Accuracy: **97.98 %**
  - Mining-Accuracy: **85.52 %**, Non-Mining: **98.87 %**
  - Mining IoU (Jaccard): **73.86 %**
  - Loss: ~0.055
- **Checkpoints:** Bester laut Log: `epoch=13`; gespeichert u. a. `last.ckpt` sowie epoch-basierte Dateien (z. B. epoch=08, 10). Dateinamen enthalten teils `epochepoch=` / `valval_loss=` (Formatierungsfehler im Template).

---

## 4. Checkpoints lokal

- Checkpoints von Kaggle (Output-Tab bzw. ZIP) heruntergeladen und im Projekt abgelegt.
- Relevante Dateien (lokal, **nicht** in Git):  
  `00_Mathias_contribution/Kaggle_Notebook/*.ckpt` sowie ggf. `models/*.ckpt`.
- `last.ckpt` entspricht dem letzten Trainingsstand (z. B. epoch 17).

---

## 5. Proof-Bilder mit Kaggle-Checkpoints

### Skript

- **Datei:** `00_Mathias_contribution/Kaggle_Notebook/plot_training_proof_kaggle_ckpt.py`
- **Funktion:** Lädt automatisch die erste `.ckpt`-Datei im Ordner `Kaggle_Notebook`, wendet das Modell auf 6 feste SmallMinesDS-Trainings-Patches an (3 Mining, 3 Non-Mining) und erzeugt ein 2-spaltiges Bild: links True-Color (B4/B3/B2), rechts Mining-Wahrscheinlichkeit.
- **Patches:** `GH_0122_2022`, `GH_0079_2016`, `GH_0105_2022` (Mining), `GH_0001_2016`, `GH_0002_2016`, `GH_0004_2016` (Non-Mining). (Hinweis: `GH_0003` existiert nicht im Datensatz, daher `GH_0004`.)
- **Ausführung (lokal):** Conda-Env `smallmines` mit PyTorch/TerraTorch verwenden; z. B.  
  `python 00_Mathias_contribution/Kaggle_Notebook/plot_training_proof_kaggle_ckpt.py`  
  von Repo-Root, oder nach `conda activate smallmines`.

### Erzeugte Bilder

- **`model_proof_on_training_patches_kaggle_ckpt.png`** – erzeugt mit der alphabetisch ersten .ckpt (z. B. epoch=08).
- **`model_proof_on_training_patches_last_ckpt.png`** – erzeugt mit `last.ckpt` (separater Aufruf / Inline-Skript).
- Beide liegen in `00_Mathias_contribution/Kaggle_Notebook/` und werden **in Git getrackt**.

---

## 6. Git: Nur Daten und Checkpoints ignorieren

### Gewünschte Regel

- **Versioniert:** Alle Skripte, Code, Markdown-Dateien und **Bilder** (inkl. Proof-PNGs).
- **Nicht versioniert:** Große Daten, Checkpoints, ZIP-Archive (z. B. Dataset-Pakete).

### Anpassungen in `.gitignore`

- **Entfernt:** `Kaggle_Notebook/*.png` – Proof-Bilder sollen gepusht werden.
- **Beibehalten/ergänzt:**
  - `*/Kaggle_Notebook/*.ckpt` – Checkpoints in jedem Kaggle_Notebook-Ordner ignorieren.
  - `*/Kaggle_Notebook/*.zip` – ZIP-Artefakte in Kaggle_Notebook ignorieren.
  - `GhanaMiningPrithvi.zip` – lokales Dataset-Archiv (groß) nicht pushen.
- Bestehende Einträge für `models/`, `*.ckpt`, `*.pt`, `data/raw/`, `data/GhanaMiningPrithvi/`, `01_Microdata/` usw. unverändert.

---

## 7. Ordnerstruktur und Commit

- **Verschiebung:** Der komplette Ordner `Kaggle_Notebook/` wurde von Repo-Root nach  
  `00_Mathias_contribution/Kaggle_Notebook/` verschoben.
- **Inhalt (alle getrackt):**
  - `BA_Thesis_01_Training_SmallMinesDS.ipynb`
  - `Next_Steps_Inference_Finetuning.md`
  - `README.md`
  - `plot_training_proof_kaggle_ckpt.py`
  - `model_proof_on_training_patches_kaggle_ckpt.png`
  - `model_proof_on_training_patches_last_ckpt.png`
- **Nicht getrackt (lokal vorhanden, ignoriert):**  
  `*.ckpt`, `*.zip` in `00_Mathias_contribution/Kaggle_Notebook/`.
- **Commit & Push:** Änderungen (inkl. Proof-Bilder, Skripte, .gitignore, Umzug Kaggle_Notebook) wurden committet und nach `main` gepusht.

---

## Nächste Schritte (kurz)

- Inferenz auf Bono-Test-Patches mit einem der neuen Kaggle-Checkpoints (z. B. `last.ckpt`) ausführen: `04_inference_bono.py` (Checkpoint-Pfad ggf. auf `00_Mathias_contribution/Kaggle_Notebook/last.ckpt` oder `models/` setzen).
- Bono-Vergleichsbild mit neuem Checkpoint: `plot_bono_test_comparison.py` erneut ausführen.
- Optional: Vollständige Bono-Inferenz (`05_inference_bono_full.py`) und Ghana-Karte (`06_ghana_map_galamsey_bono.py`) mit dem neuen Modell.
- Weitere Schritte (Fine-Tuning, manuelles Labeln Bono) siehe `Next_Steps_Inference_Finetuning.md` im gleichen Ordner.
