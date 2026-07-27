# Kaggle Notebooks – Übersicht

## Aktueller Stand

| Notebook | Status | Beschreibung |
|----------|--------|--------------|
| `BA_Thesis_01_Training_SmallMinesDS.ipynb` | ✅ Bereit | Base-Training auf SmallMinesDS |
| `BA_Thesis_02_Finetuning_Bono.ipynb` | ✅ Bereit | Fine-Tuning auf Bono-Labels (65 Patches) |

Nächste Schritte nach FT: Inferenz Base vs. Fine-tuned — siehe `Next_Steps_Inference_Finetuning.md`

---

## Notebook 1: `BA_Thesis_01_Training_SmallMinesDS.ipynb`

**Ziel:** Prithvi-EO v2 auf SmallMinesDS trainieren und validen Checkpoint erzeugen.

### Struktur (5 Zellen)

| Zelle | Inhalt |
|-------|--------|
| 1 | Pakete installieren |
| 2 | Imports, Pfade, 6-Band-Check |
| 3 | Konfiguration (Means/Stds, DataModule, Modell) |
| 4 | Training (frozen backbone, max 50 Epochen, Early Stopping) |
| 5 | Evaluation + Checkpoint-Übersicht + Download-Hinweis |

### Kaggle-Setup

1. `data/GhanaMiningPrithvi/` als ZIP verpacken und als Kaggle-Dataset hochladen
2. Notebook hochladen → **Add data**
3. GPU P100/T4
4. Besten Checkpoint lokal als `models/prithvi-v2-300-base.ckpt` speichern

---

## Notebook 2: `BA_Thesis_02_Finetuning_Bono.ipynb`

**Ziel:** Base-`last.ckpt` auf `GhanaMiningPrithvi_bono` nachtrainieren.

### Strategie (Overfitting-Schutz bei 65 Patches)

| Komponente | Status |
|---|---|
| Encoder-Blöcke 0–19 | **eingefroren** |
| Encoder-Blöcke 20–23 (letzte 4) + LayerNorm | **trainierbar** |
| UperNet-Decoder + Head | **trainierbar** |
| LR | `5e-4` |
| Class weights | `[0.2, 0.8]` (Non-Mining / Mining) |
| Monitor | `val/Multiclass_Jaccard_Index` (IoU) |
| Zusätzlich geloggt | Val **F1**, klassenweise IoU/F1 (CSV + Konsole) |

### Kaggle-Setup

1. Dataset zippen und hochladen:
   ```bash
   zip -r GhanaMiningPrithvi_bono.zip data/GhanaMiningPrithvi_bono/
   ```
2. Base-Checkpoint (`last.ckpt`) als **zweites** Dataset hochladen
3. Notebook + GPU → Zellen 1–5
4. Besten Checkpoint lokal als `models/prithvi-v2-300-finetuned.ckpt` speichern

### Erwartete Laufzeit

~20–40 min auf P100 (40 Epochen max, EarlyStopping patience=8)
