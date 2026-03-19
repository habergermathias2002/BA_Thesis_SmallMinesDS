# Kaggle Notebooks – Übersicht

## Aktueller Stand

| Notebook | Status | Beschreibung |
|----------|--------|--------------|
| `BA_Thesis_01_Training_SmallMinesDS.ipynb` | ✅ Bereit | Base-Training auf SmallMinesDS |
| `BA_Thesis_02_Finetuning_Bono.ipynb` | 🔜 Geplant | Fine-Tuning auf manuell gelabelten Bono-Patches |

Nächste Schritte: siehe `Next_Steps_Inference_Finetuning.md`

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

1. `data/GhanaMiningPrithvi/` als ZIP verpacken und als Kaggle-Dataset `ghana-mining-prithvi` hochladen:
   ```bash
   zip -r GhanaMiningPrithvi.zip data/GhanaMiningPrithvi/
   ```
2. Notebook auf [kaggle.com/code](https://www.kaggle.com/code) hochladen → **Add data** → `ghana-mining-prithvi`
3. **Settings → Accelerator → GPU P100** (empfohlen) oder T4
4. Zellen 1–5 ausführen (~2–3 h auf P100)
5. Output-Panel → `checkpoints/` → besten Checkpoint herunterladen → lokal als `models/prithvi-v2-300-base.ckpt` speichern
