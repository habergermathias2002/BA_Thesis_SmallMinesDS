# Next Steps: Inferenz, Fine-Tuning & Evaluation

Dieses Dokument beschreibt alle Schritte nach dem Base-Training auf SmallMinesDS.

---

## Voraussetzung

- [ ] Kaggle-Training abgeschlossen (`BA_Thesis_01_Training_SmallMinesDS.ipynb`)
- [ ] Besten Checkpoint heruntergeladen → lokal als `models/prithvi-v2-300-base.ckpt`

---

## Phase 2: Inferenz auf Bono (Base-Modell)

**Ziel:** Prüfen, ob das korrekt trainierte Modell auf der Bono-Region plausible Vorhersagen macht.

- [ ] `04_inference_bono.py` mit `OUTPUT_SUFFIX = "_base"` und SmallMinesDS-Normalisierungsstatistiken ausführen
- [ ] `plot_bono_test_comparison.py` ausführen → Vergleichsbild Base-Modell
- [ ] `plot_model_proof.py` ausführen → Proof auf Trainingsdaten

**Erfolgskriterium:** Mining-Patches zeigen hohe P(Mining), Non-Mining-Patches zeigen niedrige P(Mining). Bono-Vorhersage: räumlich kohärentes Muster (nicht mehr überall gleichmäßig ~55%).

---

## Phase 3: Manuelles Labeln von Bono-Patches

**Ziel:** Wenige Bono-Patches mit Ground-Truth-Masken versehen, um Fine-Tuning auf der Zielregion zu ermöglichen.

### Werkzeuge
- **QGIS:** Patch in QGIS öffnen (True Color: R=B4, G=B3, B=B2), Mining-Flächen als Polygone digitalisieren → als Raster-Maske exportieren (0=Non-Mining, 1=Mining, 128×128 px, UINT8)
- **Label Studio** oder **CVAT** als webbasierte Alternativen

### Empfehlung
- [ ] Mindestens 20–30 Patches labeln (je ~50% Mining, ~50% Non-Mining)
- [ ] Patches aus verschiedenen Distrikten wählen (Bono + Bono East)
- [ ] Masken im Format `GH_BONO_XXXX_2025_MASK.tif` (1-Band, UINT8) speichern
- [ ] In `data/GhanaMiningPrithvi_bono/training/` und `.../validation/` ablegen (gleiches Format wie SmallMinesDS)

---

## Phase 4: Fine-Tuning auf Bono-Daten (Kaggle Notebook 2)

**Ziel:** Modell an Bono-Domäne anpassen. Zwei Strategien:

### Option A: Fine-Tuning nur auf Bono-Labels (empfohlen bei wenigen Labels)
- Startet vom Base-Checkpoint (Phase 1)
- Backbone eingefroren oder partiell aufgetaut
- Nur Bono-Daten im Training
- Vorteil: schnell, kein Catastrophic Forgetting

### Option B: Mixed Training (SmallMinesDS + Bono)
- Gemeinsames Training auf beiden Datensätzen
- Verhindert, dass das Modell SmallMinesDS „vergisst"
- Empfohlen wenn ≥50 Bono-Patches vorhanden

### Backbone-Strategie
| Ansatz | Trainierbare Params | LR | Wann |
|--|--|--|--|
| Frozen backbone | ~15 M | 1e-3 | < 20 Bono-Patches |
| Letzte 4 Transformer-Blöcke auftauen | ~50 M | 5e-4 | 20–50 Patches |
| Volles Fine-Tuning | ~318 M | 1e-4 | > 50 Patches |

### To-Do
- [ ] Kaggle Notebook 2 erstellen: `BA_Thesis_02_Finetuning_Bono.ipynb`
- [ ] Bono-Daten als neues Kaggle-Dataset hochladen (`bono-labeled-patches`)
- [ ] Trainieren, besten Checkpoint speichern → `models/prithvi-v2-300-finetuned.ckpt`

---

## Phase 5: Inferenz nach Fine-Tuning

**Ziel:** Vergleich Base-Modell vs. Fine-Tuned-Modell auf Bono-Testpatches.

- [ ] `04_inference_bono.py` mit Fine-Tuned-Checkpoint ausführen, `OUTPUT_SUFFIX = "_finetuned"`
- [ ] `plot_bono_test_comparison.py` mit `OUTPUT_SUFFIX = "_finetuned"` ausführen
- [ ] 4-Panel-Vergleich erstellen: Satellitenbild | Base | Fine-Tuned | Differenz
- [ ] Optional: volle Bono-Region mit `05_inference_bono_full.py`

---

## Phase 6: Ergebnisaufbereitung für die Thesis

- [ ] Karte Bono-Region mit Mining-Vorhersage overlay (`06_ghana_map_galamsey_bono.py`)
- [ ] Quantitative Auswertung: Precision, Recall, IoU auf gelabelten Bono-Patches
- [ ] Verknüpfung mit Mikrodaten: Überlapp von Vorhersageflächen mit Survey-Standorten (`01_Microdata/`)
- [ ] Dokumentation in `20260314_Documentation.md` aktualisieren
