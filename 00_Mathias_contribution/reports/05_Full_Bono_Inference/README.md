# 05 – Full Bono Inference (Fine-Tuned)

Ergebnisse aus:
- Lokal: `00_Mathias_contribution/scripts/05_inference_bono_full.py`
- Kaggle: `00_Mathias_contribution/Kaggle_Notebook/BA_Thesis_03_Full_Bono_Inference.ipynb`

**Analyse-Schritt:** Flächige Anwendung des Bono-Fine-Tuned-Modells auf `Bono_Merged_2025.tif`.

| Datei / Ordner | Inhalt |
|---|---|
| `tables/inference_stats.txt` | Patch-Zähler, Mining-Anteil, Pfade |
| `figures/ghana_map_galamsey_bono_ft.png` | Übersichtskarte (nach Lauf von `06_ghana_map_…`) |
| `data/inference_bono_full_ft/` | GeoTIFFs `prediction_prob.tif`, `prediction_binary.tif` |

## Ablauf

1. **Smoke-Test (lokal):** `LIMIT_PATCHES=100 python 00_Mathias_contribution/scripts/05_inference_bono_full.py`
2. **Voll-Lauf (Kaggle GPU):** Notebook 03, `LIMIT_PATCHES=0`
3. **Karte:** `python 00_Mathias_contribution/scripts/06_ghana_map_galamsey_bono.py`
