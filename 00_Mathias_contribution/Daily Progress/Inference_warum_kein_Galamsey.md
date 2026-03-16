# Warum zeigt die Inferenz kein Galamsey? Mögliche Ursachen

Trotz trainiertem Modell und Anwendung auf die Bono-Region liefert die Inferenz praktisch überall „Non-Mining“ (Wahrscheinlichkeit ≈ 0, keine roten Flächen). Mögliche Gründe und konkrete Checks.

---

## 1. Domain Shift (sehr wahrscheinlich)

**Training:** SmallMinesDS – **SW Ghana**, Jahre **2016 + 2022**, andere Szenen/Zeitpunkte.  
**Inferenz:** **Bono/Bono East**, **Januar 2025**, anderes Gebiet, andere Saison.

- Das Modell hat gelernt: „So sehen Mining/Non-Mining in den Trainingsdaten aus.“
- Bono 2025 kann sich in Beleuchtung, Vegetation, Boden, Atmosphäre und S2-Verarbeitung unterscheiden.
- Dann liegen die **Feature-Vektoren** der Bono-Patches in Bereichen, die das Modell fast nur mit „Non-Mining“ verbindet → überall niedrige Mining-Wahrscheinlichkeit.

**Was hilft:** Wenige Bono-Patches mit Galamsey-Labels besorgen (oder selbst labeln) und das Modell darauf **nachtrainieren** (Fine-Tuning), oder Backbone nicht einfrieren und mit gemischtem Datensatz (SmallMinesDS + Bono) trainieren.

---

## 2. Normalisierung passt nicht zur Bono-Verteilung

Die Inferenz nutzt **means/std von SmallMinesDS** (aus dem Training).  
Formel: `(Pixel × 10.000 − mean) / std`.

- Wenn die **tatsächlichen Bono-Werte** (nach ×10.000) systematisch anders verteilt sind (z. B. andere Helligkeit/Saison), werden die normalisierten Werte in Bereiche geschoben, die das Modell kaum gelernt hat → Tendenz zu einer Klasse (hier: Non-Mining).

**Check:**  
Für ein paar Bono-Patches (z. B. aus `patches_bono_test`) pro Band **Mittelwert und Standardabweichung** berechnen und mit den SmallMinesDS-Werten vergleichen. Wenn die Abweichung groß ist, lohnt sich:  
- entweder **Bono-spezifische means/std** zu schätzen und in der Inferenz zu nutzen,  
- oder das Modell mit Bono-Daten (oder gemischt) so zu trainieren, dass die Normalisierung zur Anwendungsdomäne passt.

---

## 3. Klassenungleichgewicht im Training

Wenn im Training **sehr wenig Mining** (z. B. nur wenige Prozent der Pixel) vorkommt, kann das Modell gelernt haben, „sicherheitshalber“ fast überall **Non-Mining** zu sagen.

**Check:**  
Anteil Mining-Pixel in den Trainings-Labels prüfen (z. B. über alle `*_MASK.tif`). Wenn stark unbalanciert, beim nächsten Training z. B. Class Weights oder Oversampling für Mining in Betracht ziehen.

---

## 4. Backbone eingefroren

Im Colab-Training ist **freeze_backbone=True** gesetzt. Es wird nur der **Decoder/Head** auf SmallMinesDS angepasst; die Prithvi-Features bleiben „allgemein“.

- Für die Trainingsdomäne (SW Ghana) kann das gut funktionieren.
- Für Bono (andere Region/Zeit) können diese Features **Mining vs. Non-Mining** nicht gut trennen → das Modell bleibt bei Non-Mining.

**Was hilft:** Beim nächsten Training den Backbone **nicht** einfrieren (oder nur die letzten Backbone-Layer auftauen) und – wenn möglich – etwas Bono-Material mit ins Training nehmen.

---

## 5. Technische Fehler (weniger wahrscheinlich, aber prüfenswert)

- **Bandreihenfolge:** GEE-Export und Skripte nutzen B2, B3, B4, B8A, B11, B12 in derselben Reihenfolge wie das Modell – aktuell konsistent.
- **Skalierung:** GEE teilt durch 10.000; Skript 02 multipliziert mit 10.000 → Wertebereich wie im Training (raw DN 0–10.000).  
  Einmal prüfen: Ein Bono-Patch nach ×10.000 – liegen typische Werte in einem ähnlichen Bereich wie in SmallMinesDS (z. B. Blau ~1.400)?
- **Klasse im Code:** Mining = Kanal 1 (`probs[0, 1]`) – mit Prithvi/terratorch und 2 Klassen (Non_mining, Mining) üblich. Nur zur Sicherheit: Ein **Trainings-Patch mit bekanntem Mining** durch das geladene Modell jagen und schauen, ob dort hohe Mining-Wahrscheinlichkeit rauskommt.

---

## Kurzfassung

| Ursache              | Wahrscheinlichkeit | Nächster Schritt |
|----------------------|--------------------|-------------------|
| Domain Shift (Ort/Zeit) | Hoch            | Bono-Labels + Fine-Tuning oder gemischtes Training |
| Normalisierung       | Mittel             | Bono-Statistik (mean/std) vs. SmallMinesDS vergleichen |
| Klassenungleichgewicht | Mittel          | Mining-Anteil in Trainings-Labels prüfen |
| Backbone eingefroren | Mittel             | Beim nächsten Training Backbone (teilweise) auftauen |
| Bug (Bänder/Skala)   | Gering             | Ein Trainings-Mining-Patch durch Modell laufen lassen; Bono-Werte stichprobenartig prüfen |

Pragmatisch: **Domain Shift** und **fehlende Bono-Anpassung** sind die plausibelsten Gründe. Sobald du ein paar gesicherte Galamsey-Standorte in Bono hast (z. B. aus Mikrodaten oder Expertenwissen), lohnt sich Nachtraining/Fine-Tuning auf Bono (oder gemischt mit SmallMinesDS).
