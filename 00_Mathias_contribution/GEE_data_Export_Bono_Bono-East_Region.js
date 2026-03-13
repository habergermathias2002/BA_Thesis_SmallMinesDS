// ==============================================================================
// GEE Skript: Datenexport für SmallMinesDS Replikation (Bono & Bono East)
// Zeitraum: Januar 2025
// ==============================================================================

// ------------------------------------------------------------------------------
// 1. GRENZEN DEFINIEREN
// ------------------------------------------------------------------------------
// Wir laden den globalen FAO/GAUL Datensatz für Verwaltungsgrenzen (Level 1 = Regionen).
var gaul = ee.FeatureCollection("FAO/GAUL/2015/level1");

// Da die alte Region "Brong Ahafo" 2018 in Bono, Bono East und Ahafo aufgeteilt 
// wurde, suchen wir sicherheitshalber nach den alten UND neuen Namen.
var region = gaul.filter(ee.Filter.inList('ADM1_NAME', ['Bono', 'Bono East', 'Brong Ahafo']));

// Die interaktive Karte unten zentriert sich auf unser Gebiet (Zoom-Level 8).
Map.centerObject(region, 8);
Map.addLayer(region, {}, 'Untersuchungsgebiet (Bono / Bono East)');


// ------------------------------------------------------------------------------
// 2. WOLKENMASKIERUNG
// ------------------------------------------------------------------------------
// Satellitenbilder haben oft ein Qualitäts-Band (hier: QA60), in dem der 
// Satellit speichert, ob er an einem Pixel eine Wolke vermutet.
function maskS2clouds(image) {
  var qa = image.select('QA60');
  
  // Im QA60-Band steht Bit 10 für dicke Wolken und Bit 11 für feine Zirruswolken.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  
  // Wir behalten nur Pixel, bei denen BEIDE Wolken-Bits auf 0 (keine Wolke) stehen.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  
  // Sentinel-2 speichert Helligkeitswerte im Bereich 0 bis 10.000.
  // Foundation Models erwarten meist Werte zwischen 0 und 1. 
  // Daher teilen wir hier durch 10000 (Skalierung).
  return image.updateMask(mask).divide(10000); 
}


// ------------------------------------------------------------------------------
// 3. BILDER LADEN UND FILTERN
// ------------------------------------------------------------------------------
// Wir laden die atmosphärisch korrigierten Bilder (Surface Reflectance / L2A).
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region) // Nur Bilder, die unser Gebiet berühren
  
  // WARUM JANUAR? 
  // In Ghana herrscht von Dezember bis Februar die Trockenzeit (Harmattan).
  // 1. Wir haben fast keine Wolken (im Regenwald sonst ein riesiges Problem).
  // 2. Die Vegetation ist trockener/weniger dicht. Die offene, umgegrabene
  // Erde der Galamsey-Minen hebt sich optisch extrem stark vom Wald ab.
  .filterDate('2025-01-01', '2025-01-31') 
  
  // Vorab-Filter: Wir werfen Bilder sofort weg, wenn sie zu >10% aus Wolken bestehen.
  // Das spart Google-Rechenleistung.
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  
  // Wir wenden unsere Wolkenmaskierungs-Funktion von oben auf alle verbleibenden Bilder an.
  .map(maskS2clouds);


// ------------------------------------------------------------------------------
// 4. MOSAIK ERSTELLEN
// ------------------------------------------------------------------------------
// WARUM MEDIAN?
// Wenn der Satellit im Januar z.B. 4 Mal über denselben Ort geflogen ist, 
// berechnen wir für jeden einzelnen Pixel den Median-Wert aus diesen 4 Bildern.
// Der geniale Effekt: Temporäre Störungen (ein Vogelschwarm, ein winziger 
// Wolkenschatten, der die Maske überlebt hat) werden als "Ausreißer" einfach 
// herausgerechnet. Wir bekommen ein makelloses, synthetisches Bild.
var medianMosaic = s2.median().clipToCollection(region);


// ------------------------------------------------------------------------------
// 5. BÄNDER FÜR DAS PRITHVI-MODELL AUSWÄHLEN
// ------------------------------------------------------------------------------
// Das vortrainierte Prithvi-EO 6-Band-Modell der Kollegen erwartet EXAKT diese Bänder.
// B2=Blau, B3=Grün, B4=Rot, B8A=Schmales Nahes Infrarot, B11 & B12 = Kurzwelliges Infrarot.
var finalImage = medianMosaic.select(['B2', 'B3', 'B4', 'B8A', 'B11', 'B12']);

// Vorschau-Layer auf der Karte. Wir nutzen B11 (SWIR), B8A (NIR) und B4 (Rot).
// In dieser "Falschfarben"-Darstellung leuchtet lebende Vegetation rot, Wasser schwarz
// und nackte Erde/Minen stechen hell türkis/weißlich hervor. Perfekt zur Kontrolle!
var visParams = {bands: ['B11', 'B8A', 'B4'], min: 0, max: 0.3};
Map.addLayer(finalImage, visParams, 'Sentinel-2 Mosaik 2025 (Falschfarben)');


// ------------------------------------------------------------------------------
// 6. EXPORT STARTEN
// ------------------------------------------------------------------------------
// Wir schicken den Auftrag an die Google-Server, das Bild in dein Drive zu legen.
Export.image.toDrive({
  image: finalImage,
  description: 'Sentinel2_Bono_Januar2025_10m', // Dateiname
  folder: 'GEE_Galamsey_Exports',              // Ordner in Google Drive
  scale: 10,                                   // Auflösung in Metern pro Pixel. 
                                               // GEE rechnet die 20m-Bänder (B11/B12) 
                                               // hier automatisch auf 10m um (Resampling).
  region: region.geometry(),                   // Die Form, die exportiert wird
  maxPixels: 1e13,                             // Erhöhtes Limit, da Bono recht groß ist
  crs: 'EPSG:32630'                            // Das Koordinatensystem für Westafrika 
                                               // (UTM Zone 30N). Verhindert Verzerrungen.
});
