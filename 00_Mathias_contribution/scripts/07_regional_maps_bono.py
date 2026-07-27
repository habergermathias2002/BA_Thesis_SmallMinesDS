"""
07_regional_maps_bono.py
==========================

Kurz: Erzeugt zwei detaillierte Übersichtskarten (Bono, Bono East) auf Basis der
vollflächigen Fine-Tuned-Modell-Inferenz (Skript 05). Jede Karte zeigt:
  - Regionsgrenze (fett) und Distriktgrenzen (dünn, mit Namen)
  - Flüsse (HydroRIVERS, gefiltert auf größere Gewässer)
  - Mining-Vorhersage als rote Überlagerung (binäre Maske, feste Deckkraft)
  - Umrandungen (Bounding Boxes) um Mining-Hotspot-Cluster. Cluster werden über
    zusammenhängende Komponenten + DBSCAN gebildet, sodass auch Hotspots mit
    kurzen Unterbrechungen (z. B. wiederkehrende Minenaktivität entlang eines
    Flusses) als ein Cluster erkannt werden, nicht nur strikt zusammenhängende
    Pixel.

Post-Processing gegen False Positives:
  Skript 05 speichert bereits die volle, kontinuierliche Mining-Wahrscheinlichkeit
  P(Mining) pro Pixel in prediction_prob.tif (keine erneute GPU-Inferenz nötig).
  Der Standard-Entscheidungsgrenze von 50% (entspricht Argmax bei 2 Klassen) reagiert
  sehr sensibel und erzeugt viele False Positives abseits von Flüssen (vermutlich
  Ackerland/nackter Boden). Dieses Skript wendet daher direkt auf das Fenster jeder
  Region an:
    1. CONFIDENCE_THRESHOLD (Standard 0.90) statt 0.5 als Entscheidungsgrenze.
    2. rasterio.features.sieve mit SIEVE_SIZE (Standard 10 Pixel), um verbliebene
       kleine, isolierte Rausch-Cluster zu entfernen (echte Minen sind größere,
       zusammenhängende Krater).

Warum nicht direkt sklearn.DBSCAN auf allen Mining-Pixeln?
  Jede Region enthält ~1.4-1.7 Mio. Mining-Pixel (10 m Auflösung). DBSCAN auf
  so vielen Punkten mit einem Radius von 500 m wäre extrem langsam/speicher-
  intensiv. Stattdessen: (1) scipy.ndimage.label fasst direkt benachbarte
  Pixel zu "Blobs" zusammen (schnell, ~26k Blobs statt 1.5 Mio. Punkte),
  (2) DBSCAN läuft nur noch auf den Blob-Zentroiden und verschmilzt Blobs,
  die innerhalb von CLUSTER_EPS_M liegen (überbrückt Lücken). Ergebnis ist
  äquivalent zu einer lückentoleranten Clusterbildung, aber praktikabel schnell.

Requirements: geopandas, matplotlib, rasterio, scipy, scikit-learn.

Usage:
  python 00_Mathias_contribution/scripts/07_regional_maps_bono.py
  (im Repo-Root ausführen)

Output:
  reports/05_Full_Bono_Inference/figures/regional_map_bono.png
  reports/05_Full_Bono_Inference/figures/regional_map_bono_east.png
  reports/05_Full_Bono_Inference/figures/regional_map_bono_combined.png
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.features import sieve, rasterize
from rasterio.warp import (
    reproject,
    Resampling,
    calculate_default_transform,
    transform_bounds,
    transform as warp_transform,
)
from scipy import ndimage
from sklearn.cluster import DBSCAN

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GADM_GPKG = os.path.join(REPO_ROOT, "data", "cache", "gadm41_GHA.gpkg")
GADM_GHA_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_GHA.gpkg"

RIVERS_SHP = os.path.join(
    REPO_ROOT, "data", "cache", "rivers", "HydroRIVERS_v10_af_shp", "HydroRIVERS_v10_af.shp"
)

PRED_PROB = os.path.join(REPO_ROOT, "data", "inference_bono_full_ft", "prediction_prob.tif")

OUT_DIR = os.path.join(REPO_ROOT, "reports", "05_Full_Bono_Inference", "figures")

# --- Tunable parameters (siehe Plan) -----------------------------------
# Das Modell reagiert bei P(Mining) >= 50% (Standard-Argmax-Entscheidung) sehr sensibel
# und erzeugt viele False Positives abseits von Flüssen (vermutlich Ackerland/nackter
# Boden). CONFIDENCE_THRESHOLD hebt die Entscheidungsschwelle an; SIEVE_SIZE entfernt
# danach noch verbliebene kleine, isolierte Rausch-Cluster (< SIEVE_SIZE Pixel).
DEFAULT_CONFIDENCE_THRESHOLD = 0.90
# Per Umgebungsvariable überschreibbar, um Vergleichskarten bei anderen Schwellwerten
# zu erzeugen, z. B.: CONFIDENCE_THRESHOLD=0.75 python 07_regional_maps_bono.py
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD))
SIEVE_SIZE = 10               # rasterio.features.sieve: Cluster < 10 zusammenh. Pixel -> entfernt
SIEVE_CONNECTIVITY = 8        # 8-Konnektivität, konsistent mit der Blob-Erkennung unten
# Die Hotspot-Cluster-Parameter (CLUSTER_EPS_M, MIN_CLUSTER_PIXELS unten) sind auf den
# Standard-Threshold von 90% kalibriert. Bei niedrigeren Vergleichs-Thresholds (z. B. 50%,
# 75%) gibt es um Größenordnungen mehr Mining-Pixel, wodurch dieselben Cluster-Parameter
# tausende, kaum lesbare Boxen erzeugen würden. Für einen sauberen Dichte-Vergleich werden
# die Hotspot-Boxen daher standardmäßig ausgeblendet, sobald ein anderer Threshold als der
# Standard verwendet wird (per SHOW_HOTSPOT_BOXES env var explizit erzwingbar).
_show_boxes_env = os.environ.get("SHOW_HOTSPOT_BOXES")
if _show_boxes_env is not None:
    SHOW_HOTSPOT_BOXES = _show_boxes_env not in ("0", "false", "False")
else:
    SHOW_HOTSPOT_BOXES = abs(CONFIDENCE_THRESHOLD - DEFAULT_CONFIDENCE_THRESHOLD) < 1e-9


def _threshold_suffix():
    """Dateinamens-Suffix für Nicht-Standard-Thresholds, z. B. '_conf75' bei 0.75."""
    if abs(CONFIDENCE_THRESHOLD - DEFAULT_CONFIDENCE_THRESHOLD) < 1e-9:
        return ""
    return f"_conf{int(round(CONFIDENCE_THRESHOLD * 100))}"

MAX_MAP_PX = 2000          # Downsampling-Obergrenze für die Mining-Overlay-Anzeige
OVERLAY_ALPHA = 0.55       # feste Deckkraft der roten Mining-Überlagerung (0/1-Maske)
RIVER_MIN_ORDER = 3        # HydroRIVERS ORD_STRA (Strahler-Ordnung) >= 3 -> "größere" Flüsse
RIVER_BBOX_BUFFER_DEG = 0.1

RASTER_BUFFER_M = 3000     # Puffer um Regionsgrenze beim Fenster-Lesen der Raster
CLUSTER_EPS_M = 500.0      # Lückentoleranz zwischen Blobs (DBSCAN eps, Meter)
CLUSTER_MIN_SAMPLES = 1    # Mindestanzahl Blobs pro DBSCAN-Gruppe (Blobs sind bereits Pixel-Gruppen)
MIN_CLUSTER_PIXELS = 500   # Mindest-Gesamtpixelzahl (verschmolzen; 500 px = 5 ha), um eine
                           # Box zu zeichnen -> hebt nur die auffälligsten Hotspots hervor.
                           # (Nach Anhebung von CONFIDENCE_THRESHOLD auf 90% + Sieve-Filter
                           # gibt es insgesamt viel weniger, aber saubere Mining-Pixel; dieser
                           # Wert wurde entsprechend neu kalibriert, siehe Cluster-Größenverteilung.)
BBOX_PAD_M = 100.0         # Rand um die Cluster-Bounding-Box

REGIONS = ["Bono", "Bono East"]


# --------------------------------------------------------------------------
# Boundaries (GADM)
# --------------------------------------------------------------------------
def get_boundaries():
    """Lädt Regionen (ADM_1) und Distrikte (ADM_2) aus dem GADM-Cache (Download-Fallback)."""
    os.makedirs(os.path.dirname(GADM_GPKG), exist_ok=True)
    if not os.path.exists(GADM_GPKG):
        import urllib.request
        print("Lade GADM Ghana Grenzen herunter (einmalig)...")
        urllib.request.urlretrieve(GADM_GHA_URL, GADM_GPKG)
    regions = gpd.read_file(GADM_GPKG, layer="ADM_ADM_1").to_crs("EPSG:4326")
    districts = gpd.read_file(GADM_GPKG, layer="ADM_ADM_2").to_crs("EPSG:4326")
    return regions, districts


# --------------------------------------------------------------------------
# Rivers (HydroRIVERS)
# --------------------------------------------------------------------------
def get_rivers_for_bounds(bounds_wgs84):
    """Lädt Flüsse innerhalb einer Bounding Box, gefiltert auf größere Gewässer."""
    if not os.path.exists(RIVERS_SHP):
        raise FileNotFoundError(
            f"HydroRIVERS Shapefile nicht gefunden unter {RIVERS_SHP}. "
            "Bitte HydroRIVERS_v10_af_shp.zip von "
            "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_af_shp.zip "
            "herunterladen und nach data/cache/rivers/ entpacken."
        )
    minx, miny, maxx, maxy = bounds_wgs84
    bbox = (
        minx - RIVER_BBOX_BUFFER_DEG,
        miny - RIVER_BBOX_BUFFER_DEG,
        maxx + RIVER_BBOX_BUFFER_DEG,
        maxy + RIVER_BBOX_BUFFER_DEG,
    )
    rivers = gpd.read_file(RIVERS_SHP, bbox=bbox)
    if "ORD_STRA" in rivers.columns:
        major = rivers[rivers["ORD_STRA"] >= RIVER_MIN_ORDER]
        if len(major) > 0:
            return major
    return rivers


# --------------------------------------------------------------------------
# Thresholding + denoising (windowed read of the raw probability raster)
# --------------------------------------------------------------------------
def _clip_window_to_raster(window, src_width, src_height):
    full = Window(0, 0, src_width, src_height)
    return window.intersection(full)


def compute_region_prediction(region_geom_native):
    """Liest P(Mining) aus prediction_prob.tif für das Regionsfenster, wendet
    CONFIDENCE_THRESHOLD an (statt Standard-Argmax bei 0.5) und entfernt danach
    kleine Rausch-Cluster (< SIEVE_SIZE zusammenhängende Pixel) mit einem
    Sieve-Filter. Rückgabe: (binary_array uint8, win_transform, native_crs).

    Wichtig: Da die Mosaik-Vorhersage (prediction_prob.tif) auch benachbarte
    Regionen abdeckt (z. B. Ahafo, Ashanti, Bono East, Savannah, Western North
    überlappen die rechteckige Bounding Box von Bono), wird die Vorhersage hier
    zusätzlich auf die tatsächliche (nicht rechteckige) Regions-Polygonform
    maskiert. Ohne diese Maskierung würden Mining-Marker aus Nachbarregionen
    innerhalb der Bounding Box fälschlich auf der Bono-/Bono-East-Karte auftauchen.
    """
    minx, miny, maxx, maxy = region_geom_native.bounds
    with rasterio.open(PRED_PROB) as src:
        window = from_bounds(
            minx - RASTER_BUFFER_M, miny - RASTER_BUFFER_M,
            maxx + RASTER_BUFFER_M, maxy + RASTER_BUFFER_M,
            transform=src.transform,
        ).round_offsets().round_lengths()
        window = _clip_window_to_raster(window, src.width, src.height)

        win_transform = src.window_transform(window)
        prob = src.read(1, window=window, out_dtype=np.float32)
        native_crs = src.crs

    region_mask = rasterize(
        [(region_geom_native, 1)],
        out_shape=prob.shape,
        transform=win_transform,
        fill=0,
        dtype=np.uint8,
    )

    binary = ((prob >= CONFIDENCE_THRESHOLD) & (region_mask == 1)).astype(np.uint8)
    n_before = int(binary.sum())
    binary = sieve(binary, size=SIEVE_SIZE, connectivity=SIEVE_CONNECTIVITY)
    n_after = int(binary.sum())
    print(
        f"  Threshold {CONFIDENCE_THRESHOLD:.0%} (auf Regionsform maskiert): "
        f"{n_before} Mining-Pixel -> nach Sieve-Filter (< {SIEVE_SIZE} Pixel entfernt): "
        f"{n_after} Pixel ({100 * (1 - n_after / max(n_before, 1)):.1f}% Rauschen entfernt)."
    )
    return binary, win_transform, native_crs


def render_mining_overlay(binary_array, win_transform, native_crs):
    """Reprojiziert die (bereits geschwellte + entrauschte) Binärmaske nach WGS84.

    Resampling.max beim Downsampling sorgt dafür, dass jede noch so kleine,
    isolierte Mining-Fläche beim Verkleinern sichtbar bleibt, statt durch
    bilineares Glätten zu einem flächendeckenden "Rot-Schleier" zu verschwimmen.
    """
    height, width = binary_array.shape
    win_bounds_native = rasterio.transform.array_bounds(height, width, win_transform)
    bounds_wgs84 = transform_bounds(native_crs, "EPSG:4326", *win_bounds_native)
    dst_transform, out_width, out_height = calculate_default_transform(
        native_crs, "EPSG:4326", width, height, *win_bounds_native
    )
    if out_width > MAX_MAP_PX or out_height > MAX_MAP_PX:
        scale = min(MAX_MAP_PX / out_width, MAX_MAP_PX / out_height)
        out_width = max(1, int(out_width * scale))
        out_height = max(1, int(out_height * scale))
        dst_transform = rasterio.transform.from_bounds(*bounds_wgs84, out_width, out_height)

    out = np.zeros((out_height, out_width), dtype=np.float32)
    reproject(
        source=binary_array.astype(np.float32),
        destination=out,
        src_transform=win_transform,
        src_crs=native_crs,
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.max,
    )
    extent = (bounds_wgs84[0], bounds_wgs84[2], bounds_wgs84[1], bounds_wgs84[3])
    return out, extent


# --------------------------------------------------------------------------
# Hotspot clustering (connected components + DBSCAN, gap-tolerant)
# --------------------------------------------------------------------------
def find_hotspot_boxes_wgs84(binary_array, win_transform, native_crs):
    """Findet Mining-Hotspot-Cluster in der (bereits geschwellten + entrauschten)
    Binärmaske und liefert Bounding Boxes in WGS84.

    Rückgabe: Liste von (minx, miny, maxx, maxy, total_pixels) in EPSG:4326.
    """
    data = binary_array
    if data.sum() == 0:
        return []

    structure = np.ones((3, 3), dtype=int)  # 8-Konnektivität für direkt benachbarte Pixel
    labels, n_blobs = ndimage.label(data, structure=structure)
    if n_blobs == 0:
        return []

    objects = ndimage.find_objects(labels)
    blobs = []
    for i, sl in enumerate(objects):
        if sl is None:
            continue
        r0, r1 = sl[0].start, sl[0].stop
        c0, c1 = sl[1].start, sl[1].stop
        local_mask = labels[sl] == (i + 1)
        size = int(local_mask.sum())
        cx_px, cy_px = (c0 + c1) / 2.0, (r0 + r1) / 2.0
        x0, y0 = win_transform * (c0, r0)
        x1, y1 = win_transform * (c1, r1)
        cxn, cyn = win_transform * (cx_px, cy_px)
        blobs.append({
            "size": size,
            "minx": min(x0, x1), "maxx": max(x0, x1),
            "miny": min(y0, y1), "maxy": max(y0, y1),
            "cx": cxn, "cy": cyn,
        })

    if not blobs:
        return []

    centroids = np.array([[b["cx"], b["cy"]] for b in blobs])
    db = DBSCAN(eps=CLUSTER_EPS_M, min_samples=CLUSTER_MIN_SAMPLES).fit(centroids)

    groups = {}
    for b, lbl in zip(blobs, db.labels_):
        if lbl == -1:
            continue
        groups.setdefault(lbl, []).append(b)

    boxes_native = []
    for group in groups.values():
        total = sum(b["size"] for b in group)
        if total < MIN_CLUSTER_PIXELS:
            continue
        bminx = min(b["minx"] for b in group) - BBOX_PAD_M
        bmaxx = max(b["maxx"] for b in group) + BBOX_PAD_M
        bminy = min(b["miny"] for b in group) - BBOX_PAD_M
        bmaxy = max(b["maxy"] for b in group) + BBOX_PAD_M
        boxes_native.append((bminx, bminy, bmaxx, bmaxy, total))

    # Bounding Boxes von nativem CRS (Meter) nach WGS84 transformieren (alle 4 Ecken,
    # dann Achsen-ausgerichtete Box in WGS84 -> robust auch bei leichter UTM-Rotation)
    boxes_wgs84 = []
    for bminx, bminy, bmaxx, bmaxy, total in boxes_native:
        xs = [bminx, bminx, bmaxx, bmaxx]
        ys = [bminy, bmaxy, bminy, bmaxy]
        lons, lats = warp_transform(native_crs, "EPSG:4326", xs, ys)
        boxes_wgs84.append((min(lons), min(lats), max(lons), max(lats), total))

    return boxes_wgs84


# --------------------------------------------------------------------------
# Map rendering
# --------------------------------------------------------------------------
def _compute_region_layers(region_name, regions_gdf):
    """Bündelt die Pro-Region-Pipeline (Threshold+Sieve -> Reprojektion -> Clustering).

    Wird sowohl von build_regional_map (Einzelkarten) als auch von build_combined_map
    (kombinierte Karte) verwendet, damit beide exakt dieselbe Logik pro Region durchlaufen.

    Rückgabe: (region_geom GeoDataFrame mit 1 Zeile, mining_mask, extent, hotspot_boxes).
    """
    print(f"\n=== Verarbeite Region: {region_name} ===")
    region_geom = regions_gdf[regions_gdf["NAME_1"] == region_name]
    if region_geom.empty:
        raise ValueError(f"Region '{region_name}' nicht in GADM ADM_1 gefunden.")

    with rasterio.open(PRED_PROB) as src:
        native_crs = src.crs
    region_geom_native = region_geom.to_crs(native_crs).geometry.union_all()

    print(f"Lese Wahrscheinlichkeits-Raster, wende Threshold ({CONFIDENCE_THRESHOLD:.0%}) "
          f"+ Sieve-Filter an...")
    binary_array, win_transform, pred_crs = compute_region_prediction(region_geom_native)

    mining_mask, extent = render_mining_overlay(binary_array, win_transform, pred_crs)
    coverage_pct = 100.0 * float(np.mean(mining_mask > 0.5))
    print(f"  Mining-Anteil in Kartenausschnitt (downsampled): {coverage_pct:.2f}%")

    print("Suche Hotspot-Cluster (connected components + DBSCAN)...")
    hotspot_boxes = find_hotspot_boxes_wgs84(binary_array, win_transform, pred_crs)
    print(f"  {len(hotspot_boxes)} Hotspot-Cluster gefunden "
          f"(eps={CLUSTER_EPS_M:.0f} m, min_pixels={MIN_CLUSTER_PIXELS}).")

    return region_geom, mining_mask, extent, hotspot_boxes


def build_regional_map(region_name, regions_gdf, districts_gdf, out_path):
    district_subset = districts_gdf[districts_gdf["NAME_1"] == region_name]

    region_geom, mining_mask, extent, hotspot_boxes = _compute_region_layers(
        region_name, regions_gdf
    )

    print("Lade Flüsse (HydroRIVERS)...")
    rivers = get_rivers_for_bounds(region_geom.total_bounds)
    print(f"  {len(rivers)} Flusssegmente (ORD_STRA >= {RIVER_MIN_ORDER}, falls vorhanden).")

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(11, 13), facecolor="white")
    ax.set_facecolor("white")

    # Distrikte: weiße Fläche, dünne graue Kontur, Namen
    district_subset.plot(ax=ax, facecolor="white", edgecolor="#888888", linewidth=0.5, zorder=1)
    for _, row in district_subset.iterrows():
        centroid = row.geometry.centroid
        if centroid.is_empty:
            continue
        ax.annotate(
            str(row["NAME_2"]), (centroid.x, centroid.y),
            fontsize=6.5, ha="center", va="center", color="#555555", zorder=2,
        )

    # Regionsgrenze: fett, schwarz
    region_geom.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.4, zorder=3)

    # Flüsse: blau, Linienbreite nach Strahler-Ordnung
    if len(rivers) > 0:
        if "ORD_STRA" in rivers.columns:
            for order, group in rivers.groupby("ORD_STRA"):
                lw = 0.4 + 0.25 * float(order)
                group.plot(ax=ax, color="#3182bd", linewidth=lw, zorder=4)
        else:
            rivers.plot(ax=ax, color="#3182bd", linewidth=0.8, zorder=4)

    # Mining-Vorhersage: rote Überlagerung (binäre Maske, feste Deckkraft)
    rgba = np.zeros((*mining_mask.shape, 4))
    rgba[..., 0] = 1
    rgba[..., 3] = np.where(mining_mask > 0.5, OVERLAY_ALPHA, 0.0)
    ax.imshow(rgba, extent=extent, origin="upper", interpolation="nearest", zorder=5)

    # Hotspot-Cluster: orangene Bounding Boxes (nur wenn SHOW_HOTSPOT_BOXES aktiv)
    if SHOW_HOTSPOT_BOXES:
        for bminx, bminy, bmaxx, bmaxy, total in hotspot_boxes:
            rect = Rectangle(
                (bminx, bminy), bmaxx - bminx, bmaxy - bminy,
                linewidth=1.5, edgecolor="#e6550d", facecolor="none", zorder=6,
            )
            ax.add_patch(rect)

    # Kartenausschnitt
    minx, miny, maxx, maxy = region_geom.total_bounds
    pad = 0.08
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_title(
        f"{region_name} – Galamsey-Hotspots (Fine-Tuned-Modell, P(Mining) \u2265 "
        f"{CONFIDENCE_THRESHOLD:.0%}, Sieve-gefiltert)",
        fontsize=13,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.4, label="Regionsgrenze"),
        Line2D([0], [0], color="#888888", linewidth=0.5, label="Distriktgrenze"),
        Line2D([0], [0], color="#3182bd", linewidth=1.2, label="Fluss (HydroRIVERS)"),
        Patch(facecolor="red", alpha=OVERLAY_ALPHA,
              label=f"Modell-Vorhersage: Mining (P \u2265 {CONFIDENCE_THRESHOLD:.0%})"),
    ]
    if SHOW_HOTSPOT_BOXES:
        legend_handles.append(
            Patch(facecolor="none", edgecolor="#e6550d", linewidth=1.5, label="Hotspot-Cluster")
        )
    ax.legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=8, framealpha=0.9, borderaxespad=0.0,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Karte gespeichert: {out_path}")


def build_combined_map(regions_gdf, districts_gdf, out_path):
    """Kombinierte Übersichtskarte für Bono + Bono East auf einer gemeinsamen Achse.

    Durchläuft dieselbe Pro-Region-Pipeline wie build_regional_map (über
    _compute_region_layers) für beide Regionen, akkumuliert die Ergebnisse
    (Mining-Maske+Extent, Hotspot-Boxen, Distrikte) und zeichnet alles zusammen
    auf eine gemeinsame Karte mit Kartenausschnitt über die Vereinigung beider
    Regionsgrenzen.
    """
    print("\n=== Kombinierte Karte für Bono & Bono East ===")

    region_layers = []
    for region_name in REGIONS:
        region_geom, mining_mask, extent, hotspot_boxes = _compute_region_layers(
            region_name, regions_gdf
        )
        region_layers.append({
            "name": region_name,
            "geom": region_geom,
            "mask": mining_mask,
            "extent": extent,
            "boxes": hotspot_boxes,
        })

    district_subset = districts_gdf[districts_gdf["NAME_1"].isin(REGIONS)]
    combined_regions = regions_gdf[regions_gdf["NAME_1"].isin(REGIONS)]

    all_bounds = np.array([layer["geom"].total_bounds for layer in region_layers])
    minx, miny, maxx, maxy = (
        all_bounds[:, 0].min(), all_bounds[:, 1].min(),
        all_bounds[:, 2].max(), all_bounds[:, 3].max(),
    )

    print("Lade Flüsse (HydroRIVERS)...")
    rivers = get_rivers_for_bounds((minx, miny, maxx, maxy))
    print(f"  {len(rivers)} Flusssegmente (ORD_STRA >= {RIVER_MIN_ORDER}, falls vorhanden).")

    total_boxes = sum(len(layer["boxes"]) for layer in region_layers)

    # --- Plot ---
    pad = 0.08
    width_deg = (maxx - minx) + 2 * pad
    height_deg = (maxy - miny) + 2 * pad
    fig_width = 13.0
    fig_height = float(np.clip(fig_width * height_deg / max(width_deg, 1e-6), 13.0, 20.0))
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), facecolor="white")
    ax.set_facecolor("white")

    # Distrikte (beider Regionen): weiße Fläche, dünne graue Kontur, Namen
    district_subset.plot(ax=ax, facecolor="white", edgecolor="#888888", linewidth=0.5, zorder=1)
    for _, row in district_subset.iterrows():
        centroid = row.geometry.centroid
        if centroid.is_empty:
            continue
        ax.annotate(
            str(row["NAME_2"]), (centroid.x, centroid.y),
            fontsize=6.5, ha="center", va="center", color="#555555", zorder=2,
        )

    # Regionsgrenzen (beider Regionen): fett, schwarz
    combined_regions.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.4, zorder=3)

    # Flüsse: blau, Linienbreite nach Strahler-Ordnung
    if len(rivers) > 0:
        if "ORD_STRA" in rivers.columns:
            for order, group in rivers.groupby("ORD_STRA"):
                lw = 0.4 + 0.25 * float(order)
                group.plot(ax=ax, color="#3182bd", linewidth=lw, zorder=4)
        else:
            rivers.plot(ax=ax, color="#3182bd", linewidth=0.8, zorder=4)

    # Mining-Vorhersage: rote Überlagerung pro Region (unterschiedliche Reprojektions-
    # Fenster je Region -> je Region einzeln zeichnen, gleicher zorder)
    for layer in region_layers:
        mining_mask, extent = layer["mask"], layer["extent"]
        rgba = np.zeros((*mining_mask.shape, 4))
        rgba[..., 0] = 1
        rgba[..., 3] = np.where(mining_mask > 0.5, OVERLAY_ALPHA, 0.0)
        ax.imshow(rgba, extent=extent, origin="upper", interpolation="nearest", zorder=5)

    # Hotspot-Cluster (beider Regionen): orangene Bounding Boxes (nur wenn aktiviert)
    if SHOW_HOTSPOT_BOXES:
        for layer in region_layers:
            for bminx, bminy, bmaxx, bmaxy, total in layer["boxes"]:
                rect = Rectangle(
                    (bminx, bminy), bmaxx - bminx, bmaxy - bminy,
                    linewidth=1.5, edgecolor="#e6550d", facecolor="none", zorder=6,
                )
                ax.add_patch(rect)

    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_title(
        f"Bono & Bono East \u2013 Galamsey-Hotspots (Fine-Tuned-Modell, P(Mining) \u2265 "
        f"{CONFIDENCE_THRESHOLD:.0%}, Sieve-gefiltert)",
        fontsize=13,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.4, label="Regionsgrenze"),
        Line2D([0], [0], color="#888888", linewidth=0.5, label="Distriktgrenze"),
        Line2D([0], [0], color="#3182bd", linewidth=1.2, label="Fluss (HydroRIVERS)"),
        Patch(facecolor="red", alpha=OVERLAY_ALPHA,
              label=f"Modell-Vorhersage: Mining (P \u2265 {CONFIDENCE_THRESHOLD:.0%})"),
    ]
    if SHOW_HOTSPOT_BOXES:
        legend_handles.append(
            Patch(facecolor="none", edgecolor="#e6550d", linewidth=1.5, label="Hotspot-Cluster")
        )
    ax.legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=8, framealpha=0.9, borderaxespad=0.0,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Kombinierte Karte gespeichert: {out_path} ({total_boxes} Hotspot-Cluster insgesamt).")


def main():
    if not os.path.exists(PRED_PROB):
        print(
            "Vorhersage-Raster nicht gefunden. Bitte zuerst 05_inference_bono_full.py "
            "ausführen (-> data/inference_bono_full_ft/)."
        )
        return

    regions_gdf, districts_gdf = get_boundaries()

    suffix = _threshold_suffix()
    print(f"CONFIDENCE_THRESHOLD={CONFIDENCE_THRESHOLD:.0%}  SHOW_HOTSPOT_BOXES={SHOW_HOTSPOT_BOXES}"
          f"  (Dateisuffix: '{suffix}')")

    out_paths = {
        "Bono": os.path.join(OUT_DIR, f"regional_map_bono{suffix}.png"),
        "Bono East": os.path.join(OUT_DIR, f"regional_map_bono_east{suffix}.png"),
    }
    for region_name in REGIONS:
        build_regional_map(region_name, regions_gdf, districts_gdf, out_paths[region_name])

    combined_out_path = os.path.join(OUT_DIR, f"regional_map_bono_combined{suffix}.png")
    build_combined_map(regions_gdf, districts_gdf, combined_out_path)


if __name__ == "__main__":
    main()
