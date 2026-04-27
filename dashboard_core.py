"""
dashboard_core.py
-----------------
Shared constants, cached loaders, and helpers used by every Streamlit
page (the main `app.py` and each file under `pages/`).

Importing this module does NOT render any UI — it only exposes callables
and constants.  The `@st.cache_data` decorators are safe to apply at
import time because they return cached-wrapper functions; they don't run
until someone calls them.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from mock_data import (
    get_districts_df,
    generate_mock_pharmacies,
    generate_mock_geojson,
)
from data_pipeline import (
    load_malaysia_districts_geojson,
    load_malaysia_mukim_geojson,
    load_kkm_zones_geojson,
    build_voronoi_catchments,
    normalize_district_names,
    normalize_geojson_names,
    load_pharmacies_from_csv,
    load_population_from_api,
    load_district_geojson,
    load_population_district_dosm,
    fetch_pharmacies_from_osm,
)
from local_sources import (
    parse_kmz,
    parse_scraped_store_csv,
    merge_pharmacy_sources,
    compute_worldpop_per_polygons,
)


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

MALAYSIA_CENTER = [4.2105, 109.4053]
DEFAULT_ZOOM = 6

# Basemap tile providers. Google options hit Google's public tile servers;
# strict ToS reading says those URLs are only licensed inside Google's own
# products.  Fall back to Carto/OSM for a license-clean basemap.
MAP_TILE_PROVIDERS: dict = {
    "Google Maps — Roadmap":   {"url": "https://mt{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
                                "attr": "© Google", "max_zoom": 20, "subdomains": "0123"},
    "Google Maps — Satellite": {"url": "https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                                "attr": "© Google", "max_zoom": 20, "subdomains": "0123"},
    "Google Maps — Hybrid":    {"url": "https://mt{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
                                "attr": "© Google", "max_zoom": 20, "subdomains": "0123"},
    "Google Maps — Terrain":   {"url": "https://mt{s}.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
                                "attr": "© Google", "max_zoom": 20, "subdomains": "0123"},
    "CartoDB Positron":        {"builtin": "cartodbpositron"},
    "OpenStreetMap":           {"builtin": "openstreetmap"},
}

# Folium's built-in Marker-icon palette is a fixed set of named colours:
#   red, darkred, lightred, orange, beige, green, darkgreen, lightgreen,
#   blue, darkblue, cadetblue, lightblue, purple, darkpurple, pink,
#   gray, lightgray, black, white.
# BRAND_COLORS below are hex (used on CircleMarker and the Plotly bar chart);
# BRAND_ICON_COLORS maps each chain to the closest member of the palette
# above so that the classic Google-Maps-style pin marker still renders
# recognisable per-brand colours.
BRAND_ICON_COLORS = {
    "Caring":            "darkgreen",
    "Alpro":             "darkpurple",
    "BIG Pharmacy":      "red",
    "Healthlane":        "cadetblue",
    "Sunway Multicare":  "darkblue",
    "AA Pharmacy":       "orange",
    "Georgetown":        "green",
    "Wellings":          "darkgreen",
    "Straits":           "cadetblue",
    "Guardian":          "darkblue",
    "Guardian Retail":   "lightblue",
    "Watsons":           "red",
    "PMG":               "darkgreen",
    "AM PM":             "darkred",
    "Nazen":             "beige",
    "Siang":             "orange",
    "Alliance":          "purple",
    "Constant":          "purple",
    "Be Pharmacy":       "blue",
    "MediQ":             "darkgreen",
    "Rx":                "blue",
    "Mega Kulim":        "beige",
    "Rejoice":           "darkred",
    "Farmasi":           "cadetblue",
    "KS":                "gray",
    "NPRA":              "orange",
    "Independent":       "gray",
    "Other":             "darkblue",
}

# Pharmacy chain → marker colour.  Kept here so pages stay visually consistent.
BRAND_COLORS = {
    "Caring":            "#2e7d32",
    "Alpro":             "#6a1b9a",
    "BIG Pharmacy":      "#c62828",
    "Healthlane":        "#546e7a",
    "Sunway Multicare":  "#1a237e",
    "AA Pharmacy":       "#f9a825",
    "Georgetown":        "#66bb6a",
    "Wellings":          "#388e3c",
    "Straits":           "#78909c",
    "Guardian":          "#0d47a1",
    "Guardian Retail":   "#64b5f6",
    "Watsons":           "#d32f2f",
    "PMG":               "#00695c",
    "AM PM":             "#ad1457",
    "Nazen":             "#5d4037",
    "Siang":             "#e65100",
    "Alliance":          "#4527a0",
    "Constant":          "#7b1fa2",
    "Be Pharmacy":       "#0277bd",
    "MediQ":             "#2e7d32",
    "Rx":                "#1565c0",
    "Mega Kulim":        "#6d4c41",
    "Rejoice":           "#bf360c",
    "Farmasi":           "#455a64",
    "KS":                "#37474f",
    "NPRA":              "#ef6c00",
    "Independent":       "#455a64",
    "Other":             "#1f4e79",
}

# Data-source identifiers (used by the main app's sidebar radio).
DATA_SOURCE_LOCAL = "Local (NPRA PDF + KMZ + WorldPop)"
DATA_SOURCE_LIVE = "Live (OSM + DOSM)"
DATA_SOURCE_MOCK = "Mock (demo data)"
DATA_SOURCE_CUSTOM = "Custom CSV + GeoJSON"

# Geography granularities (used by every page).
GEO_DISTRICT = "District (Daerah) — 159"
GEO_MUKIM = "Mukim (Sub-district) — 2,000"
GEO_ZONE = "KKM Zone — 5"
GEO_CATCHMENT = "Pharmacy Catchment (Voronoi)"

# Committed data artefacts.
LOCAL_KMZ_PATH = "data/source/Project Pharma.kmz"
LOCAL_NPRA_GEOCODED_CSV = "data/pharmacies_npra_geocoded.csv"
LOCAL_PMG_GEOCODED_CSV = "data/pharmacies_pmg_geocoded.csv"
LOCAL_WATSONS_CSV = "data/pharmacies_watsons.csv"
LOCAL_GUARDIAN_CSV = "data/pharmacies_guardian.csv"
LOCAL_WORLDPOP_PER_DISTRICT = "data/worldpop_per_district.csv"
LOCAL_WORLDPOP_PER_MUKIM = "data/worldpop_per_mukim.csv"
LOCAL_WORLDPOP_PER_ZONE = "data/worldpop_per_zone.csv"
LOCAL_WORLDPOP_RAW_CSV = "/Users/evoverebitda/Downloads/mys_general_2020.csv"


# --------------------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading pharmacies...")
def load_pharmacies(data_source: str,
                    pharmacy_csv: str | None = None) -> pd.DataFrame:
    """Pharmacies DataFrame for the chosen data source. Geography-agnostic."""
    if data_source == DATA_SOURCE_MOCK:
        return generate_mock_pharmacies()
    if data_source == DATA_SOURCE_LOCAL:
        sources = []
        if Path(LOCAL_KMZ_PATH).exists():
            sources.append(parse_kmz(LOCAL_KMZ_PATH))
        if Path(LOCAL_NPRA_GEOCODED_CSV).exists():
            npra = pd.read_csv(LOCAL_NPRA_GEOCODED_CSV)
            npra["source"] = npra.get("source", "NPRA").fillna("NPRA")
            npra["brand"] = npra.get("brand", "NPRA").fillna("NPRA")
            sources.append(npra)
        if Path(LOCAL_PMG_GEOCODED_CSV).exists():
            pmg = pd.read_csv(LOCAL_PMG_GEOCODED_CSV)
            pmg["source"] = pmg.get("source", "PMG").fillna("PMG")
            pmg["brand"] = pmg.get("brand", "PMG").fillna("PMG")
            sources.append(pmg)
        if Path(LOCAL_WATSONS_CSV).exists():
            sources.append(parse_scraped_store_csv(
                LOCAL_WATSONS_CSV, source_label="Watsons-Web", default_brand="Watsons"))
        if Path(LOCAL_GUARDIAN_CSV).exists():
            sources.append(parse_scraped_store_csv(
                LOCAL_GUARDIAN_CSV,
                source_label="Guardian-Web",
                default_brand="Guardian",
                pharmacy_only=False,
                retail_brand_suffix=" Retail",
            ))
        if not sources:
            st.error(
                f"No local pharmacy sources found — place the KMZ at {LOCAL_KMZ_PATH}, "
                "and run `python geocode_npra.py` / `python geocode_pmg.py` / "
                "`python geocode_watsons.py`."
            )
            st.stop()
        return merge_pharmacy_sources(*sources)
    if data_source == DATA_SOURCE_LIVE:
        return fetch_pharmacies_from_osm()
    return load_pharmacies_from_csv(pharmacy_csv)


@st.cache_data(show_spinner="Loading population & boundaries...")
def load_geography(
    data_source: str,
    geography: str,
    pharmacies_for_catchment: Optional[pd.DataFrame] = None,
    custom_geojson_path: str | None = None,
):
    """Return the geography-specific context dict:
        geojson, population, join_keys, label_key, base_geojson
    """
    base_geojson = normalize_geojson_names(load_malaysia_districts_geojson())

    if data_source == DATA_SOURCE_MOCK:
        pop = get_districts_df()[["district", "state", "population", "strata"]]
        return {
            "geojson": generate_mock_geojson(),
            "population": pop,
            "join_keys": ["district", "state"],
            "label_key": "district",
            "base_geojson": generate_mock_geojson(),
        }

    if geography == GEO_DISTRICT:
        geojson = base_geojson
        if data_source == DATA_SOURCE_LOCAL and Path(LOCAL_WORLDPOP_PER_DISTRICT).exists():
            pop = pd.read_csv(LOCAL_WORLDPOP_PER_DISTRICT)
        elif data_source == DATA_SOURCE_CUSTOM and custom_geojson_path:
            geojson = normalize_geojson_names(load_district_geojson(custom_geojson_path))
            pop = load_population_from_api()
        else:
            pop = load_population_district_dosm()
        return {
            "geojson": geojson,
            "population": normalize_district_names(pop),
            "join_keys": ["district", "state"],
            "label_key": "district",
            "base_geojson": base_geojson,
        }

    if geography == GEO_MUKIM:
        geojson = normalize_geojson_names(load_malaysia_mukim_geojson())
        if Path(LOCAL_WORLDPOP_PER_MUKIM).exists():
            pop = pd.read_csv(LOCAL_WORLDPOP_PER_MUKIM)
        else:
            pop = compute_worldpop_per_polygons(
                csv_path=LOCAL_WORLDPOP_RAW_CSV,
                polygons_geojson=geojson,
                cache_path=LOCAL_WORLDPOP_PER_MUKIM,
                id_properties=["mukim", "district", "state"],
            )
        pop = normalize_district_names(pop)
        return {
            "geojson": geojson,
            "population": pop,
            "join_keys": ["mukim", "district", "state"],
            "label_key": "mukim",
            "base_geojson": base_geojson,
        }

    if geography == GEO_ZONE:
        geojson = load_kkm_zones_geojson()
        if Path(LOCAL_WORLDPOP_PER_ZONE).exists():
            pop = pd.read_csv(LOCAL_WORLDPOP_PER_ZONE)
        else:
            pop = compute_worldpop_per_polygons(
                csv_path=LOCAL_WORLDPOP_RAW_CSV,
                polygons_geojson=geojson,
                cache_path=LOCAL_WORLDPOP_PER_ZONE,
                id_properties=["zone"],
            )
        return {
            "geojson": geojson,
            "population": pop,
            "join_keys": ["zone"],
            "label_key": "zone",
            "base_geojson": base_geojson,
        }

    # GEO_CATCHMENT
    if pharmacies_for_catchment is None or len(pharmacies_for_catchment) < 3:
        st.error("Catchment geography needs at least 3 pharmacies.")
        st.stop()
    voronoi = build_voronoi_catchments(
        pharmacies_for_catchment[["pharmacy_id", "name", "brand",
                                  "latitude", "longitude"]].dropna(
            subset=["latitude", "longitude"]),
        boundary_geojson=base_geojson,
    )
    cache_name = f"data/worldpop_per_catchment_{len(voronoi['features'])}.csv"
    if Path(LOCAL_WORLDPOP_RAW_CSV).exists():
        pop = compute_worldpop_per_polygons(
            csv_path=LOCAL_WORLDPOP_RAW_CSV,
            polygons_geojson=voronoi,
            cache_path=cache_name,
            id_properties=["catchment_id", "pharmacy", "brand"],
        )
    else:
        pop = pd.DataFrame({
            "catchment_id": [f["properties"]["catchment_id"] for f in voronoi["features"]],
            "pharmacy": [f["properties"]["pharmacy"] for f in voronoi["features"]],
            "brand": [f["properties"]["brand"] for f in voronoi["features"]],
            "population": 0,
        })
    return {
        "geojson": voronoi,
        "population": pop,
        "join_keys": ["catchment_id"],
        "label_key": "pharmacy",
        "base_geojson": base_geojson,
    }


@st.cache_data(show_spinner="Computing metrics...")
def build_metrics(pharmacies: pd.DataFrame, geo_ctx: dict):
    """Spatial-join pharmacies onto the chosen geography, compute metrics,
    and enrich the GeoJSON so tooltips read the metrics directly."""
    from data_pipeline import (
        stamp_polygon_props, compute_polygon_metrics,
        enrich_geojson_with_polygon_metrics,
    )
    with_base = stamp_polygon_props(
        pharmacies, geo_ctx["base_geojson"], ["district", "state"]
    )
    extra_keys = [k for k in geo_ctx["join_keys"] if k not in ("district", "state")]
    if extra_keys:
        with_all = stamp_polygon_props(with_base, geo_ctx["geojson"], extra_keys)
    else:
        with_all = with_base
    metrics = compute_polygon_metrics(
        with_all, geo_ctx["population"], on=geo_ctx["join_keys"]
    )
    enriched = enrich_geojson_with_polygon_metrics(
        geo_ctx["geojson"], metrics, on=geo_ctx["join_keys"]
    )
    return with_all, metrics, enriched


def rebuild_metrics_from_joined_pharmacies(
    pharmacies_with_keys: pd.DataFrame,
    geo_ctx: dict,
):
    """Recompute metrics from an already-joined pharmacy frame.

    Use this after UI filters (for example brand/chain filters) so the
    choropleth, tooltips, and KPIs stay aligned with the visible markers
    without repeating the spatial joins from `build_metrics`.
    """
    from data_pipeline import (
        compute_polygon_metrics,
        enrich_geojson_with_polygon_metrics,
    )

    metrics = compute_polygon_metrics(
        pharmacies_with_keys,
        geo_ctx["population"],
        on=geo_ctx["join_keys"],
    )
    enriched = enrich_geojson_with_polygon_metrics(
        geo_ctx["geojson"],
        metrics,
        on=geo_ctx["join_keys"],
    )
    return metrics, enriched


# --------------------------------------------------------------------------------------
# Pharmacy markers — classic Leaflet/Google-Maps-style pin (teardrop)
# --------------------------------------------------------------------------------------

def make_pharmacy_marker(row, *, popup_extras: dict | None = None):
    """Build a folium.Marker with a coloured pin matching the row's brand.

    Uses folium.Icon (the stock Leaflet teardrop) coloured from the
    BRAND_ICON_COLORS palette, and a small 'plus' glyph so it reads as a
    health/pharmacy pin from a distance.
    """
    import folium
    brand = row.get("brand", "Other") or "Other"
    icon_color = BRAND_ICON_COLORS.get(brand, "darkblue")

    popup_html = (
        f"<b>{row['name']}</b><br>"
        f"Brand: {brand}<br>"
        f"District: {row.get('district', '—')}<br>"
        f"Source: {row.get('source', '—')}"
    )
    for k, v in (popup_extras or {}).items():
        popup_html += f"<br>{k}: {v}"

    return folium.Marker(
        location=[row["latitude"], row["longitude"]],
        tooltip=f"{row['name']} ({brand})",
        popup=folium.Popup(popup_html, max_width=300),
        icon=folium.Icon(color=icon_color, icon="plus", prefix="fa"),
    )


# --------------------------------------------------------------------------------------
# Choropleth binning — shared between pages
# --------------------------------------------------------------------------------------

def choropleth_bins(metric: str, series: pd.Series) -> list | None:
    """Policy-anchored threshold_scale per metric.  Last bin stretched to
    cover the max so Folium accepts every value; outliers pool into the
    top bucket without squashing the scale."""
    import numpy as np
    s = series.dropna()
    if s.empty:
        return None
    hi = float(s.max())

    def _cap(base: list, top_policy: float) -> list:
        if hi <= top_policy:
            return base
        return base + [float(np.ceil(hi))]

    if metric == "pop_per_pharmacy":
        base = [0, 3000, 5000, 10000, 20000, 50000]
        return _cap(base, 50000) if hi <= 50000 else base + [int(np.ceil(hi / 1000) * 1000)]
    if metric == "pop_per_pharmacy_5km":
        # 5 km neighborhood ratios sit an order of magnitude above the per-cell
        # version (e.g. 30k–120k typical). Spread the bin edges evenly across
        # that range so the legend ticks don't pile up at the low end.
        base = [0, 5000, 15000, 30000, 50000, 75000]
        return _cap(base, 75000) if hi <= 75000 else base + [int(np.ceil(hi / 5000) * 5000)]
    if metric in ("pharmacies_per_1000", "pharmacies_per_1000_5km"):
        return _cap([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0], 1.0)
    if metric == "pharmacies_per_100k":
        return _cap([0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0], 100.0)
    if metric == "population":
        qs = list(s.quantile([0.0, 0.2, 0.4, 0.6, 0.8, 0.95]).round().astype(int))
        return sorted(set(qs + [int(np.ceil(hi))]))
    return None
