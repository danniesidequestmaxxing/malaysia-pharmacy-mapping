"""
Malaysia Pharmacy Mapping Dashboard
-----------------------------------
Run with:
    streamlit run app.py

Toggle the "Use mock data" switch in the sidebar OFF once you've wired in your
real CSV / data.gov.my datasets in `data_pipeline.py`.
"""

from __future__ import annotations
import json
from pathlib import Path

import folium
from folium.plugins import MarkerCluster
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from mock_data import (
    get_districts_df,
    generate_mock_pharmacies,
    generate_mock_geojson,
)
from data_pipeline import (
    spatial_join_pharmacies_to_districts,
    compute_district_metrics,
    enrich_geojson_with_metrics,
    load_pharmacies_from_csv,
    load_population_from_api,
    load_district_geojson,
    # Live-data loaders (no API key required)
    fetch_pharmacies_from_osm,
    load_population_district_dosm,
    load_malaysia_districts_geojson,
    normalize_district_names,
    normalize_geojson_names,
)
from local_sources import (
    parse_kmz,
    merge_pharmacy_sources,
    compute_worldpop_per_district,
)

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Malaysia Pharmacy Mapping",
    page_icon="💊",
    layout="wide",
)

MALAYSIA_CENTER = [4.2105, 109.4053]
DEFAULT_ZOOM = 6

# Basemap providers. The Google entries use Google's public tile infrastructure
# (the same `mt{0-3}.google.com/vt/` endpoints their own Maps site serves).
# Strict reading of Google Maps ToS only licenses these URLs for use inside
# Google's own products — use the Carto/OSM fallbacks for anything public.
# For a ToS-compliant Google deploy, switch to the Map Tiles API (session
# token flow); that's a larger rework, see the plan notes in the repo.
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


# --------------------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------------------
DATA_SOURCE_LOCAL = "Local (NPRA PDF + KMZ + WorldPop)"
DATA_SOURCE_LIVE = "Live (OSM + DOSM)"
DATA_SOURCE_MOCK = "Mock (demo data)"
DATA_SOURCE_CUSTOM = "Custom CSV + GeoJSON"

# Paths for the Local pipeline — committed processed artifacts so the deploy
# does not need access to the 17 GB WorldPop TIF or the raw NPRA PDF.
LOCAL_KMZ_PATH = "data/source/Project Pharma.kmz"
LOCAL_NPRA_GEOCODED_CSV = "data/pharmacies_npra_geocoded.csv"
LOCAL_WORLDPOP_PER_DISTRICT = "data/worldpop_per_district.csv"
LOCAL_WORLDPOP_RAW_CSV = "/Users/evoverebitda/Downloads/mys_general_2020.csv"  # only needed for refresh


@st.cache_data(show_spinner="Loading data...")
def load_all(data_source: str,
             pharmacy_csv: str | None = None,
             geojson_path: str | None = None):
    """Returns (pharmacies_df, population_df, districts_geojson).

    `data_source` routes to one of four pipelines:
      * Local — KMZ placemarks + NPRA-geocoded PDF + WorldPop per-district.
      * Live — OpenStreetMap Overpass + DOSM population + geoBoundaries ADM2.
      * Mock — bundled sample data (no network).
      * Custom — user-supplied NPRA CSV + district GeoJSON, DOSM population via API.
    """
    if data_source == DATA_SOURCE_MOCK:
        pharmacies = generate_mock_pharmacies()
        population = get_districts_df()[["district", "state", "population", "strata"]]
        geojson = generate_mock_geojson()
        return pharmacies, population, geojson

    # All three real-data modes reuse the geoBoundaries ADM2 polygons.
    geojson = load_malaysia_districts_geojson()

    if data_source == DATA_SOURCE_LOCAL:
        # Pharmacies: KMZ ∪ NPRA-geocoded (dedup on name+rounded-coords).
        sources = []
        kmz_path = Path(LOCAL_KMZ_PATH)
        if kmz_path.exists():
            sources.append(parse_kmz(kmz_path))
        npra_path = Path(LOCAL_NPRA_GEOCODED_CSV)
        if npra_path.exists():
            npra = pd.read_csv(npra_path)
            npra["source"] = npra.get("source", "NPRA").fillna("NPRA")
            npra["brand"] = npra.get("brand", "NPRA").fillna("NPRA")
            sources.append(npra)
        if not sources:
            st.error(
                "No local pharmacy sources found. Place 'Project Pharma.kmz' in "
                f"{LOCAL_KMZ_PATH} and/or run `python geocode_npra.py` to produce "
                f"{LOCAL_NPRA_GEOCODED_CSV}."
            )
            st.stop()
        pharmacies = merge_pharmacy_sources(*sources)

        # Population: WorldPop aggregated per ADM2 (cached CSV; falls back to raw).
        pop_cache = Path(LOCAL_WORLDPOP_PER_DISTRICT)
        if pop_cache.exists():
            population = pd.read_csv(pop_cache)
        else:
            population = compute_worldpop_per_district(
                csv_path=LOCAL_WORLDPOP_RAW_CSV,
                districts_geojson=geojson,
                cache_path=LOCAL_WORLDPOP_PER_DISTRICT,
            )
    elif data_source == DATA_SOURCE_LIVE:
        pharmacies = fetch_pharmacies_from_osm()
        population = load_population_district_dosm()
    else:  # Custom
        pharmacies = load_pharmacies_from_csv(pharmacy_csv)
        population = load_population_from_api()
        geojson = load_district_geojson(geojson_path)

    # Canonicalize spellings on BOTH sides of the merge so districts don't
    # drop to NaN because one source says "Pulau Pinang" and the other "Penang".
    population = normalize_district_names(population)
    geojson = normalize_geojson_names(geojson)
    return pharmacies, population, geojson


@st.cache_data(show_spinner="Computing district metrics...")
def build_metrics(pharmacies: pd.DataFrame, population: pd.DataFrame, geojson: dict):
    pharmacies_joined = spatial_join_pharmacies_to_districts(pharmacies, geojson)
    metrics = compute_district_metrics(pharmacies_joined, population)
    enriched_geojson = enrich_geojson_with_metrics(geojson, metrics)
    return pharmacies_joined, metrics, enriched_geojson


# --------------------------------------------------------------------------------------
# Sidebar — data source + filters
# --------------------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

data_source = st.sidebar.radio(
    "Data source",
    [DATA_SOURCE_LOCAL, DATA_SOURCE_LIVE, DATA_SOURCE_MOCK, DATA_SOURCE_CUSTOM],
    index=0,
    help=(
        "Local: KMZ chain-map + geocoded NPRA PDF + WorldPop per-district.  "
        "Live: OpenStreetMap + DOSM + geoBoundaries (no API key; cached in ./data/).  "
        "Mock: bundled sample data, runs offline.  "
        "Custom: point to your own NPRA CSV and district GeoJSON."
    ),
)

if data_source == DATA_SOURCE_LOCAL:
    st.sidebar.success(
        "✅ Using authoritative sources: NPRA Senarai Premis (594 pharmacies) "
        "+ Project Pharma chain KMZ (957 pharmacies) + WorldPop 2020 population."
    )
elif data_source == DATA_SOURCE_LIVE:
    st.sidebar.info(
        "ℹ️ OSM `amenity=pharmacy` covers most Malaysian retail pharmacies but "
        "under-counts the NPRA registry. Treat counts as directional."
    )

pharmacy_csv = geojson_path = None
if data_source == DATA_SOURCE_CUSTOM:
    pharmacy_csv = st.sidebar.text_input("Pharmacy CSV path", "data/pharmacies.csv")
    geojson_path = st.sidebar.text_input("District GeoJSON path/URL", "data/districts.geojson")

pharmacies, population, geojson = load_all(data_source, pharmacy_csv, geojson_path)
pharmacies_joined, metrics, enriched_geojson = build_metrics(pharmacies, population, geojson)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filters")

# State filter
all_states = sorted(metrics["state"].dropna().unique())
selected_states = st.sidebar.multiselect("State (Negeri)", all_states, default=all_states)

# District filter — cascades from states
districts_in_states = sorted(
    metrics.loc[metrics["state"].isin(selected_states), "district"].dropna().unique()
)
selected_districts = st.sidebar.multiselect(
    "District (Daerah)", districts_in_states, default=districts_in_states
)

# Strata filter
strata_options = sorted(metrics["strata"].dropna().unique()) if "strata" in metrics.columns else []
selected_strata = st.sidebar.multiselect(
    "Strata (Urban/Rural)", strata_options, default=strata_options
) if strata_options else None

# Brand filter (only meaningful when the pharmacy DF has brand labels,
# i.e. in Local mode where KMZ tags chain colours).
brand_options = (
    sorted(pharmacies_joined["brand"].dropna().unique())
    if "brand" in pharmacies_joined.columns else []
)
selected_brands = (
    st.sidebar.multiselect("Brand / Chain", brand_options, default=brand_options)
    if brand_options else None
)

# Choropleth metric chooser
metric_choice = st.sidebar.radio(
    "Choropleth metric",
    ["pop_per_pharmacy", "pharmacies_per_100k", "population"],
    format_func=lambda x: {
        "pop_per_pharmacy": "Population per Pharmacy (lower = better access)",
        "pharmacies_per_100k": "Pharmacies per 100k people",
        "population": "Total Population",
    }[x],
)

# Basemap (tile layer) chooser — Google Maps by default; see MAP_TILE_PROVIDERS.
basemap_name = st.sidebar.selectbox(
    "Basemap",
    list(MAP_TILE_PROVIDERS),
    index=0,
    help=("Google tiles come from Google's public tile servers — great look, "
          "but outside strict ToS for non-Google products. Switch to CartoDB "
          "Positron for a compliant, license-clean basemap."),
)

# --------------------------------------------------------------------------------------
# Apply filters
# --------------------------------------------------------------------------------------
mask = metrics["state"].isin(selected_states) & metrics["district"].isin(selected_districts)
if selected_strata:
    mask &= metrics["strata"].isin(selected_strata)
metrics_f = metrics[mask].copy()

p_mask = pharmacies_joined["state"].isin(selected_states) & \
         pharmacies_joined["district"].isin(selected_districts)
if selected_strata and "strata" in pharmacies_joined.columns:
    p_mask &= pharmacies_joined["strata"].isin(selected_strata)
if selected_brands and "brand" in pharmacies_joined.columns:
    p_mask &= pharmacies_joined["brand"].isin(selected_brands)
pharmacies_f = pharmacies_joined[p_mask].copy()

# Filter the GeoJSON features as well so the choropleth respects filters
allowed_districts = set(metrics_f["district"])
filtered_geojson = {
    "type": "FeatureCollection",
    "features": [
        f for f in enriched_geojson["features"]
        if f["properties"].get("district") in allowed_districts
    ],
}

# --------------------------------------------------------------------------------------
# Header + KPIs
# --------------------------------------------------------------------------------------
st.title("💊 Malaysia Pharmacy Mapping Dashboard")
st.caption("Pharmacy access analytics across Malaysian daerah — built with Streamlit + Folium.")

c1, c2, c3, c4 = st.columns(4)
total_pop = int(metrics_f["population"].sum())
total_phar = int(metrics_f["pharmacy_count"].sum())
overall_ratio = total_pop / total_phar if total_phar else float("nan")
underserved = int((metrics_f["pop_per_pharmacy"] > 10_000).sum())

c1.metric("Districts in view", f"{len(metrics_f):,}")
c2.metric("Total population", f"{total_pop:,}")
c3.metric("Total pharmacies", f"{total_phar:,}")
c4.metric("Overall ratio",
          "N/A" if pd.isna(overall_ratio) else f"1 : {int(overall_ratio):,}",
          delta=f"{underserved} districts >10k:1", delta_color="inverse")

# --------------------------------------------------------------------------------------
# Map
# --------------------------------------------------------------------------------------
st.subheader("🗺️ Map view")

# Initialize the map with the basemap chosen in the sidebar.
_provider = MAP_TILE_PROVIDERS[basemap_name]
if "builtin" in _provider:
    m = folium.Map(location=MALAYSIA_CENTER, zoom_start=DEFAULT_ZOOM,
                   tiles=_provider["builtin"], control_scale=True)
else:
    m = folium.Map(location=MALAYSIA_CENTER, zoom_start=DEFAULT_ZOOM,
                   tiles=None, control_scale=True)
    folium.TileLayer(
        tiles=_provider["url"],
        attr=_provider["attr"],
        name=basemap_name,
        subdomains=_provider["subdomains"],
        max_zoom=_provider["max_zoom"],
        overlay=False,
        control=True,
    ).add_to(m)

# Choropleth layer
folium.Choropleth(
    geo_data=filtered_geojson,
    data=metrics_f,
    columns=["district", metric_choice],
    key_on="feature.properties.district",
    fill_color="YlOrRd" if metric_choice == "pop_per_pharmacy" else "YlGnBu",
    fill_opacity=0.7,
    line_opacity=0.3,
    nan_fill_color="lightgray",
    legend_name={
        "pop_per_pharmacy": "Population per Pharmacy",
        "pharmacies_per_100k": "Pharmacies per 100k",
        "population": "Population",
    }[metric_choice],
    name="Choropleth",
).add_to(m)

# Transparent overlay carrying the rich tooltip (Choropleth's own tooltip is limited)
folium.GeoJson(
    filtered_geojson,
    name="District info",
    style_function=lambda _: {"fillOpacity": 0, "color": "transparent", "weight": 0},
    highlight_function=lambda _: {"weight": 2, "color": "#333", "fillOpacity": 0.1},
    tooltip=folium.GeoJsonTooltip(
        fields=["district", "state", "population", "pharmacy_count",
                "pop_per_pharmacy", "pharmacies_per_100k"],
        aliases=["District:", "State:", "Population:", "Pharmacies:",
                 "Pop per Pharmacy:", "Per 100k:"],
        localize=True, sticky=True,
        style=("background-color: white; color: #222; "
               "font-family: arial; font-size: 12px; padding: 6px;"),
    ),
).add_to(m)

# Pharmacy markers (clustered for performance). Colour by brand when we have
# that signal (Local mode), fall back to a single navy marker otherwise.
BRAND_COLORS = {
    "CARiNG":      "#2e7d32",  # green
    "Guardian":    "#1a237e",  # dark blue
    "Watsons":     "#c62828",  # red
    "Alpro":       "#6a1b9a",  # purple
    "Independent": "#455a64",  # slate
    "NPRA":        "#ef6c00",  # orange (NPRA-only, no KMZ match)
    "Other":       "#1f4e79",
}

cluster = MarkerCluster(name="Pharmacies").add_to(m)
for _, row in pharmacies_f.iterrows():
    brand = row.get("brand", "Other") or "Other"
    color = BRAND_COLORS.get(brand, "#1f4e79")
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=3, weight=1, color=color, fill=True, fill_color=color, fill_opacity=0.85,
        popup=folium.Popup(
            f"<b>{row['name']}</b><br>"
            f"Brand: {brand}<br>"
            f"Source: {row.get('source','—')}<br>"
            f"District: {row.get('district','—')}<br>"
            f"State: {row.get('state','—')}",
            max_width=280,
        ),
    ).add_to(cluster)

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, height=620, use_container_width=True, returned_objects=[])

# --------------------------------------------------------------------------------------
# Charts + table
# --------------------------------------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("📊 Districts by Population per Pharmacy (worst access on top)")
    chart_df = (metrics_f.dropna(subset=["pop_per_pharmacy"])
                         .sort_values("pop_per_pharmacy", ascending=False)
                         .head(20))
    fig = px.bar(
        chart_df, x="pop_per_pharmacy", y="district", color="state",
        orientation="h", height=520,
        labels={"pop_per_pharmacy": "Population per Pharmacy", "district": ""},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("📋 District metrics")
    st.dataframe(
        metrics_f.sort_values("pop_per_pharmacy", ascending=False)
                 .reset_index(drop=True)
                 .style.format({
                     "population": "{:,.0f}",
                     "pharmacy_count": "{:,.0f}",
                     "pop_per_pharmacy": "{:,.0f}",
                     "pharmacies_per_100k": "{:.2f}",
                 }),
        use_container_width=True, height=520,
    )

with st.expander("ℹ️ How the ratio is computed"):
    st.markdown(
        "- Each pharmacy lat/lon is spatially joined to a district polygon "
        "(point-in-polygon with an STRtree index).\n"
        "- Pharmacies per district are counted, then merged onto the DOSM "
        "population table by `(district, state)`.\n"
        "- **Population per Pharmacy** = `population / pharmacy_count`. "
        "Districts with zero pharmacies show as N/A."
    )
