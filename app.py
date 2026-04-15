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


# --------------------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------------------
DATA_SOURCE_MOCK = "Mock (demo data)"
DATA_SOURCE_LIVE = "Live (OSM + DOSM)"
DATA_SOURCE_CUSTOM = "Custom CSV + GeoJSON"


@st.cache_data(show_spinner="Loading data...")
def load_all(data_source: str,
             pharmacy_csv: str | None = None,
             geojson_path: str | None = None):
    """Returns (pharmacies_df, population_df, districts_geojson).

    `data_source` routes to one of three pipelines:
      * Mock — bundled sample data (no network).
      * Live — OpenStreetMap Overpass + DOSM population + geoBoundaries ADM2.
      * Custom — user-supplied NPRA CSV + district GeoJSON, DOSM population via API.
    """
    if data_source == DATA_SOURCE_MOCK:
        pharmacies = generate_mock_pharmacies()
        population = get_districts_df()[["district", "state", "population", "strata"]]
        geojson = generate_mock_geojson()
        return pharmacies, population, geojson

    if data_source == DATA_SOURCE_LIVE:
        pharmacies = fetch_pharmacies_from_osm()
        population = load_population_district_dosm()
        geojson = load_malaysia_districts_geojson()
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
    [DATA_SOURCE_LIVE, DATA_SOURCE_MOCK, DATA_SOURCE_CUSTOM],
    index=0,
    help=(
        "Live: OpenStreetMap pharmacies + DOSM population + geoBoundaries ADM2 "
        "(no API key; cached under ./data/).  "
        "Mock: bundled sample data, runs offline.  "
        "Custom: point to your own NPRA CSV and district GeoJSON."
    ),
)

if data_source == DATA_SOURCE_LIVE:
    st.sidebar.info(
        "ℹ️ OSM `amenity=pharmacy` covers most Malaysian retail pharmacies but "
        "under-counts the NPRA registry. Treat counts as directional until an "
        "authoritative KKM/NPRA feed is wired in via the Custom path."
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

m = folium.Map(location=MALAYSIA_CENTER, zoom_start=DEFAULT_ZOOM,
               tiles="cartodbpositron", control_scale=True)

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

# Pharmacy markers (clustered for performance)
cluster = MarkerCluster(name="Pharmacies").add_to(m)
for _, row in pharmacies_f.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=3, weight=1, color="#1f4e79", fill=True, fill_opacity=0.8,
        popup=folium.Popup(
            f"<b>{row['name']}</b><br>"
            f"License: {row.get('license_no','—')}<br>"
            f"District: {row.get('district','—')}<br>"
            f"State: {row.get('state','—')}",
            max_width=260,
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
