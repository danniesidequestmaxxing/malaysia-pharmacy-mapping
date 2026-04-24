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
from typing import Optional

import folium
from folium.plugins import MarkerCluster
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from dashboard_core import (
    MALAYSIA_CENTER, DEFAULT_ZOOM, MAP_TILE_PROVIDERS, BRAND_COLORS,
    DATA_SOURCE_LOCAL, DATA_SOURCE_LIVE, DATA_SOURCE_MOCK, DATA_SOURCE_CUSTOM,
    GEO_DISTRICT, GEO_MUKIM, GEO_ZONE, GEO_CATCHMENT,
    load_pharmacies, load_geography, build_metrics, choropleth_bins,
    make_pharmacy_marker,
)

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Malaysia Pharmacy Mapping",
    page_icon="💊",
    layout="wide",
)


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
        "✅ Authoritative sources: Project Pharma chain KMZ (957) + NPRA "
        "Senarai Premis PDF (594) + PMG Outlets Excel (187) + WorldPop 2020."
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

# Geography (choropleth granularity) — independent of data source.
geography = st.sidebar.selectbox(
    "Geography (choropleth granularity)",
    [GEO_DISTRICT, GEO_MUKIM, GEO_ZONE, GEO_CATCHMENT],
    index=0,
    help=(
        "District: 159 Malaysian daerah (ADM2).  "
        "Mukim: 1,859 sub-districts (ADM3) — shows intra-district variation.  "
        "Zone: 5 KKM pharmacy zones.  "
        "Catchment: Voronoi polygon per pharmacy — each cell is the area "
        "closer to that pharmacy than to any other."
    ),
)

pharmacies = load_pharmacies(data_source, pharmacy_csv)
geo_ctx = load_geography(
    data_source, geography,
    pharmacies_for_catchment=pharmacies if geography == GEO_CATCHMENT else None,
    custom_geojson_path=geojson_path,
)
pharmacies_joined, metrics, enriched_geojson = build_metrics(pharmacies, geo_ctx)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filters")

# Build the full (state, district) universe from the base ADM2 layer so the
# filter lists include polygons with *zero* pharmacies (e.g. remote daerah) —
# otherwise you can't zoom into an underserved district because it's been
# filtered out of the widget.
_base_props = [f["properties"] for f in geo_ctx["base_geojson"]["features"]]
_all_state_district_pairs = pd.DataFrame(_base_props)
all_states = sorted(_all_state_district_pairs["state"].dropna().unique())
selected_states = st.sidebar.multiselect("State (Negeri)", all_states, default=all_states)

districts_in_states = sorted(
    _all_state_district_pairs.loc[
        _all_state_district_pairs["state"].isin(selected_states), "district"
    ].dropna().unique()
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

# Choropleth metric chooser. Higher-is-better metrics get a green scale;
# higher-is-worse (ratio) gets a red scale.
metric_choice = st.sidebar.radio(
    "Choropleth metric",
    ["pop_per_pharmacy", "pharmacies_per_1000", "pharmacies_per_100k", "population"],
    index=1,  # default to the user-requested "per 1,000"
    format_func=lambda x: {
        "pop_per_pharmacy": "Population per Pharmacy (lower = better access)",
        "pharmacies_per_1000": "Pharmacies per 1,000 residents (higher = better)",
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
# State / district filters apply to pharmacies in every geography (both
# columns are always stamped). For the choropleth metrics, only apply those
# columns if the current geography actually exposes them.
metric_mask = pd.Series(True, index=metrics.index)
if "state" in metrics.columns:
    metric_mask &= metrics["state"].isin(selected_states)
if "district" in metrics.columns:
    metric_mask &= metrics["district"].isin(selected_districts)
if selected_strata and "strata" in metrics.columns:
    metric_mask &= metrics["strata"].isin(selected_strata)
metrics_f = metrics[metric_mask].copy()

p_mask = pharmacies_joined["state"].isin(selected_states) & \
         pharmacies_joined["district"].isin(selected_districts)
if selected_strata and "strata" in pharmacies_joined.columns:
    p_mask &= pharmacies_joined["strata"].isin(selected_strata)
if selected_brands and "brand" in pharmacies_joined.columns:
    p_mask &= pharmacies_joined["brand"].isin(selected_brands)
pharmacies_f = pharmacies_joined[p_mask].copy()

# Filter the GeoJSON features using whichever key identifies the current
# geography (district / mukim / zone / catchment_id).
label_key = geo_ctx["label_key"]
allowed_districts = set(metrics_f[label_key].dropna()) if label_key in metrics_f.columns else set(metrics_f["district"].dropna())
filtered_geojson = {
    "type": "FeatureCollection",
    "features": [
        f for f in enriched_geojson["features"]
        if f["properties"].get(label_key) in allowed_districts
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

c1.metric(f"{geography.split(' — ')[0]} in view", f"{len(metrics_f):,}")
c2.metric("Total population", f"{total_pop:,}")
c3.metric("Total pharmacies", f"{total_phar:,}")
c4.metric("Overall ratio",
          "N/A" if pd.isna(overall_ratio) else f"1 : {int(overall_ratio):,}",
          delta=f"{underserved} polygons >10k:1", delta_color="inverse")

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

# Choropleth layer — key by the active geography's label column, not hard-coded to district.
# Use green-is-good for "per-1000/100k" metrics (higher = better access)
# and red-is-bad for the "pop_per_pharmacy" ratio (higher = worse access).
_fill_color = "YlOrRd" if metric_choice == "pop_per_pharmacy" else "YlGnBu"


_bins = choropleth_bins(metric_choice, metrics_f[metric_choice])
_choropleth_kwargs = {"threshold_scale": _bins} if _bins and len(_bins) >= 3 else {}

folium.Choropleth(
    geo_data=filtered_geojson,
    data=metrics_f,
    columns=[label_key, metric_choice],
    key_on=f"feature.properties.{label_key}",
    fill_color=_fill_color,
    fill_opacity=0.7,
    line_opacity=0.3,
    nan_fill_color="lightgray",
    legend_name={
        "pop_per_pharmacy": "Population per Pharmacy (lower = better)",
        "pharmacies_per_1000": "Pharmacies per 1,000 residents (higher = better)",
        "pharmacies_per_100k": "Pharmacies per 100k",
        "population": "Population",
    }[metric_choice],
    name="Choropleth",
    **_choropleth_kwargs,
).add_to(m)

# Transparent overlay carrying the rich tooltip (Choropleth's own tooltip is
# limited). Build the tooltip fields dynamically so each geography only shows
# properties that actually exist on its features.
_candidate_fields = [label_key]
if "district" not in _candidate_fields and "district" in geo_ctx["join_keys"]:
    _candidate_fields.append("district")
if "state" not in _candidate_fields and "state" in geo_ctx["join_keys"]:
    _candidate_fields.append("state")
_candidate_fields += ["population", "pharmacy_count", "pop_per_pharmacy", "pharmacies_per_1000"]
_field_aliases = {
    "mukim": "Mukim:", "district": "District:", "state": "State:",
    "zone": "Zone:", "catchment_id": "Catchment:", "pharmacy": "Pharmacy:",
    "population": "Population:", "pharmacy_count": "Pharmacies:",
    "pop_per_pharmacy": "Pop per Pharmacy:",
    "pharmacies_per_1000": "Pharmacies / 1,000:",
}
# Only keep fields that exist in the first feature's properties.
_sample_props = filtered_geojson["features"][0]["properties"] if filtered_geojson["features"] else {}
_tooltip_fields = [f for f in _candidate_fields if f in _sample_props]
folium.GeoJson(
    filtered_geojson,
    name=f"{geography.split(' — ')[0]} info",
    style_function=lambda _: {"fillOpacity": 0, "color": "transparent", "weight": 0},
    highlight_function=lambda _: {"weight": 2, "color": "#333", "fillOpacity": 0.1},
    tooltip=folium.GeoJsonTooltip(
        fields=_tooltip_fields,
        aliases=[_field_aliases.get(f, f) for f in _tooltip_fields],
        localize=True, sticky=True,
        style=("background-color: white; color: #222; "
               "font-family: arial; font-size: 12px; padding: 6px;"),
    ),
).add_to(m)

# Pharmacy markers — render each as a cluster-styled "1" bubble so a
# single pharmacy reads consistently with how 2+ pharmacies cluster
# (e.g. the yellow "22" / green "6" circles).  singleMarkerMode tells
# Leaflet.markercluster to draw even solo markers with the cluster icon.
cluster = MarkerCluster(
    name="Pharmacies",
    options={"singleMarkerMode": True, "showCoverageOnHover": False},
).add_to(m)
for _, row in pharmacies_f.iterrows():
    make_pharmacy_marker(row).add_to(cluster)

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, height=620, use_container_width=True, returned_objects=[])

# --------------------------------------------------------------------------------------
# Charts + table
# --------------------------------------------------------------------------------------
left, right = st.columns([3, 2])

_geo_label = geography.split(" — ")[0]
with left:
    st.subheader(f"📊 {_geo_label}: worst-access on top (Population per Pharmacy)")
    chart_df = (metrics_f.dropna(subset=["pop_per_pharmacy"])
                         .sort_values("pop_per_pharmacy", ascending=False)
                         .head(20))
    _color_col = "state" if "state" in chart_df.columns else ("zone" if "zone" in chart_df.columns else None)
    fig = px.bar(
        chart_df, x="pop_per_pharmacy", y=label_key,
        color=_color_col,
        orientation="h", height=520,
        labels={"pop_per_pharmacy": "Population per Pharmacy", label_key: ""},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader(f"📋 {_geo_label} metrics")
    st.dataframe(
        metrics_f.sort_values("pop_per_pharmacy", ascending=False)
                 .reset_index(drop=True)
                 .style.format({
                     "population": "{:,.0f}",
                     "pharmacy_count": "{:,.0f}",
                     "pop_per_pharmacy": "{:,.0f}",
                     "pharmacies_per_100k": "{:.2f}",
                     "pharmacies_per_1000": "{:.3f}",
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
