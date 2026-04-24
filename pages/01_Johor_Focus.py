"""
Johor Focus page — state-filtered dashboard with an extra "Sub-Mukim Grid"
geography for the dense urban belt around Johor Bahru + Kulai.

Sub-Mukim Grid builds a 1-km-square grid clipped to six target mukim
(Tebrau, Plentong, Pulai, Kota Tinggi, Senai, Kulai) and stamps each cell
with its pharmacy count and WorldPop population.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence

import folium
from folium.plugins import MarkerCluster
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from shapely.geometry import shape, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from dashboard_core import (
    MAP_TILE_PROVIDERS, BRAND_COLORS,
    DATA_SOURCE_LOCAL, GEO_DISTRICT, GEO_MUKIM,
    LOCAL_WORLDPOP_RAW_CSV,
    load_pharmacies, load_geography, build_metrics, choropleth_bins,
    make_pharmacy_marker,
)
from data_pipeline import (
    stamp_polygon_props, compute_polygon_metrics,
    enrich_geojson_with_polygon_metrics,
    load_malaysia_mukim_geojson, normalize_geojson_names,
)
from local_sources import compute_worldpop_per_polygons


# ==============================================================================
# Page config
# ==============================================================================
st.set_page_config(page_title="Johor Focus", page_icon="🔍", layout="wide")

JOHOR_CENTER = [2.0, 103.3]
JOHOR_ZOOM = 9
JOHOR_BAHRU_CENTER = [1.55, 103.7]
JB_ZOOM = 11

GEO_SUBMUKIM = "Sub-Mukim Grid — 1 km cells (JB+Kulai belt)"

# Urban mukim to slice further.  Match is case-insensitive.
# geoBoundaries ADM3 carves town centres into their own "Bandar …"
# polygons nested inside the surrounding mukim, so we include both the
# Mukim and the Bandar explicitly — otherwise the Bandar leaves a
# visible hole in the grid (e.g. Kulai town inside Mukim Kulai).
SUBMUKIM_TARGETS = (
    "Mukim Tebrau",
    "Mukim Plentong",
    "Mukim Pulai",
    "Mukim Kota Tinggi",
    "Mukim Senai",
    "Mukim Kulai",
    "Bandar Johor Bahru",
    "Bandar Kulai",
)

# Bump this whenever the target set or cell size changes so old caches
# are invalidated.
SUBMUKIM_CACHE_KEY = "v3_8mukim_1km"


# ==============================================================================
# Sub-mukim grid builder
# ==============================================================================

def _build_grid(mukim_gj: dict, target_names: Sequence[str],
                cell_deg: float = 0.009,
                state_filter: str = "Johor") -> dict:
    """Build a square grid (cell_deg ≈ 1km at the equator) covering the union
    of the specified mukim, clipped to their boundaries.  Each cell gets a
    unique cell_id + parent_mukim + district + state property.

    Filters by state_filter because mukim names repeat across Malaysia
    (e.g. "Mukim Pulai" exists in both Johor and Kedah).
    """
    target_lower = {n.lower() for n in target_names}
    focus_polys: list = []
    focus_props: list = []
    for f in mukim_gj["features"]:
        props = f["properties"]
        nm = (props.get("mukim") or "").strip().lower()
        if nm in target_lower and (not state_filter or props.get("state") == state_filter):
            focus_polys.append(shape(f["geometry"]))
            focus_props.append(props)

    if not focus_polys:
        return {"type": "FeatureCollection", "features": []}

    union = unary_union(focus_polys)
    minx, miny, maxx, maxy = union.bounds
    tree = STRtree(focus_polys)

    features = []
    cell_id = 0
    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            cell = box(x, y, x + cell_deg, y + cell_deg)
            if union.intersects(cell):
                clipped = cell.intersection(union)
                if not clipped.is_empty and clipped.area > 1e-8:
                    # Assign the cell to its containing mukim via representative_point.
                    rp = clipped.representative_point()
                    parent_mukim = parent_district = parent_state = None
                    for idx in tree.query(rp):
                        if focus_polys[idx].contains(rp):
                            p = focus_props[idx]
                            parent_mukim = p.get("mukim")
                            parent_district = p.get("district")
                            parent_state = p.get("state")
                            break
                    features.append({
                        "type": "Feature",
                        "geometry": clipped.__geo_interface__,
                        "properties": {
                            "cell_id": f"G{cell_id:04d}",
                            "parent_mukim": parent_mukim,
                            "district": parent_district,
                            "state": parent_state,
                        },
                    })
                    cell_id += 1
            x += cell_deg
        y += cell_deg

    return {"type": "FeatureCollection", "features": features}


@st.cache_data(show_spinner="Building sub-mukim grid...")
def _cached_submukim(cache_key: str, cell_deg: float) -> dict:
    """Use the committed pre-built grid GeoJSON when available (much faster
    on Streamlit Cloud first load); fall back to rebuilding from scratch."""
    grid_path = Path("data/submukim_grid_johor.geojson")
    if grid_path.exists():
        return json.loads(grid_path.read_text(encoding="utf-8"))
    mukim_gj = normalize_geojson_names(load_malaysia_mukim_geojson())
    return _build_grid(mukim_gj, SUBMUKIM_TARGETS, cell_deg)


@st.cache_data(show_spinner="Aggregating WorldPop to grid cells...")
def _cached_grid_population(grid_cache_key: str) -> pd.DataFrame:
    """Return per-cell population.  Order of preference:
       1. Pre-computed CSV committed to the repo (works on Streamlit Cloud).
       2. Re-aggregate from the 543 MB raw WorldPop CSV (local dev only).
       3. Zero-population fallback — keeps the map rendering even when
          neither of the above is reachable.
    """
    grid = _cached_submukim(SUBMUKIM_CACHE_KEY, 0.009)
    cache_path = f"data/worldpop_per_submukim_{grid_cache_key}.csv"

    if Path(cache_path).exists():
        return pd.read_csv(cache_path)

    if Path(LOCAL_WORLDPOP_RAW_CSV).exists():
        return compute_worldpop_per_polygons(
            csv_path=LOCAL_WORLDPOP_RAW_CSV,
            polygons_geojson=grid,
            cache_path=cache_path,
            id_properties=["cell_id", "parent_mukim", "district", "state"],
        )

    # Final fallback — neither cache nor raw raster.  Zero pop per cell so
    # the choropleth still renders; the KPI will show "0" obviously.
    st.warning(
        f"`{cache_path}` is missing from this deploy, and the raw WorldPop "
        "CSV isn't available either — population metrics will show as 0."
    )
    return pd.DataFrame({
        "cell_id": [f["properties"]["cell_id"] for f in grid["features"]],
        "parent_mukim": [f["properties"]["parent_mukim"] for f in grid["features"]],
        "district": [f["properties"]["district"] for f in grid["features"]],
        "state": [f["properties"]["state"] for f in grid["features"]],
        "population": 0,
    })


# ==============================================================================
# Sidebar
# ==============================================================================

st.sidebar.title("🔍 Johor Focus")
st.sidebar.info(
    "All metrics filtered to Johor state. Switch to **Sub-Mukim Grid** for "
    "1 km resolution across the Bandar JB / Tebrau / Plentong / Pulai / "
    "Kota Tinggi / Senai / Kulai belt."
)

geography = st.sidebar.radio(
    "Geography",
    [GEO_DISTRICT, GEO_MUKIM, GEO_SUBMUKIM],
    index=1,
)

metric_choice = st.sidebar.radio(
    "Choropleth metric",
    ["pop_per_pharmacy", "pharmacies_per_1000", "pharmacies_per_100k", "population"],
    index=1,
    format_func=lambda x: {
        "pop_per_pharmacy": "Population per Pharmacy (lower = better)",
        "pharmacies_per_1000": "Pharmacies per 1,000 residents",
        "pharmacies_per_100k": "Pharmacies per 100k",
        "population": "Total Population",
    }[x],
)

basemap_name = st.sidebar.selectbox(
    "Basemap", list(MAP_TILE_PROVIDERS), index=0,
)


# ==============================================================================
# Load data (Local source, Johor filter applied throughout)
# ==============================================================================

pharmacies_all = load_pharmacies(DATA_SOURCE_LOCAL)

if geography == GEO_SUBMUKIM:
    grid = _cached_submukim(SUBMUKIM_CACHE_KEY, 0.009)
    pop = _cached_grid_population(SUBMUKIM_CACHE_KEY)
    geo_ctx = {
        "geojson": grid,
        "population": pop,
        "join_keys": ["cell_id"],
        "label_key": "cell_id",
        "base_geojson": grid,  # used for pharmacy-filter state stamping
    }
else:
    geo_ctx = load_geography(DATA_SOURCE_LOCAL, geography)

# Filter the geojson + population to Johor only (for District / Mukim).
if geography in (GEO_DISTRICT, GEO_MUKIM):
    geo_ctx = dict(geo_ctx)  # shallow copy
    johor_only_feats = [
        f for f in geo_ctx["geojson"]["features"]
        if f["properties"].get("state") == "Johor"
    ]
    geo_ctx["geojson"] = {"type": "FeatureCollection", "features": johor_only_feats}
    geo_ctx["population"] = geo_ctx["population"][
        geo_ctx["population"]["state"] == "Johor"
    ].copy()

pharmacies_joined, metrics, enriched_geojson = build_metrics(pharmacies_all, geo_ctx)

# Keep only pharmacies actually in Johor (either by spatial stamp for
# District/Mukim, or by geographic match to the JB belt for Sub-Mukim).
pharmacies_joined = pharmacies_joined[pharmacies_joined["state"] == "Johor"].copy()

# Brand filter (only for the filtered pharmacy set).
brand_options = sorted(pharmacies_joined["brand"].dropna().unique())
selected_brands = st.sidebar.multiselect("Brand / Chain", brand_options, default=brand_options)
pharmacies_f = pharmacies_joined[pharmacies_joined["brand"].isin(selected_brands)].copy()


# ==============================================================================
# Header + KPIs
# ==============================================================================

st.title("🔍 Johor Pharmacy Access — Focused View")
st.caption("Johor-only slice of the national dashboard, with an optional "
           "1-km grid for the Johor Bahru / Kulai urban belt.")

label_key = geo_ctx["label_key"]

# KPI strip
c1, c2, c3, c4 = st.columns(4)
total_pop = int(metrics["population"].sum())
total_phar = int(metrics["pharmacy_count"].sum())
overall_ratio = total_pop / total_phar if total_phar else float("nan")
underserved = int((metrics["pop_per_pharmacy"] > 10_000).sum())

scope = {GEO_DISTRICT: "Johor districts", GEO_MUKIM: "Johor mukim",
         GEO_SUBMUKIM: "Grid cells"}.get(geography, "Polygons")
c1.metric(f"{scope} in view", f"{len(metrics):,}")
c2.metric("Population", f"{total_pop:,}")
c3.metric("Pharmacies", f"{total_phar:,}")
c4.metric(
    "Overall ratio",
    "N/A" if pd.isna(overall_ratio) else f"1 : {int(overall_ratio):,}",
    delta=f"{underserved} polygons >10k:1", delta_color="inverse",
)

# ==============================================================================
# Map
# ==============================================================================

st.subheader("🗺️ Map")

center, zoom = (JOHOR_CENTER, JOHOR_ZOOM)
if geography == GEO_SUBMUKIM:
    center, zoom = (JOHOR_BAHRU_CENTER, JB_ZOOM)

# Init map with chosen basemap.
_provider = MAP_TILE_PROVIDERS[basemap_name]
if "builtin" in _provider:
    m = folium.Map(location=center, zoom_start=zoom,
                   tiles=_provider["builtin"], control_scale=True)
else:
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles=_provider["url"], attr=_provider["attr"], name=basemap_name,
        subdomains=_provider["subdomains"], max_zoom=_provider["max_zoom"],
    ).add_to(m)

# Choropleth
_bins = choropleth_bins(metric_choice, metrics[metric_choice])
_kw = {"threshold_scale": _bins} if _bins and len(_bins) >= 3 else {}
folium.Choropleth(
    geo_data=enriched_geojson,
    data=metrics,
    columns=[label_key, metric_choice],
    key_on=f"feature.properties.{label_key}",
    fill_color="YlOrRd" if metric_choice == "pop_per_pharmacy" else "YlGnBu",
    fill_opacity=0.65, line_opacity=0.4, nan_fill_color="lightgray",
    legend_name={
        "pop_per_pharmacy": "Population per Pharmacy (lower = better)",
        "pharmacies_per_1000": "Pharmacies per 1,000 residents",
        "pharmacies_per_100k": "Pharmacies per 100k",
        "population": "Population",
    }[metric_choice],
    name="Choropleth",
    **_kw,
).add_to(m)

# Tooltip overlay
_candidate_fields = [label_key]
if "district" in geo_ctx["join_keys"]:
    _candidate_fields.append("district")
if "state" in geo_ctx["join_keys"]:
    _candidate_fields.append("state")
if geography == GEO_SUBMUKIM:
    _candidate_fields.extend(["parent_mukim", "district", "state"])
_candidate_fields += ["population", "pharmacy_count", "pop_per_pharmacy", "pharmacies_per_1000"]
_aliases = {
    "cell_id": "Grid Cell:", "parent_mukim": "Mukim:",
    "mukim": "Mukim:", "district": "District:", "state": "State:",
    "population": "Population:", "pharmacy_count": "Pharmacies:",
    "pop_per_pharmacy": "Pop / Pharmacy:",
    "pharmacies_per_1000": "Pharmacies / 1,000:",
}
_sample = enriched_geojson["features"][0]["properties"] if enriched_geojson["features"] else {}
# Dedupe while preserving order.
seen = set()
_tooltip_fields = []
for f in _candidate_fields:
    if f not in seen and f in _sample:
        seen.add(f); _tooltip_fields.append(f)

folium.GeoJson(
    enriched_geojson,
    name=f"{scope} info",
    style_function=lambda _: {"fillOpacity": 0, "color": "transparent", "weight": 0},
    highlight_function=lambda _: {"weight": 2, "color": "#333", "fillOpacity": 0.15},
    tooltip=folium.GeoJsonTooltip(
        fields=_tooltip_fields,
        aliases=[_aliases.get(f, f) for f in _tooltip_fields],
        localize=True, sticky=True,
        style="background-color: white; color: #222; font-family: arial; font-size: 12px; padding: 6px;",
    ),
).add_to(m)

# Pharmacy markers — render each as a cluster-styled "1" bubble to match
# the visual language of the 2+ cluster counts (same yellow/green/orange
# scale Leaflet.markercluster uses by default).
cluster = MarkerCluster(
    name="Pharmacies",
    options={"singleMarkerMode": True, "showCoverageOnHover": False},
).add_to(m)
for _, row in pharmacies_f.iterrows():
    make_pharmacy_marker(row).add_to(cluster)

folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, height=640, use_container_width=True, returned_objects=[])


# ==============================================================================
# Charts + table
# ==============================================================================

left, right = st.columns([3, 2])
with left:
    st.subheader(f"📊 {scope} — worst-access first")
    chart_df = (metrics.dropna(subset=["pop_per_pharmacy"])
                        .sort_values("pop_per_pharmacy", ascending=False)
                        .head(25))
    color_col = "district" if "district" in chart_df.columns else "parent_mukim"
    if color_col not in chart_df.columns:
        color_col = None
    fig = px.bar(
        chart_df, x="pop_per_pharmacy", y=label_key,
        color=color_col, orientation="h", height=520,
        labels={"pop_per_pharmacy": "Population per Pharmacy", label_key: ""},
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader(f"📋 {scope} table")
    sort_col = "pop_per_pharmacy" if metric_choice == "pop_per_pharmacy" else metric_choice
    ascending = metric_choice == "pop_per_pharmacy"  # higher=worse for this one
    st.dataframe(
        metrics.sort_values(sort_col, ascending=False)
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


with st.expander("ℹ️ Notes on the Sub-Mukim Grid"):
    st.markdown(
        f"""
The Sub-Mukim Grid divides the **seven most urban units** in southern Johor
— Bandar Johor Bahru (city centre), Mukim Tebrau, Plentong, Pulai,
Kota Tinggi, Senai, and Kulai — into **~1 km square cells** (0.009° per
side) clipped to each unit's geoBoundaries ADM3 outline. Each cell is
stamped with its containing mukim and gets its own WorldPop 2020
population sum, which surfaces intra-mukim coverage gaps (e.g. dense CBD
blocks around Bandar JB vs suburbs, or UTM-Skudai cluster vs western
Mukim Pulai, or the Senai airport area vs Kulai town core).

For districts outside this belt, stick with the **Mukim** geography — there
are ~80 Johor mukim in total, most of which are rural and don't warrant
sub-division.
"""
    )
