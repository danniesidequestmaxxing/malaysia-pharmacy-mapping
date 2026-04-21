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

from mock_data import (
    get_districts_df,
    generate_mock_pharmacies,
    generate_mock_geojson,
)
from data_pipeline import (
    spatial_join_pharmacies_to_districts,
    stamp_polygon_props,
    compute_district_metrics,
    compute_polygon_metrics,
    enrich_geojson_with_metrics,
    enrich_geojson_with_polygon_metrics,
    load_pharmacies_from_csv,
    load_population_from_api,
    load_district_geojson,
    # Live-data loaders (no API key required)
    fetch_pharmacies_from_osm,
    load_population_district_dosm,
    load_malaysia_districts_geojson,
    load_malaysia_mukim_geojson,
    load_kkm_zones_geojson,
    build_voronoi_catchments,
    zone_for_state,
    KKM_ZONES,
    normalize_district_names,
    normalize_geojson_names,
)
from local_sources import (
    parse_kmz,
    parse_pmg_excel,
    merge_pharmacy_sources,
    compute_worldpop_per_district,
    compute_worldpop_per_polygons,
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

# Aggregation granularity for the choropleth. District is the default.
GEO_DISTRICT = "District (Daerah) — 159"
GEO_MUKIM = "Mukim (Sub-district) — 1,859"
GEO_ZONE = "KKM Zone — 5"
GEO_CATCHMENT = "Pharmacy Catchment (Voronoi)"

# Paths for the Local pipeline — committed processed artifacts so the deploy
# does not need access to the 17 GB WorldPop TIF or the raw NPRA PDF.
LOCAL_KMZ_PATH = "data/source/Project Pharma.kmz"
LOCAL_NPRA_GEOCODED_CSV = "data/pharmacies_npra_geocoded.csv"
LOCAL_PMG_GEOCODED_CSV = "data/pharmacies_pmg_geocoded.csv"
LOCAL_WORLDPOP_PER_DISTRICT = "data/worldpop_per_district.csv"
LOCAL_WORLDPOP_PER_MUKIM = "data/worldpop_per_mukim.csv"
LOCAL_WORLDPOP_PER_ZONE = "data/worldpop_per_zone.csv"
LOCAL_WORLDPOP_RAW_CSV = "/Users/evoverebitda/Downloads/mys_general_2020.csv"  # only needed for refresh


@st.cache_data(show_spinner="Loading pharmacies...")
def load_pharmacies(data_source: str,
                    pharmacy_csv: str | None = None) -> pd.DataFrame:
    """Just the pharmacy DataFrame for the chosen data source. Geography-agnostic."""
    if data_source == DATA_SOURCE_MOCK:
        return generate_mock_pharmacies()
    if data_source == DATA_SOURCE_LOCAL:
        sources = []
        if Path(LOCAL_KMZ_PATH).exists():
            sources.append(parse_kmz(LOCAL_KMZ_PATH))
        if Path(LOCAL_NPRA_GEOCODED_CSV).exists():
            npra = pd.read_csv(LOCAL_NPRA_GEOCODED_CSV)
            npra["source"] = npra.get("source", "NPRA").fillna("NPRA")
            # Brand column exists in newer CSVs; back-compat: default to "NPRA".
            npra["brand"] = npra.get("brand", "NPRA").fillna("NPRA")
            sources.append(npra)
        if Path(LOCAL_PMG_GEOCODED_CSV).exists():
            pmg = pd.read_csv(LOCAL_PMG_GEOCODED_CSV)
            pmg["source"] = pmg.get("source", "PMG").fillna("PMG")
            pmg["brand"] = pmg.get("brand", "PMG").fillna("PMG")
            sources.append(pmg)
        if not sources:
            st.error(
                f"No local pharmacy sources found — place KMZ at {LOCAL_KMZ_PATH}, "
                f"run `python geocode_npra.py` for {LOCAL_NPRA_GEOCODED_CSV}, "
                f"and `python geocode_pmg.py` for {LOCAL_PMG_GEOCODED_CSV}."
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
    """Return the geography-specific (geojson, population_df, join_keys, label_key).

    `label_key` is the primary display column (e.g. "district", "mukim", "zone",
    "pharmacy").
    """
    # ADM2 district is the base layer for everything — we use it as the "always"
    # context for state/district filtering of pharmacies regardless of choropleth
    # granularity.
    base_geojson = normalize_geojson_names(load_malaysia_districts_geojson())

    if data_source == DATA_SOURCE_MOCK:
        # Mock stays district-only — the tutorial / offline path.
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

    # GEO_CATCHMENT — needs pharmacies to build the Voronoi.
    if pharmacies_for_catchment is None or len(pharmacies_for_catchment) < 3:
        st.error("Catchment geography needs at least 3 pharmacies.")
        st.stop()
    voronoi = build_voronoi_catchments(
        pharmacies_for_catchment[["pharmacy_id", "name", "brand",
                                  "latitude", "longitude"]].dropna(
            subset=["latitude", "longitude"]),
        boundary_geojson=base_geojson,
    )
    # Compute population per catchment from raw WorldPop if the raw CSV is
    # available; otherwise skip (catchment mode needs the raw raster).
    cache_name = f"data/worldpop_per_catchment_{len(voronoi['features'])}.csv"
    if Path(LOCAL_WORLDPOP_RAW_CSV).exists():
        pop = compute_worldpop_per_polygons(
            csv_path=LOCAL_WORLDPOP_RAW_CSV,
            polygons_geojson=voronoi,
            cache_path=cache_name,
            id_properties=["catchment_id", "pharmacy", "brand"],
        )
    else:
        # No raw raster — can still render the polygons with 0 population.
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
    enrich the GeoJSON so tooltips read the metrics directly."""
    # Always stamp base district/state for pharmacy filtering, regardless of
    # choropleth granularity.
    with_base = stamp_polygon_props(
        pharmacies, geo_ctx["base_geojson"], ["district", "state"]
    )
    # Then stamp with the choropleth geography's keys (may overlap with base
    # keys — stamp_polygon_props handles the _raw collision by renaming).
    extra_keys = [k for k in geo_ctx["join_keys"] if k not in ("district", "state")]
    if extra_keys:
        with_all = stamp_polygon_props(
            with_base, geo_ctx["geojson"], extra_keys
        )
    else:
        with_all = with_base

    metrics = compute_polygon_metrics(
        with_all, geo_ctx["population"], on=geo_ctx["join_keys"]
    )
    enriched = enrich_geojson_with_polygon_metrics(
        geo_ctx["geojson"], metrics, on=geo_ctx["join_keys"]
    )
    return with_all, metrics, enriched


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


def _choropleth_bins(metric: str, series: pd.Series) -> list | None:
    """Tuned threshold_scale per metric so the colour ramp actually shows
    variation.  Folium's default is 7 quantile bins, which breaks badly on
    skewed distributions: at mukim level most polygons are 0 and a few are
    outliers, so the default legend ends up 0→15 while real values live in
    0.05→0.5.

    The breakpoints below are anchored to KKM/WHO pharmacy-access norms
    (Malaysia target is ~1 retail pharmacy per 3-5k people = 0.2-0.33 per
    1,000). The last bin is always stretched to cover the actual max so
    Folium's strict range check accepts every value — outliers get pooled
    into the top bin instead of blowing up the scale on the way there."""
    s = series.dropna()
    if s.empty:
        return None
    hi = float(s.max())

    def _cap(base: list, top_policy: float, round_to: int = 2) -> list:
        """Extend `base` to cover `hi` if needed, keeping the policy band intact."""
        if hi <= top_policy:
            return base  # all values inside the policy-anchored range
        return base + [round(float(np.ceil(hi)), round_to)]

    if metric == "pop_per_pharmacy":
        # Ratio — higher is worse.  Policy band 0-50k, round outlier to 1k.
        base = [0, 3000, 5000, 10000, 20000, 50000]
        return _cap(base, 50000, round_to=0) if hi <= 50000 else base + [int(np.ceil(hi / 1000) * 1000)]
    if metric == "pharmacies_per_1000":
        # Saturation ≈ 1.0 per 1,000. Outliers (tiny-pop mukim) go in the top bin.
        base = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
        return _cap(base, 1.0, round_to=1)
    if metric == "pharmacies_per_100k":
        base = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]
        return _cap(base, 100.0, round_to=0)
    if metric == "population":
        qs = list(s.quantile([0.0, 0.2, 0.4, 0.6, 0.8, 0.95]).round().astype(int))
        return sorted(set(qs + [int(np.ceil(hi))]))
    return None


_bins = _choropleth_bins(metric_choice, metrics_f[metric_choice])
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

# Pharmacy markers (clustered for performance). Colour by brand when we have
# that signal (Local mode), fall back to a single navy marker otherwise.
BRAND_COLORS = {
    # KMZ chains (matches the folder colours from Project Pharma.kmz)
    "Caring":            "#2e7d32",  # green
    "Alpro":             "#6a1b9a",  # purple
    "BIG Pharmacy":      "#c62828",  # red
    "Healthlane":        "#546e7a",  # slate
    "Sunway Multicare":  "#1a237e",  # dark blue
    "AA Pharmacy":       "#f9a825",  # yellow
    # KMZ sub-brands nested under Caring / Healthlane
    "Georgetown":        "#66bb6a",  # light green
    "Wellings":          "#388e3c",  # dark green
    "Straits":           "#78909c",  # light slate
    # Chains that appear in NPRA / PMG but not in the Project Pharma KMZ
    "Guardian":          "#0d47a1",  # navy
    "Watsons":           "#d32f2f",  # dark red
    "PMG":               "#00695c",  # teal
    "AM PM":             "#ad1457",  # crimson
    "Nazen":             "#5d4037",  # brown
    "Siang":             "#e65100",  # deep orange
    "Alliance":          "#4527a0",  # indigo
    "Constant":          "#7b1fa2",  # magenta
    "Be Pharmacy":       "#0277bd",  # cyan
    "MediQ":             "#2e7d32",
    "Rx":                "#1565c0",
    "Mega Kulim":        "#6d4c41",
    "Rejoice":           "#bf360c",
    "Farmasi":           "#455a64",
    "KS":                "#37474f",
    # Catch-alls
    "NPRA":              "#ef6c00",  # orange — NPRA row without chain match
    "Independent":       "#455a64",  # slate — generic independent
    "Other":             "#1f4e79",
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
