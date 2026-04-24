"""
metro_focus.py
--------------
Shared scaffolding for every metro-focus page under `pages/`.

Each metro page calls `render_metro_focus(config)` with its own config
dict; this module handles the common work:
  * building / loading the 1 km grid clipped to the metro's polygons
  * aggregating WorldPop to each cell (or reading the pre-committed CSV)
  * rendering the sidebar, KPI strip, choropleth, markers, and tables

Supports two clipping strategies:
  * `strategy="mukim"` — clip to selected mukim (ADM3) polygons,
    identified either by explicit name list or by parent district.
    Used for Johor, Klang Valley, Kuching.
  * `strategy="district"` — clip to ADM2 district polygons.  Used for
    Penang + KK where geoBoundaries has no ADM3 data for the state.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence

import folium
from folium.plugins import MarkerCluster
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from shapely.geometry import shape, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from dashboard_core import (
    MAP_TILE_PROVIDERS, DATA_SOURCE_LOCAL, GEO_DISTRICT, GEO_MUKIM,
    LOCAL_WORLDPOP_RAW_CSV,
    load_pharmacies, load_geography, build_metrics, choropleth_bins,
    make_pharmacy_marker,
)
from data_pipeline import (
    load_malaysia_mukim_geojson, load_malaysia_districts_geojson,
    normalize_geojson_names,
)
from local_sources import compute_worldpop_per_polygons


# --------------------------------------------------------------------------------------
# Target-polygon selection
# --------------------------------------------------------------------------------------

def select_mukim_by_names(mukim_gj: dict, names: Sequence[str],
                          state_filter: Sequence[str] | None = None
                          ) -> tuple[list, list]:
    """Explicit-name mukim selector (used by the Johor page).  Case-insensitive
    match on the `mukim` property; optionally further restricted by state."""
    wanted = {n.lower() for n in names}
    states = {s for s in (state_filter or [])}
    polys, props = [], []
    for f in mukim_gj["features"]:
        p = f["properties"]
        if states and p.get("state") not in states:
            continue
        if (p.get("mukim") or "").strip().lower() in wanted:
            polys.append(shape(f["geometry"]))
            props.append(p)
    return polys, props


def select_mukim_by_districts(mukim_gj: dict, districts: Sequence[str],
                              state_filter: Sequence[str] | None = None
                              ) -> tuple[list, list]:
    """All mukim whose parent district is in `districts`.  Used for the
    Klang Valley + Kuching metro pages."""
    states = {s for s in (state_filter or [])}
    wanted = set(districts)
    polys, props = [], []
    for f in mukim_gj["features"]:
        p = f["properties"]
        if states and p.get("state") not in states:
            continue
        if p.get("district") in wanted:
            polys.append(shape(f["geometry"]))
            props.append(p)
    return polys, props


def select_districts_adm2(adm2_gj: dict, districts: Sequence[str],
                          state_filter: Sequence[str] | None = None
                          ) -> tuple[list, list]:
    """ADM2-level clipping (Penang + KK).  Falls back to using the district
    polygons themselves as the sub-metro layer because geoBoundaries has no
    ADM3 for these states.  Returns (polys, props) where props[i] includes
    a synthesised `mukim` key (= the district name) so the downstream
    grid-building code stays uniform."""
    states = {s for s in (state_filter or [])}
    wanted = set(districts)
    polys, props = [], []
    for f in adm2_gj["features"]:
        p = f["properties"]
        if states and p.get("state") not in states:
            continue
        if p.get("district") in wanted:
            polys.append(shape(f["geometry"]))
            props.append({
                "mukim": p.get("district"),      # synthesised
                "district": p.get("district"),
                "state": p.get("state"),
            })
    return polys, props


# --------------------------------------------------------------------------------------
# Grid building
# --------------------------------------------------------------------------------------

def build_grid(polys: list, props: list, cell_deg: float = 0.009) -> dict:
    """Build a regular `cell_deg`-degree grid (≈1 km at the equator) clipped
    to the union of `polys`.  Every cell gets a unique cell_id plus
    parent_mukim + district + state from the polygon containing its
    representative point."""
    if not polys:
        return {"type": "FeatureCollection", "features": []}
    union = unary_union(polys)
    tree = STRtree(polys)
    minx, miny, maxx, maxy = union.bounds

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
                    rp = clipped.representative_point()
                    parent = {}
                    for idx in tree.query(rp):
                        if polys[idx].contains(rp):
                            parent = props[idx]
                            break
                    features.append({
                        "type": "Feature",
                        "geometry": clipped.__geo_interface__,
                        "properties": {
                            "cell_id": f"G{cell_id:04d}",
                            "parent_mukim": parent.get("mukim"),
                            "district": parent.get("district"),
                            "state": parent.get("state"),
                        },
                    })
                    cell_id += 1
            x += cell_deg
        y += cell_deg

    return {"type": "FeatureCollection", "features": features}


# --------------------------------------------------------------------------------------
# Cached loaders — keyed by config so Streamlit's cache stays clean across pages
# --------------------------------------------------------------------------------------

@st.cache_data(show_spinner="Building sub-mukim grid...")
def _cached_grid(metro_key: str, grid_path: str, cell_deg: float,
                 strategy: str, target_names: tuple[str, ...] | None,
                 target_districts: tuple[str, ...] | None,
                 state_filter: tuple[str, ...]) -> dict:
    # 1. Committed grid file wins if present (fast path on Streamlit Cloud).
    gp = Path(grid_path)
    if gp.exists():
        return json.loads(gp.read_text(encoding="utf-8"))

    # 2. Build the grid from whichever polygon layer the config asks for.
    if strategy == "mukim_names":
        mukim_gj = normalize_geojson_names(load_malaysia_mukim_geojson())
        polys, props = select_mukim_by_names(mukim_gj, target_names or (), state_filter)
    elif strategy == "mukim_districts":
        mukim_gj = normalize_geojson_names(load_malaysia_mukim_geojson())
        polys, props = select_mukim_by_districts(mukim_gj, target_districts or (), state_filter)
    elif strategy == "adm2":
        adm2_gj = normalize_geojson_names(load_malaysia_districts_geojson())
        polys, props = select_districts_adm2(adm2_gj, target_districts or (), state_filter)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    if not polys:
        st.warning(f"[{metro_key}] No polygons matched the target list.")
    grid = build_grid(polys, props, cell_deg)

    gp.parent.mkdir(parents=True, exist_ok=True)
    gp.write_text(json.dumps(grid), encoding="utf-8")
    return grid


@st.cache_data(show_spinner="Aggregating WorldPop to grid cells...")
def _cached_grid_population(metro_key: str, pop_cache_path: str,
                             grid: dict) -> pd.DataFrame:
    # 1. Pre-computed CSV (committed to repo).
    pp = Path(pop_cache_path)
    if pp.exists():
        return pd.read_csv(pp)

    # 2. Re-aggregate from the raw WorldPop CSV (local dev only).
    if Path(LOCAL_WORLDPOP_RAW_CSV).exists():
        return compute_worldpop_per_polygons(
            csv_path=LOCAL_WORLDPOP_RAW_CSV,
            polygons_geojson=grid,
            cache_path=pop_cache_path,
            id_properties=["cell_id", "parent_mukim", "district", "state"],
        )

    # 3. Zero-population fallback.
    st.warning(
        f"{pop_cache_path} missing from this deploy and raw WorldPop is "
        "unavailable — population metrics will show as 0."
    )
    return pd.DataFrame({
        "cell_id": [f["properties"]["cell_id"] for f in grid["features"]],
        "parent_mukim": [f["properties"]["parent_mukim"] for f in grid["features"]],
        "district": [f["properties"]["district"] for f in grid["features"]],
        "state": [f["properties"]["state"] for f in grid["features"]],
        "population": 0,
    })


# --------------------------------------------------------------------------------------
# Page renderer
# --------------------------------------------------------------------------------------

def render_metro_focus(config: dict) -> None:
    """Render a full metro-focus page from a config dict.

    Required keys:
      name              human-readable metro name ("Klang Valley")
      icon              page icon emoji
      center, zoom      default Folium map location
      sub_center,
      sub_zoom          zoom used when the sub-mukim grid is active
      state_filter      list of states; pharmacies + metrics filtered here
      strategy          "mukim_names" | "mukim_districts" | "adm2"
      target_names      tuple[str] (if strategy == mukim_names)
      target_districts  tuple[str] (if strategy in {mukim_districts, adm2})
      cache_key         used to name the grid + population files
      grid_path         path under data/ for the pre-built grid GeoJSON
      pop_path          path under data/ for the pre-computed WorldPop CSV
      intro             one-line summary shown in the sidebar info box
    """
    st.set_page_config(
        page_title=f"{config['name']} Focus",
        page_icon=config.get("icon", "🗺️"),
        layout="wide",
    )

    grid_geo_key = f"Sub-{'Mukim' if config['strategy'].startswith('mukim') else 'District'} Grid — 1 km cells"

    # ---- Sidebar ----
    st.sidebar.title(f"{config.get('icon','🗺️')} {config['name']} Focus")
    st.sidebar.info(config.get("intro", f"All metrics filtered to {config['name']}."))

    geography = st.sidebar.radio(
        "Geography", [GEO_DISTRICT, GEO_MUKIM, grid_geo_key], index=2,
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
    basemap_name = st.sidebar.selectbox("Basemap", list(MAP_TILE_PROVIDERS), index=0)

    # ---- Load ----
    pharmacies_all = load_pharmacies(DATA_SOURCE_LOCAL)

    state_filter = tuple(config["state_filter"])
    target_names = tuple(config.get("target_names") or ())
    target_districts = tuple(config.get("target_districts") or ())

    if geography == grid_geo_key:
        grid = _cached_grid(
            metro_key=config["cache_key"],
            grid_path=config["grid_path"],
            cell_deg=0.009,
            strategy=config["strategy"],
            target_names=target_names or None,
            target_districts=target_districts or None,
            state_filter=state_filter,
        )
        pop = _cached_grid_population(config["cache_key"], config["pop_path"], grid)
        geo_ctx = {
            "geojson": grid, "population": pop,
            "join_keys": ["cell_id"], "label_key": "cell_id",
            "base_geojson": grid,
        }
    else:
        geo_ctx = load_geography(DATA_SOURCE_LOCAL, geography)

    # Filter geojson + population to the metro's state(s) (District / Mukim only).
    if geography in (GEO_DISTRICT, GEO_MUKIM):
        geo_ctx = dict(geo_ctx)
        geo_ctx["geojson"] = {
            "type": "FeatureCollection",
            "features": [f for f in geo_ctx["geojson"]["features"]
                         if f["properties"].get("state") in state_filter],
        }
        geo_ctx["population"] = geo_ctx["population"][
            geo_ctx["population"]["state"].isin(state_filter)
        ].copy()

    pharmacies_joined, metrics, enriched_geojson = build_metrics(pharmacies_all, geo_ctx)
    pharmacies_joined = pharmacies_joined[
        pharmacies_joined["state"].isin(state_filter)
    ].copy()

    # Brand filter on the state-filtered pharmacy set.
    brand_options = sorted(pharmacies_joined["brand"].dropna().unique())
    selected_brands = st.sidebar.multiselect(
        "Brand / Chain", brand_options, default=brand_options,
    )
    pharmacies_f = pharmacies_joined[pharmacies_joined["brand"].isin(selected_brands)].copy()

    # ---- Header + KPIs ----
    st.title(f"{config.get('icon','🗺️')} {config['name']} Pharmacy Access")
    st.caption(config.get("intro", ""))

    label_key = geo_ctx["label_key"]
    c1, c2, c3, c4 = st.columns(4)
    total_pop = int(metrics["population"].sum())
    total_phar = int(metrics["pharmacy_count"].sum())
    overall_ratio = total_pop / total_phar if total_phar else float("nan")
    underserved = int((metrics["pop_per_pharmacy"] > 10_000).sum())

    scope_label = {GEO_DISTRICT: "Districts",
                   GEO_MUKIM: "Mukim",
                   grid_geo_key: "Grid cells"}.get(geography, "Polygons")
    c1.metric(f"{scope_label} in view", f"{len(metrics):,}")
    c2.metric("Population", f"{total_pop:,}")
    c3.metric("Pharmacies", f"{total_phar:,}")
    c4.metric(
        "Overall ratio",
        "N/A" if pd.isna(overall_ratio) else f"1 : {int(overall_ratio):,}",
        delta=f"{underserved} polygons >10k:1", delta_color="inverse",
    )

    # ---- Map ----
    st.subheader("🗺️ Map")
    center = config["sub_center"] if geography == grid_geo_key else config["center"]
    zoom = config["sub_zoom"] if geography == grid_geo_key else config["zoom"]

    provider = MAP_TILE_PROVIDERS[basemap_name]
    if "builtin" in provider:
        m = folium.Map(location=center, zoom_start=zoom,
                       tiles=provider["builtin"], control_scale=True)
    else:
        m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
        folium.TileLayer(
            tiles=provider["url"], attr=provider["attr"], name=basemap_name,
            subdomains=provider["subdomains"], max_zoom=provider["max_zoom"],
        ).add_to(m)

    bins = choropleth_bins(metric_choice, metrics[metric_choice])
    kw = {"threshold_scale": bins} if bins and len(bins) >= 3 else {}
    folium.Choropleth(
        geo_data=enriched_geojson, data=metrics,
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
        name="Choropleth", **kw,
    ).add_to(m)

    # Tooltip fields dynamic per geography.
    candidates = [label_key]
    if "district" in geo_ctx["join_keys"] and "district" not in candidates:
        candidates.append("district")
    if "state" in geo_ctx["join_keys"] and "state" not in candidates:
        candidates.append("state")
    if geography == grid_geo_key:
        candidates += ["parent_mukim", "district", "state"]
    candidates += ["population", "pharmacy_count", "pop_per_pharmacy", "pharmacies_per_1000"]
    aliases = {
        "cell_id": "Grid Cell:", "parent_mukim": "Mukim:",
        "mukim": "Mukim:", "district": "District:", "state": "State:",
        "population": "Population:", "pharmacy_count": "Pharmacies:",
        "pop_per_pharmacy": "Pop / Pharmacy:",
        "pharmacies_per_1000": "Pharmacies / 1,000:",
    }
    sample = enriched_geojson["features"][0]["properties"] if enriched_geojson["features"] else {}
    seen, tooltip_fields = set(), []
    for f in candidates:
        if f not in seen and f in sample:
            seen.add(f); tooltip_fields.append(f)
    folium.GeoJson(
        enriched_geojson, name=f"{scope_label} info",
        style_function=lambda _: {"fillOpacity": 0, "color": "transparent", "weight": 0},
        highlight_function=lambda _: {"weight": 2, "color": "#333", "fillOpacity": 0.15},
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=[aliases.get(f, f) for f in tooltip_fields],
            localize=True, sticky=True,
            style="background-color: white; color: #222; font-family: arial; font-size: 12px; padding: 6px;",
        ),
    ).add_to(m)

    cluster = MarkerCluster(
        name="Pharmacies",
        options={"singleMarkerMode": True, "showCoverageOnHover": False},
    ).add_to(m)
    for _, row in pharmacies_f.iterrows():
        make_pharmacy_marker(row).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, height=640, use_container_width=True, returned_objects=[])

    # ---- Table + chart ----
    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"📊 {scope_label} — worst-access first")
        chart_df = (metrics.dropna(subset=["pop_per_pharmacy"])
                           .sort_values("pop_per_pharmacy", ascending=False)
                           .head(25))
        color_col = "district" if "district" in chart_df.columns else (
            "parent_mukim" if "parent_mukim" in chart_df.columns else None
        )
        fig = px.bar(
            chart_df, x="pop_per_pharmacy", y=label_key,
            color=color_col, orientation="h", height=520,
            labels={"pop_per_pharmacy": "Population per Pharmacy", label_key: ""},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"},
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader(f"📋 {scope_label} table")
        st.dataframe(
            metrics.sort_values("pop_per_pharmacy", ascending=False)
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
