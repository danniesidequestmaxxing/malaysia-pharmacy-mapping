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
import numpy as np
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
    load_pharmacies, load_geography, build_metrics,
    rebuild_metrics_from_joined_pharmacies, choropleth_bins,
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
# 5 km neighborhood metrics (grid view only)
# --------------------------------------------------------------------------------------
#
# For each grid cell we expose four extra access metrics built from the cells
# *and* pharmacies within a 5 km radius of the cell's representative point:
#
#   population_5km           sum of cell populations whose centroid is ≤ 5 km away
#   pharmacies_5km           count of pharmacies whose lat/lon is ≤ 5 km away
#   pop_per_pharmacy_5km     population_5km / pharmacies_5km (NaN if zero pharmacies)
#   pharmacies_per_1000_5km  pharmacies_5km / population_5km × 1,000
#
# Distance is computed with an equirectangular projection anchored at the
# metro's mean cell latitude — well within sub-percent error at Malaysia's
# 1°-7°N latitude band, and avoids pulling in scipy/sklearn.

NEIGHBORHOOD_RADIUS_KM = 5.0
_KM_PER_DEG_LAT = 111.32


def _project_xy(lons: np.ndarray, lats: np.ndarray, ref_lat_rad: float) -> np.ndarray:
    """Equirectangular projection to local kilometres around `ref_lat_rad`."""
    km_per_deg_lon = _KM_PER_DEG_LAT * float(np.cos(ref_lat_rad))
    return np.column_stack([lons * km_per_deg_lon, lats * _KM_PER_DEG_LAT])


def _grid_cell_centroids(grid: dict) -> pd.DataFrame:
    """Per-cell representative point. One row per feature."""
    rows = []
    for f in grid["features"]:
        geom = shape(f["geometry"])
        rp = geom.representative_point()
        rows.append({
            "cell_id": f["properties"]["cell_id"],
            "lon": float(rp.x),
            "lat": float(rp.y),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Aggregating population within 5 km of each cell...")
def _cached_population_5km(metro_key: str, grid_path: str, pop_path: str,
                           radius_km: float = NEIGHBORHOOD_RADIUS_KM) -> pd.DataFrame:
    """Cell→cell aggregate that depends only on the static grid + WorldPop CSV.

    Cached by `metro_key` so it's computed once per page deploy and reused
    across reruns regardless of the user's brand filter.
    """
    grid = json.loads(Path(grid_path).read_text(encoding="utf-8"))
    centroids = _grid_cell_centroids(grid)

    pop = pd.read_csv(pop_path)[["cell_id", "population"]] \
        if Path(pop_path).exists() else pd.DataFrame(
            {"cell_id": centroids["cell_id"], "population": 0})
    df = centroids.merge(pop, on="cell_id", how="left")
    df["population"] = df["population"].fillna(0).astype(float)

    if df.empty:
        return pd.DataFrame(columns=["cell_id", "population_5km"])

    ref_lat_rad = float(np.deg2rad(df["lat"].mean()))
    xy = _project_xy(df["lon"].to_numpy(), df["lat"].to_numpy(), ref_lat_rad)
    pops = df["population"].to_numpy()

    n = len(df)
    pop_5km = np.zeros(n, dtype=float)
    r2 = float(radius_km) ** 2

    # Chunked broadcasting keeps peak memory bounded on the larger grids
    # (Klang Valley has ~8.7 k cells → an n×n matrix would be ~610 MB at f64).
    chunk = 200
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dx = xy[start:end, 0:1] - xy[None, :, 0]
        dy = xy[start:end, 1:2] - xy[None, :, 1]
        d2 = dx * dx + dy * dy
        mask = d2 <= r2
        pop_5km[start:end] = mask @ pops

    return pd.DataFrame({
        "cell_id": df["cell_id"].to_numpy(),
        "lon": df["lon"].to_numpy(),
        "lat": df["lat"].to_numpy(),
        "population_5km": pop_5km,
    })


def _pharmacies_within_5km(centroids: pd.DataFrame,
                            pharmacies: pd.DataFrame,
                            radius_km: float = NEIGHBORHOOD_RADIUS_KM
                            ) -> pd.Series:
    """Per-cell pharmacy count within `radius_km`. Re-runs on every brand
    filter change — fast enough that caching is unnecessary."""
    if centroids.empty:
        return pd.Series(dtype=int)
    valid = pharmacies.dropna(subset=["latitude", "longitude"])
    n = len(centroids)
    if valid.empty:
        return pd.Series(np.zeros(n, dtype=int), index=centroids["cell_id"].to_numpy())

    ref_lat_rad = float(np.deg2rad(centroids["lat"].mean()))
    cell_xy = _project_xy(centroids["lon"].to_numpy(),
                           centroids["lat"].to_numpy(), ref_lat_rad)
    pharm_xy = _project_xy(valid["longitude"].to_numpy(),
                            valid["latitude"].to_numpy(), ref_lat_rad)

    counts = np.zeros(n, dtype=int)
    r2 = float(radius_km) ** 2
    chunk = 500
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dx = cell_xy[start:end, 0:1] - pharm_xy[None, :, 0]
        dy = cell_xy[start:end, 1:2] - pharm_xy[None, :, 1]
        d2 = dx * dx + dy * dy
        counts[start:end] = (d2 <= r2).sum(axis=1)

    return pd.Series(counts, index=centroids["cell_id"].to_numpy(), name="pharmacies_5km")


def compute_neighborhood_metrics(metro_key: str, grid_path: str, pop_path: str,
                                  pharmacies: pd.DataFrame,
                                  radius_km: float = NEIGHBORHOOD_RADIUS_KM
                                  ) -> pd.DataFrame:
    """Build the per-cell 5 km neighborhood frame.

    Output columns: `cell_id, population_5km, pharmacies_5km,
    pop_per_pharmacy_5km, pharmacies_per_1000_5km`.
    """
    pop_df = _cached_population_5km(metro_key, grid_path, pop_path, radius_km)
    if pop_df.empty:
        return pd.DataFrame(columns=[
            "cell_id", "population_5km", "pharmacies_5km",
            "pop_per_pharmacy_5km", "pharmacies_per_1000_5km",
        ])

    pharm_counts = _pharmacies_within_5km(
        pop_df[["cell_id", "lon", "lat"]], pharmacies, radius_km
    )
    out = pop_df[["cell_id", "population_5km"]].copy()
    out["pharmacies_5km"] = (
        out["cell_id"].map(pharm_counts).fillna(0).astype(int)
    )
    out["pop_per_pharmacy_5km"] = np.where(
        out["pharmacies_5km"] > 0,
        out["population_5km"] / out["pharmacies_5km"],
        np.nan,
    )
    pop_safe = out["population_5km"].replace(0, np.nan)
    out["pharmacies_per_1000_5km"] = (
        out["pharmacies_5km"] / pop_safe * 1_000
    )
    return out


def _inject_neighborhood_props(geojson: dict, neighborhood: pd.DataFrame) -> dict:
    """Stamp the 5 km metrics onto each feature's `properties` so the Folium
    GeoJsonTooltip can read them directly."""
    if neighborhood.empty:
        return geojson
    lookup = neighborhood.set_index("cell_id").to_dict(orient="index")
    out = json.loads(json.dumps(geojson))
    for feat in out["features"]:
        cid = feat["properties"].get("cell_id")
        n = lookup.get(cid)
        if not n:
            continue
        feat["properties"]["population_5km"] = int(round(n.get("population_5km", 0) or 0))
        feat["properties"]["pharmacies_5km"] = int(n.get("pharmacies_5km", 0) or 0)
        ratio = n.get("pop_per_pharmacy_5km")
        feat["properties"]["pop_per_pharmacy_5km"] = (
            f"1 : {int(ratio):,}" if pd.notna(ratio) else "No pharmacy in 5 km"
        )
        per_1k = n.get("pharmacies_per_1000_5km")
        feat["properties"]["pharmacies_per_1000_5km"] = (
            round(float(per_1k), 3) if pd.notna(per_1k) else 0.0
        )
    return out


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
    on_grid = (geography == grid_geo_key)
    if on_grid:
        # On the grid view the 5 km neighborhood metrics are the headline
        # signal — promote them above the per-cell variants and drop the
        # `per 100k` ratio (redundant at 1 km cell scale).
        metric_options = [
            "pop_per_pharmacy_5km", "pharmacies_per_1000_5km",
            "pop_per_pharmacy", "pharmacies_per_1000", "population",
        ]
    else:
        metric_options = [
            "pop_per_pharmacy", "pharmacies_per_1000", "pharmacies_per_100k", "population",
        ]
    metric_labels = {
        # On the grid the per-cell variants get the explicit "inside the cell"
        # qualifier so they read clearly next to the 5 km options.
        "pop_per_pharmacy": (
            "Population per Pharmacy inside the cell (lower = better)"
            if on_grid else "Population per Pharmacy (lower = better)"
        ),
        "pharmacies_per_1000": (
            "Pharmacies per 1,000 residents inside the cell"
            if on_grid else "Pharmacies per 1,000 residents"
        ),
        "pharmacies_per_100k": "Pharmacies per 100k",
        "population": "Cell Population" if on_grid else "Total Population",
        "pop_per_pharmacy_5km": "Population per Pharmacy within 5 km (lower = better)",
        "pharmacies_per_1000_5km": "Pharmacies per 1,000 residents within 5 km",
    }
    metric_choice = st.sidebar.radio(
        "Choropleth metric", metric_options,
        index=0 if on_grid else 1,
        format_func=lambda x: metric_labels[x],
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

    pharmacies_joined, _, _ = build_metrics(pharmacies_all, geo_ctx)
    pharmacies_joined = pharmacies_joined[
        pharmacies_joined["state"].isin(state_filter)
    ].copy()

    # Brand filter on the state-filtered pharmacy set.
    brand_options = sorted(pharmacies_joined["brand"].dropna().unique())
    selected_brands = st.sidebar.multiselect(
        "Brand / Chain", brand_options, default=brand_options,
    )
    pharmacies_f = pharmacies_joined[pharmacies_joined["brand"].isin(selected_brands)].copy()
    metrics, enriched_geojson = rebuild_metrics_from_joined_pharmacies(
        pharmacies_f, geo_ctx
    )

    # 5 km neighborhood metrics — computed only for the grid view.  The cell
    # population aggregate is cached per metro; the pharmacy-radius pass
    # re-runs on every brand-filter change.
    if on_grid:
        neighborhood = compute_neighborhood_metrics(
            metro_key=config["cache_key"],
            grid_path=config["grid_path"],
            pop_path=config["pop_path"],
            pharmacies=pharmacies_f,
        )
        metrics = metrics.merge(neighborhood, on="cell_id", how="left")
        # Cells outside the cached grid (shouldn't happen) get safe defaults.
        for col, fill in [("population_5km", 0.0), ("pharmacies_5km", 0)]:
            if col in metrics.columns:
                metrics[col] = metrics[col].fillna(fill)
        enriched_geojson = _inject_neighborhood_props(enriched_geojson, neighborhood)

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
    # "Population per Pharmacy" (own-cell or 5 km) is a higher-is-worse ratio
    # → red ramp; everything else is higher-is-better → green ramp.
    fill_color = (
        "YlOrRd"
        if metric_choice in ("pop_per_pharmacy", "pop_per_pharmacy_5km")
        else "YlGnBu"
    )
    # Reuse the metric-radio labels for the choropleth legend so the two
    # always agree (and inherit the on-grid "inside the cell" qualifiers).
    legend_names = {
        **metric_labels,
        "population": "Cell Population" if on_grid else "Population",
    }
    folium.Choropleth(
        geo_data=enriched_geojson, data=metrics,
        columns=[label_key, metric_choice],
        key_on=f"feature.properties.{label_key}",
        fill_color=fill_color,
        fill_opacity=0.65, line_opacity=0.4, nan_fill_color="lightgray",
        legend_name=legend_names[metric_choice],
        name="Choropleth", **kw,
    ).add_to(m)

    # Tooltip fields dynamic per geography. Order is: location info →
    # neighborhood (5 km) block on grid view → own-cell block. Putting 5 km
    # ahead of the cell totals keeps the headline access metric front-and-
    # centre when hovering.
    candidates = [label_key]
    if "district" in geo_ctx["join_keys"] and "district" not in candidates:
        candidates.append("district")
    if "state" in geo_ctx["join_keys"] and "state" not in candidates:
        candidates.append("state")
    if geography == grid_geo_key:
        candidates += ["parent_mukim", "district", "state"]
    if on_grid:
        candidates += [
            "population_5km", "pharmacies_5km",
            "pop_per_pharmacy_5km", "pharmacies_per_1000_5km",
        ]
    candidates += ["population", "pharmacy_count", "pop_per_pharmacy", "pharmacies_per_1000"]
    aliases = {
        "cell_id": "Grid Cell:", "parent_mukim": "Mukim:",
        "mukim": "Mukim:", "district": "District:", "state": "State:",
        "population": "Population:", "pharmacy_count": "Pharmacies:",
        "pop_per_pharmacy": "Pop / Pharmacy:",
        "pharmacies_per_1000": "Pharmacies / 1,000:",
        "population_5km": "Population within 5 km:",
        "pharmacies_5km": "Pharmacies within 5 km:",
        "pop_per_pharmacy_5km": "5 km Pop / Pharmacy:",
        "pharmacies_per_1000_5km": "5 km Pharmacies / 1,000:",
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

        if on_grid:
            # Surface zero-pharmacy cells FIRST (ranked by population within
            # 5 km), then served cells ranked by Population per Pharmacy
            # within 5 km.  The bar's x-axis uses the 5 km ratio; for
            # unserved cells we substitute population_5km so the bar still
            # carries scale information.
            df = metrics.copy()
            df["unserved_5km"] = df["pharmacies_5km"] == 0
            df["access_status"] = np.where(
                df["unserved_5km"],
                "No pharmacy in 5 km",
                "Has pharmacy in 5 km",
            )
            df["worst_access_value"] = np.where(
                df["unserved_5km"],
                df["population_5km"],
                df["pop_per_pharmacy_5km"],
            )
            chart_df = df.sort_values(
                ["unserved_5km", "worst_access_value"],
                ascending=[False, False],
            ).head(25)
            fig = px.bar(
                chart_df, x="worst_access_value", y=label_key,
                color="access_status",
                color_discrete_map={
                    "No pharmacy in 5 km": "#c62828",
                    "Has pharmacy in 5 km": "#ef6c00",
                },
                orientation="h", height=520,
                hover_data={
                    "population_5km": ":,.0f",
                    "pharmacies_5km": True,
                    "pop_per_pharmacy_5km": ":,.0f",
                    "worst_access_value": False,
                },
                labels={
                    "worst_access_value":
                        "Pop / Pharmacy (5 km) — or population if 0 pharmacies",
                    label_key: "",
                    "access_status": "",
                },
            )
            # Worst-access at the top: bar values mix two units (ratio for
            # served cells, raw population for unserved), so we can't rely on
            # `total ascending` — pin the y-axis order to our explicit ranking.
            fig.update_layout(
                yaxis={"categoryorder": "array",
                       "categoryarray": chart_df[label_key].tolist()[::-1]},
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=1.06),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Red bars are cells with **zero pharmacies within 5 km** — "
                "ranked by the population that has no nearby access."
            )
        else:
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
        if on_grid:
            sorted_metrics = metrics.assign(
                _unserved=metrics["pharmacies_5km"] == 0,
                _rank=np.where(
                    metrics["pharmacies_5km"] == 0,
                    metrics["population_5km"],
                    metrics["pop_per_pharmacy_5km"],
                ),
            ).sort_values(["_unserved", "_rank"], ascending=[False, False]) \
             .drop(columns=["_unserved", "_rank"])
            fmt = {
                "population": "{:,.0f}",
                "pharmacy_count": "{:,.0f}",
                "pop_per_pharmacy": "{:,.0f}",
                "pharmacies_per_100k": "{:.2f}",
                "pharmacies_per_1000": "{:.3f}",
                "population_5km": "{:,.0f}",
                "pharmacies_5km": "{:,.0f}",
                "pop_per_pharmacy_5km": "{:,.0f}",
                "pharmacies_per_1000_5km": "{:.3f}",
            }
        else:
            sorted_metrics = metrics.sort_values(
                "pop_per_pharmacy", ascending=False
            )
            fmt = {
                "population": "{:,.0f}",
                "pharmacy_count": "{:,.0f}",
                "pop_per_pharmacy": "{:,.0f}",
                "pharmacies_per_100k": "{:.2f}",
                "pharmacies_per_1000": "{:.3f}",
            }
        st.dataframe(
            sorted_metrics.reset_index(drop=True).style.format(fmt),
            use_container_width=True, height=520,
        )
