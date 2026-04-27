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

# Brands counted as "chain" pharmacies for the chain/independent split.
# Anything else (kedai ubat, hospital pharmacies, single-outlet operators)
# is treated as independent.
CHAIN_BRANDS = frozenset({
    "AA Pharmacy", "AM PM", "Alpro", "BIG Pharmacy", "Caring",
    "Guardian", "Guardian Retail", "Healthlane", "PMG",
    "Sunway Multicare", "Watsons",
})


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
                            ) -> pd.DataFrame:
    """Per-cell counts of total / chain / independent pharmacies within
    `radius_km`. Re-runs on every brand filter change — fast enough that
    caching is unnecessary."""
    cols = ["cell_id", "pharmacies_5km", "chain_5km", "independent_5km"]
    if centroids.empty:
        return pd.DataFrame(columns=cols)
    valid = pharmacies.dropna(subset=["latitude", "longitude"])
    n = len(centroids)
    if valid.empty:
        return pd.DataFrame({
            "cell_id": centroids["cell_id"].to_numpy(),
            "pharmacies_5km": 0, "chain_5km": 0, "independent_5km": 0,
        })

    is_chain = valid["brand"].fillna("").isin(CHAIN_BRANDS).to_numpy()

    ref_lat_rad = float(np.deg2rad(centroids["lat"].mean()))
    cell_xy = _project_xy(centroids["lon"].to_numpy(),
                           centroids["lat"].to_numpy(), ref_lat_rad)
    pharm_xy = _project_xy(valid["longitude"].to_numpy(),
                            valid["latitude"].to_numpy(), ref_lat_rad)

    total = np.zeros(n, dtype=int)
    chain = np.zeros(n, dtype=int)
    r2 = float(radius_km) ** 2
    chunk = 500
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dx = cell_xy[start:end, 0:1] - pharm_xy[None, :, 0]
        dy = cell_xy[start:end, 1:2] - pharm_xy[None, :, 1]
        mask = (dx * dx + dy * dy) <= r2
        total[start:end] = mask.sum(axis=1)
        chain[start:end] = (mask & is_chain[None, :]).sum(axis=1)

    return pd.DataFrame({
        "cell_id": centroids["cell_id"].to_numpy(),
        "pharmacies_5km": total,
        "chain_5km": chain,
        "independent_5km": total - chain,
    })


def compute_neighborhood_metrics(metro_key: str, grid_path: str, pop_path: str,
                                  pharmacies: pd.DataFrame,
                                  radius_km: float = NEIGHBORHOOD_RADIUS_KM
                                  ) -> pd.DataFrame:
    """Build the per-cell 5 km neighborhood frame.

    Output columns: `cell_id, population_5km, pharmacies_5km, chain_5km,
    independent_5km, pop_per_pharmacy_5km, pharmacies_per_1000_5km`.
    """
    pop_df = _cached_population_5km(metro_key, grid_path, pop_path, radius_km)
    if pop_df.empty:
        return pd.DataFrame(columns=[
            "cell_id", "population_5km", "pharmacies_5km",
            "chain_5km", "independent_5km",
            "pop_per_pharmacy_5km", "pharmacies_per_1000_5km",
        ])

    pharm_counts = _pharmacies_within_5km(
        pop_df[["cell_id", "lon", "lat"]], pharmacies, radius_km
    )
    out = pop_df[["cell_id", "population_5km"]].merge(
        pharm_counts, on="cell_id", how="left"
    )
    for col in ("pharmacies_5km", "chain_5km", "independent_5km"):
        out[col] = out[col].fillna(0).astype(int)
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
        chain = int(n.get("chain_5km", 0) or 0)
        indep = int(n.get("independent_5km", 0) or 0)
        feat["properties"]["chain_independent_5km"] = (
            f"{chain} chain : {indep} indep"
            if (chain + indep) > 0 else "No pharmacy in 5 km"
        )
        feat["properties"]["chain_share_5km"] = (
            f"{round(chain / (chain + indep) * 100)}%"
            if (chain + indep) > 0 else "—"
        )
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
# Grid-cell ranker: interactive top-N by metric, with row selection + CSV export.
# --------------------------------------------------------------------------------------

# Cells with population below this threshold are excluded from the ranker —
# they're already masked gray on the choropleth and only inflate "worst
# access" lists artificially (a 1 km cell with 12 residents tells us nothing
# about 5 km neighborhood access).
_RANKER_MIN_CELL_POP = 100

# Metric → (display label, plotly tick format, dataframe format string).
_RANKER_METRICS = {
    "population_5km":          ("Population within 5 km",        ",.0f", "{:,.0f}"),
    "pharmacies_5km":          ("Pharmacies within 5 km",        ",.0f", "{:,.0f}"),
    "chain_share_5km":         ("5 km Chain Share (%)",          ".0f",  "{:.0f}%"),
    "pop_per_pharmacy_5km":    ("5 km Pop / Pharmacy",           ",.0f", "{:,.0f}"),
    "pharmacies_per_1000_5km": ("5 km Pharmacies / 1,000",       ".3f",  "{:.3f}"),
    "population":              ("Cell Population",               ",.0f", "{:,.0f}"),
}

# Columns shown in the ranker table (in this order). Anything missing from the
# metrics frame is silently skipped.
_RANKER_TABLE_COLS = [
    "cell_id", "parent_mukim", "district",
    "population", "population_5km",
    "pharmacies_5km", "chain_5km", "independent_5km", "chain_share_5km",
    "pop_per_pharmacy_5km", "pharmacies_per_1000_5km",
]


def _render_grid_ranker(metrics: pd.DataFrame, label_key: str, config: dict,
                         focus_state_key: str | None = None,
                         map_html_provider=None) -> None:
    """Top-N grid cells by chosen metric, with row selection, "show on map"
    focus mode, and CSV / HTML-map exports.

    Parameters
    ----------
    focus_state_key
        Streamlit session-state key that holds the list of cell_ids the map
        is focused on. Buttons in this section read / write it.
    map_html_provider
        Zero-arg callable returning the standalone Folium HTML for the
        currently rendered map. Called lazily — only if the user clicks
        the map download button.
    """
    st.subheader("📊 Grid cells — pick top N by metric")

    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 2])
    metric_keys = [k for k in _RANKER_METRICS if k in metrics.columns]
    rank_by = ctrl1.selectbox(
        "Rank by", metric_keys,
        format_func=lambda k: _RANKER_METRICS[k][0],
        index=metric_keys.index("population_5km") if "population_5km" in metric_keys else 0,
        key="grid_rank_by",
    )
    direction = ctrl2.radio(
        "Show", ["Most", "Least"], horizontal=True, key="grid_direction",
    )
    with ctrl3:
        rng1, rng2 = st.columns(2)
        rank_min = int(rng1.number_input(
            "From rank #", min_value=1, value=1, step=1, key="grid_rank_min",
        ))
        rank_max = int(rng2.number_input(
            "To rank #", min_value=1, value=20, step=1, key="grid_rank_max",
        ))
    if rank_max < rank_min:
        rank_max = rank_min

    excluded_low_pop = int((metrics["population"] < _RANKER_MIN_CELL_POP).sum())
    eligible = metrics[metrics["population"] >= _RANKER_MIN_CELL_POP].copy()
    sorted_eligible = eligible.dropna(subset=[rank_by]).sort_values(
        rank_by, ascending=(direction == "Least")
    ).reset_index(drop=True)
    ranked = sorted_eligible.iloc[rank_min - 1: rank_max].reset_index(drop=True)

    eligible_total = len(sorted_eligible)
    caption_bits = []
    if excluded_low_pop:
        caption_bits.append(
            f"Excluding {excluded_low_pop:,} cells with population < "
            f"{_RANKER_MIN_CELL_POP}."
        )
    caption_bits.append(
        f"{eligible_total:,} eligible cells in the ranking; "
        f"showing ranks {rank_min:,}–{min(rank_max, eligible_total):,}."
    )
    st.caption(" ".join(caption_bits))
    if ranked.empty:
        st.info(
            f"No cells in this rank range. The eligible pool has "
            f"{eligible_total:,} rows — adjust **From rank** / **To rank** "
            "to land within it."
        )
        return

    label, tickfmt, _ = _RANKER_METRICS[rank_by]

    left, right = st.columns([3, 2])
    with left:
        fig = px.bar(
            ranked, x=rank_by, y=label_key,
            orientation="h", height=540,
            hover_data={
                "parent_mukim": True, "district": True,
                "population": ":,.0f", "population_5km": ":,.0f",
                "pharmacies_5km": True, "chain_5km": True,
                "independent_5km": True,
            },
            labels={rank_by: label, label_key: ""},
        )
        fig.update_layout(
            yaxis={
                "categoryorder": "array",
                "categoryarray": ranked[label_key].tolist()[::-1],
            },
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.update_xaxes(tickformat=tickfmt)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Ranks {rank_min:,}–{rank_min + len(ranked) - 1:,} by **{label}** "
            f"({'highest' if direction == 'Most' else 'lowest'} first)."
        )

    with right:
        st.subheader("📋 Ranked cells — click rows to select")
        cols = [c for c in _RANKER_TABLE_COLS if c in ranked.columns]
        fmt = {
            "population": "{:,.0f}", "population_5km": "{:,.0f}",
            "pharmacies_5km": "{:,.0f}",
            "chain_5km": "{:,.0f}", "independent_5km": "{:,.0f}",
            "chain_share_5km": "{:.0f}%",
            "pop_per_pharmacy_5km": "{:,.0f}",
            "pharmacies_per_1000_5km": "{:.3f}",
        }
        # Reset row selection when the ranker params change — otherwise stale
        # row indices point at different cells under the new ranking.
        table_key = (
            f"grid_ranker_table__{rank_by}__{direction}"
            f"__{rank_min}__{rank_max}"
        )
        try:
            event = st.dataframe(
                ranked[cols].style.format(fmt, na_rep="—"),
                use_container_width=True, height=540, hide_index=True,
                on_select="rerun", selection_mode="multi-row",
                key=table_key,
            )
            chosen_rows = list(event.selection.rows) if event and event.selection else []
        except TypeError:
            # Streamlit < 1.35 — fall back to a non-selectable table.
            st.dataframe(
                ranked[cols].style.format(fmt, na_rep="—"),
                use_container_width=True, height=540, hide_index=True,
            )
            chosen_rows = []

    chosen_cell_ids = (
        ranked.iloc[chosen_rows]["cell_id"].tolist() if chosen_rows else []
    )
    focus_active = bool(
        focus_state_key and st.session_state.get(focus_state_key)
    )

    # ---- Action buttons: focus the map on the selection / reset to default ----
    st.markdown("---")
    btn1, btn2, _spacer = st.columns([2, 2, 6])
    if btn1.button(
        f"🎯 Show {len(chosen_cell_ids)} selected on map",
        disabled=not chosen_cell_ids or focus_state_key is None,
        key="grid_show_on_map",
        use_container_width=True,
    ):
        st.session_state[focus_state_key] = chosen_cell_ids
        st.rerun()
    if btn2.button(
        "↺ Reset map",
        disabled=not focus_active,
        key="grid_reset_map",
        use_container_width=True,
    ):
        st.session_state.pop(focus_state_key, None)
        st.rerun()

    # ---- Downloads: CSV + HTML map ----
    export_df = ranked.iloc[chosen_rows] if chosen_rows else ranked
    csv_label = (
        f"📥 Download {len(export_df)} selected cell(s) — CSV"
        if chosen_rows else
        f"📥 Download all {len(export_df)} ranked cells — CSV"
    )
    dl1, dl2 = st.columns(2)
    dl1.download_button(
        label=csv_label,
        data=export_df[cols].to_csv(index=False).encode("utf-8"),
        file_name=(
            f"{config['cache_key']}_{rank_by}_{direction.lower()}"
            f"_rank{rank_min}-{rank_max}.csv"
        ),
        mime="text/csv",
        key="grid_ranker_csv",
        use_container_width=True,
    )
    if map_html_provider is not None:
        suffix = "focus" if focus_active else "all"
        dl2.download_button(
            label=f"🗺️ Download current map — HTML ({suffix})",
            data=map_html_provider(),
            file_name=f"{config['cache_key']}_map_{suffix}.html",
            mime="text/html",
            key="grid_ranker_map_html",
            use_container_width=True,
        )


def _render_polygon_table(metrics: pd.DataFrame, label_key: str,
                           scope_label: str) -> None:
    """Original district / mukim view: bar chart on the left, table on the right."""
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
        sorted_metrics = metrics.sort_values("pop_per_pharmacy", ascending=False)
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
        for col, fill in [
            ("population_5km", 0.0), ("pharmacies_5km", 0),
            ("chain_5km", 0), ("independent_5km", 0),
        ]:
            if col in metrics.columns:
                metrics[col] = metrics[col].fillna(fill)
        # Numeric chain-share column for ranking / table — the geojson props
        # carry the formatted string version for tooltips.
        metrics["chain_share_5km"] = np.where(
            metrics["pharmacies_5km"] > 0,
            metrics["chain_5km"] / metrics["pharmacies_5km"] * 100,
            np.nan,
        )
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

    # Focus mode: when the user picks cells in the grid ranker and clicks
    # "Show selected on map", we filter the geojson down to just those
    # cells so the choropleth + tooltip + gray mask all render as a focused
    # subset. Pharmacy markers are kept for context.
    focus_state_key = f"{config['cache_key']}_focus_cells"
    focus_cells = st.session_state.get(focus_state_key) if on_grid else None
    if focus_cells:
        focus_set = set(focus_cells)
        enriched_geojson = {
            "type": "FeatureCollection",
            "features": [
                f for f in enriched_geojson["features"]
                if f["properties"].get("cell_id") in focus_set
            ],
        }
        st.info(
            f"🎯 Map focused on **{len(focus_cells)} cell(s)** picked from the "
            "ranker below. Use the **Reset map** button to show all cells."
        )

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
    # All ratio choropleths use the reversed RdYlGn ramp so the legend reads
    # green→red left-to-right. Lower-is-better metrics (pop_per_pharmacy*) get
    # green at the low end naturally; pharmacies_per_1000* is plotted in the
    # same direction so the two ratio metrics have matching legends. Magnitude
    # metrics (population) stay on a neutral sequential ramp.
    lower_is_better = metric_choice in ("pop_per_pharmacy", "pop_per_pharmacy_5km")
    ratio_metric = metric_choice in (
        "pop_per_pharmacy", "pop_per_pharmacy_5km",
        "pharmacies_per_1000", "pharmacies_per_1000_5km",
    )
    if metric_choice == "population":
        fill_color = "YlGnBu"
    elif ratio_metric:
        fill_color = "RdYlGn_r"
    else:
        fill_color = "RdYlGn"
    # Cells with no pharmacy inside the 5 km radius are NaN on
    # `pop_per_pharmacy_5km` — they're the *worst* access, not missing data, so
    # render them in deep red rather than neutral gray.
    nan_fill_color = "#67001f" if lower_is_better else "lightgray"
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
        fill_opacity=0.65, line_opacity=0.4, nan_fill_color=nan_fill_color,
        legend_name=legend_names[metric_choice],
        name="Choropleth", **kw,
    ).add_to(m)

    # Mask cells whose own population is below the noise threshold — at 1 km
    # resolution clipped fragments of mukim boundaries can land with single-
    # digit populations that produce wildly skewed ratios. Render them gray
    # over the choropleth so they read as "not enough signal" rather than as
    # genuine outliers.
    LOW_POP_THRESHOLD = 100
    low_pop_features = [
        f for f in enriched_geojson["features"]
        if (f["properties"].get("population") or 0) < LOW_POP_THRESHOLD
    ]
    if low_pop_features:
        folium.GeoJson(
            {"type": "FeatureCollection", "features": low_pop_features},
            name="Low-population cells (masked)",
            style_function=lambda _: {
                "fillColor": "#b0b0b0", "fillOpacity": 1.0,
                "color": "#7a7a7a", "weight": 0.4,
            },
            control=False,
        ).add_to(m)

    # Tooltip fields dynamic per geography. On the grid view we drop state
    # (implied by the page focus) and the cell-level pharmacy metrics
    # (redundant with the 5 km block); only the cell `population` stays so
    # the user can sanity-check the 5 km neighborhood signal.
    candidates = [label_key]
    if "district" in geo_ctx["join_keys"] and "district" not in candidates:
        candidates.append("district")
    if not on_grid and "state" in geo_ctx["join_keys"] and "state" not in candidates:
        candidates.append("state")
    if geography == grid_geo_key:
        candidates.append("parent_mukim")
        candidates.append("district")
    if on_grid:
        candidates += [
            "population_5km", "pharmacies_5km",
            "chain_independent_5km", "chain_share_5km",
            "pop_per_pharmacy_5km", "pharmacies_per_1000_5km",
            "population",
        ]
    else:
        candidates += [
            "population", "pharmacy_count",
            "pop_per_pharmacy", "pharmacies_per_1000",
        ]
    aliases = {
        "cell_id": "Grid Cell:", "parent_mukim": "Mukim:",
        "mukim": "Mukim:", "district": "District:", "state": "State:",
        "population": "Population:", "pharmacy_count": "Pharmacies:",
        "pop_per_pharmacy": "Pop / Pharmacy:",
        "pharmacies_per_1000": "Pharmacies / 1,000:",
        "population_5km": "Population within 5 km:",
        "pharmacies_5km": "Pharmacies within 5 km:",
        "chain_independent_5km": "5 km Chain : Independent:",
        "chain_share_5km": "5 km Chain Share:",
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

    # Stack under the zoom buttons (top-left) so it doesn't collide with the
    # choropleth legend, which Folium pins to the top-right corner.
    folium.LayerControl(collapsed=False, position="topleft").add_to(m)
    st_folium(m, height=640, use_container_width=True, returned_objects=[])

    # ---- Table + chart ----
    if on_grid:
        # Render the standalone HTML once, lazily — it's only shipped via the
        # download button and we don't want to pay for it when the user never
        # exports the map.
        def _render_map_html() -> bytes:
            return m.get_root().render().encode("utf-8")
        _render_grid_ranker(
            metrics, label_key, config,
            focus_state_key=focus_state_key,
            map_html_provider=_render_map_html,
        )
    else:
        _render_polygon_table(metrics, label_key, scope_label)
