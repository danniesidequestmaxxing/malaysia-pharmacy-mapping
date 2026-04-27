"""
build_johor_full_grid.py
------------------------
Extends the committed 8-mukim Johor 1 km grid to cover the entire state.

  * Reuses the existing high-resolution 1 km WorldPop in the JB / Tebrau /
    Plentong / Pulai / Kota Tinggi / Senai / Kulai belt verbatim.
  * For the remaining 100+ Johor mukim, builds new 1 km cells and assigns
    population by uniformly distributing each mukim's total (from
    `data/worldpop_per_mukim.csv`) across its cells. Approximate, but
    every cell ends up with a non-zero, plausibly-shaped population so
    the 5 km neighborhood metrics work state-wide.

Outputs:
  * data/submukim_grid_johor_full.geojson
  * data/worldpop_per_submukim_johor_full_1km.csv

Run from the repo root:
    python scripts/build_johor_full_grid.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from shapely.geometry import box, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from data_pipeline import (
    load_malaysia_mukim_geojson, normalize_geojson_names,
)


URBAN_MUKIM = (
    "Mukim Tebrau", "Mukim Plentong", "Mukim Pulai",
    "Mukim Kota Tinggi", "Mukim Senai", "Mukim Kulai",
    "Bandar Johor Bahru", "Bandar Kulai",
)
URBAN_MUKIM_LOWER = {m.lower() for m in URBAN_MUKIM}

CELL_DEG = 0.009  # ≈ 1 km at the equator

EXISTING_GRID_PATH = Path("data/submukim_grid_johor.geojson")
EXISTING_POP_PATH = Path("data/worldpop_per_submukim_v3_8mukim_1km.csv")
PER_MUKIM_POP_PATH = Path("data/worldpop_per_mukim.csv")

OUT_GRID_PATH = Path("data/submukim_grid_johor_full.geojson")
OUT_POP_PATH = Path("data/worldpop_per_submukim_johor_full_1km.csv")


def _canonical_for_match(name: str) -> str:
    """Lowercase + collapse whitespace so 'BANDAR JOHOR BAHRU' and
    'Bandar Johor Bahru' collide in lookups."""
    return " ".join((name or "").lower().split())


def _select_non_urban_johor_mukim(mukim_gj: dict):
    polys, props = [], []
    for f in mukim_gj["features"]:
        p = f["properties"]
        if p.get("state") != "Johor":
            continue
        if (p.get("mukim") or "").lower() in URBAN_MUKIM_LOWER:
            continue
        polys.append(shape(f["geometry"]))
        props.append(p)
    return polys, props


def _build_grid_features(polys, props, start_id: int) -> list[dict]:
    """1 km grid clipped to the union of `polys`, with cell_ids starting
    at `start_id` so they don't collide with the existing urban belt."""
    if not polys:
        return []
    union = unary_union(polys)
    tree = STRtree(polys)
    minx, miny, maxx, maxy = union.bounds

    features: list[dict] = []
    cell_id = start_id
    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            cell = box(x, y, x + CELL_DEG, y + CELL_DEG)
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
            x += CELL_DEG
        y += CELL_DEG
    return features


def _approximate_pop_by_uniform_split(features: list[dict],
                                       per_mukim_pop: pd.DataFrame) -> pd.DataFrame:
    """For each new cell, look up its parent mukim's total population in
    `per_mukim_pop` and divide evenly by the number of new cells in that
    mukim. Mukim that don't appear in the per-mukim CSV (typically ones
    with name-canonicalisation drift between the geojson and the CSV)
    fall back to 0."""
    pop_lookup: dict[str, float] = {}
    for _, row in per_mukim_pop[per_mukim_pop["state"] == "Johor"].iterrows():
        pop_lookup[_canonical_for_match(row["mukim"])] = float(row["population"] or 0)

    rows: list[dict] = []
    cells_per_mukim: dict[str, int] = {}
    for f in features:
        m = _canonical_for_match(f["properties"].get("parent_mukim") or "")
        cells_per_mukim[m] = cells_per_mukim.get(m, 0) + 1

    matched, missing = set(), set()
    for f in features:
        p = f["properties"]
        m_lower = _canonical_for_match(p.get("parent_mukim") or "")
        total_pop = pop_lookup.get(m_lower)
        if total_pop is None:
            missing.add(p.get("parent_mukim"))
            cell_pop = 0.0
        else:
            matched.add(p.get("parent_mukim"))
            cell_pop = round(total_pop / cells_per_mukim[m_lower])
        rows.append({
            "cell_id": p["cell_id"],
            "parent_mukim": p.get("parent_mukim"),
            "district": p.get("district"),
            "state": p.get("state"),
            "population": int(cell_pop),
        })

    print(f"  Mukim matched in per-mukim pop CSV: {len(matched)}")
    if missing:
        print(f"  Mukim with no per-mukim pop entry ({len(missing)}, will be 0): {sorted(missing)[:10]}")
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading existing 8-mukim grid + pop...")
    existing_grid = json.loads(EXISTING_GRID_PATH.read_text(encoding="utf-8"))
    existing_pop = pd.read_csv(EXISTING_POP_PATH)
    print(f"  {len(existing_grid['features'])} cells, {existing_pop['population'].sum():,.0f} total pop")

    next_cell_id = 0
    for f in existing_grid["features"]:
        cid = int(f["properties"]["cell_id"][1:])
        next_cell_id = max(next_cell_id, cid + 1)
    print(f"  Next available cell_id: G{next_cell_id:04d}")

    print("Loading + normalizing Malaysia mukim geojson...")
    mukim_gj = normalize_geojson_names(load_malaysia_mukim_geojson())
    polys, props = _select_non_urban_johor_mukim(mukim_gj)
    print(f"  {len(polys)} non-urban Johor mukim polygons selected")

    print(f"Building 1 km grid for the remaining state ({CELL_DEG}° ≈ 1 km cells)...")
    new_features = _build_grid_features(polys, props, start_id=next_cell_id)
    print(f"  {len(new_features)} new cells generated")

    print("Computing approximate populations from per-mukim aggregate...")
    per_mukim = pd.read_csv(PER_MUKIM_POP_PATH)
    new_pop = _approximate_pop_by_uniform_split(new_features, per_mukim)
    print(f"  New-cell population total: {new_pop['population'].sum():,.0f}")

    combined_features = list(existing_grid["features"]) + new_features
    combined_pop = pd.concat([existing_pop, new_pop], ignore_index=True)

    out_grid = {"type": "FeatureCollection", "features": combined_features}
    OUT_GRID_PATH.write_text(json.dumps(out_grid), encoding="utf-8")
    combined_pop.to_csv(OUT_POP_PATH, index=False)
    print()
    print(f"Wrote {OUT_GRID_PATH}: {len(combined_features):,} cells")
    print(f"Wrote {OUT_POP_PATH}: total pop = {combined_pop['population'].sum():,.0f}")


if __name__ == "__main__":
    main()
