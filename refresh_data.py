"""
refresh_data.py
---------------
One-shot CLI that pulls live data from:

  * OpenStreetMap Overpass API    — retail pharmacies in Malaysia
  * DOSM storage.dosm.gov.my      — district-level population (latest year)
  * geoBoundaries (MYS ADM1/ADM2) — state + district polygons

...and caches the normalized results under ./data/ so the Streamlit app can
boot instantly and keep working if the network is unavailable.

Usage:
    python refresh_data.py                 # normal refresh, respects TTL
    python refresh_data.py --force         # bypass cache, re-fetch everything
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

from data_pipeline import (
    fetch_pharmacies_from_osm,
    load_population_district_dosm,
    load_malaysia_districts_geojson,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-cache live data for the dashboard.")
    parser.add_argument("--force", action="store_true",
                        help="Bypass TTL and re-download every source.")
    parser.add_argument("--data-dir", default="data",
                        help="Cache directory (default: ./data)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(f"→ Refreshing live data into {data_dir.resolve()} (force={args.force})")

    # 1. Pharmacies
    print("  [1/3] OSM Overpass: amenity=pharmacy in Malaysia...", flush=True)
    t0 = time.time()
    pharmacies = fetch_pharmacies_from_osm(
        cache_path=data_dir / "pharmacies_osm.json",
        force_refresh=args.force,
    )
    print(f"        {len(pharmacies):,} pharmacies  ({time.time() - t0:.1f}s)")

    # 2. Population
    print("  [2/3] DOSM: population_district CSV...", flush=True)
    t0 = time.time()
    population = load_population_district_dosm(
        cache_path=data_dir / "population_district.csv",
        force_refresh=args.force,
    )
    print(f"        {len(population):,} district rows  "
          f"(total pop {int(population['population'].sum()):,})  "
          f"({time.time() - t0:.1f}s)")

    # 3. Boundaries
    print("  [3/3] geoBoundaries: MYS ADM2 (+ ADM1 for state lookup)...", flush=True)
    t0 = time.time()
    geojson = load_malaysia_districts_geojson(
        cache_path=data_dir / "districts_my_adm2.geojson",
        force_refresh=args.force,
    )
    print(f"        {len(geojson['features']):,} district polygons  ({time.time() - t0:.1f}s)")

    # Manifest — handy when debugging "why are my numbers different today?"
    manifest = {
        "refreshed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pharmacies": len(pharmacies),
        "population_rows": len(population),
        "district_features": len(geojson["features"]),
        "elapsed_sec": round(time.time() - started, 1),
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"✓ Done in {manifest['elapsed_sec']}s. Manifest: {data_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
