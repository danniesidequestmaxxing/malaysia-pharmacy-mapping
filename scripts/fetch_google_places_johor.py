"""
fetch_google_places_johor.py
----------------------------
One-shot ingestion: pulls pharmacies in Johor from Google Places API (New)
and writes them to `data/pharmacies_google_johor.csv` in the schema the
rest of the pipeline expects.

Strategy:
  1. **Nearby sweep** — tile Johor's land area with overlapping ~5 km
     radius circles on a 5 km grid. For each circle, call
     `places:searchNearby` with `includedTypes=["pharmacy", "drugstore"]`.
     The new Places API returns up to 20 results per call (no
     pagination); the 5 km grid + 5 km radius means each pharmacy gets
     hit by ~4 overlapping circles, so even cells that cap at 20 still
     leave room for neighboring circles to catch the rest.
  2. **Chain text search** — for each known chain
     ("BIG Pharmacy", "Watsons", "Guardian", …), run
     `places:searchText` "{chain} Johor" to backfill outlets that the
     Nearby sweep might have missed in dense urban cells.

Dedup is deferred to `local_sources.merge_pharmacy_sources`, which
already runs a 2-pass (4 dp + brand-aware 3 dp) dedup against the union
of all pharmacy sources. This script only needs to dedup *within* its
own output (by Google `place_id`).

Run from the repo root:

    GOOGLE_MAPS_API_KEY=... python scripts/fetch_google_places_johor.py
    # or, with a .env file containing GOOGLE_MAPS_API_KEY=...
    python scripts/fetch_google_places_johor.py

Cost (Places API New, post-free-tier):
  * Nearby Search: $32 / 1000 calls. Johor sweep is ~700 circles → ~$22.
  * Text Search:   $40 / 1000 calls. ~30 chain queries → ~$1.2.

Required: Places API (New) must be enabled in your GCP project, on the
same key that's already used for Geocoding.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from data_pipeline import (
    load_malaysia_mukim_geojson, normalize_geojson_names,
)
from local_sources import _load_google_api_key


PLACES_API_BASE = "https://places.googleapis.com/v1"
NEARBY_URL = f"{PLACES_API_BASE}/places:searchNearby"
TEXT_URL = f"{PLACES_API_BASE}/places:searchText"

# Field mask: only ask for the fields we use, to keep cost in the cheaper
# "Pro" SKU and avoid pulling photo bytes etc.
FIELD_MASK = ",".join([
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.types",
    "places.businessStatus",
])

# Brand inference — name regex → canonical brand label used elsewhere in
# the app. First match wins; "Other" is the fallback (treated as
# independent by the chain/independent metric).
BRAND_RULES = [
    (r"\bbig\s*pharmacy\b",      "BIG Pharmacy"),
    (r"\bwatson",                 "Watsons"),
    (r"\bguardian",               "Guardian"),
    (r"\bcaring\s*pharmacy\b",   "Caring"),
    (r"\baa\s*pharmacy\b",       "AA Pharmacy"),
    (r"\bam\s*pm\b",              "AM PM"),
    (r"\balpro\b",                "Alpro"),
    (r"\bhealthlane\b",           "Healthlane"),
    (r"\bpmg\b",                  "PMG"),
    (r"\bsunway\s*multicare\b",  "Sunway Multicare"),
]

CHAIN_QUERIES = [
    "BIG Pharmacy Johor",
    "Watsons Johor",
    "Guardian Pharmacy Johor",
    "Caring Pharmacy Johor",
    "AA Pharmacy Johor",
    "AM PM Pharmacy Johor",
    "ALPRO Pharmacy Johor",
    "Healthlane Pharmacy Johor",
    "PMG Pharmacy Johor",
    "Sunway Multicare Johor",
]

OUTPUT_CSV = Path("data/pharmacies_google_johor.csv")
EXCLUDE_FILE = Path("data/pharmacies_google_johor_excluded.txt")


def _load_excludes(path: Path = EXCLUDE_FILE) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            out.add(stripped)
    return out


def _infer_brand(name: str) -> str:
    import re
    n = (name or "").lower()
    for pattern, brand in BRAND_RULES:
        if re.search(pattern, n):
            return brand
    return "Other"


def _johor_land_polygon():
    gj = normalize_geojson_names(load_malaysia_mukim_geojson())
    polys = [shape(f["geometry"]) for f in gj["features"]
             if f["properties"].get("state") == "Johor"]
    if not polys:
        raise SystemExit("No Johor mukim polygons found in data/mukim_my_adm3.geojson")
    return unary_union(polys)


def _grid_search_points(land_poly, spacing_km: float) -> list[tuple[float, float]]:
    """Lon/lat grid clipped to the Johor land polygon. ~spacing_km spacing
    in both directions (using equirectangular approx at Johor's latitude)."""
    minx, miny, maxx, maxy = land_poly.bounds
    deg_per_km_lat = 1 / 111.32
    deg_per_km_lon = 1 / (111.32 * 0.999)  # close enough at ~2°N

    step_lat = spacing_km * deg_per_km_lat
    step_lon = spacing_km * deg_per_km_lon

    points = []
    y = miny
    while y <= maxy:
        x = minx
        while x <= maxx:
            if land_poly.intersects(Point(x, y)):
                points.append((y, x))   # (lat, lon)
            x += step_lon
        y += step_lat
    return points


def _nearby_search(api_key: str, lat: float, lon: float,
                    radius_m: float) -> list[dict]:
    body = {
        "includedTypes": ["pharmacy", "drugstore"],
        "maxResultCount": 20,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": radius_m,
            }
        },
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": FIELD_MASK,
    }
    r = requests.post(NEARBY_URL, json=body, headers=headers, timeout=30)
    if r.status_code != 200:
        sys.stderr.write(f"  [warn] {r.status_code} {r.text[:200]}\n")
        return []
    return (r.json() or {}).get("places", []) or []


def _text_search(api_key: str, query: str,
                  region_bias: tuple[float, float, float, float] | None = None
                  ) -> list[dict]:
    body: dict = {"textQuery": query, "includedType": "pharmacy", "pageSize": 20}
    if region_bias:
        miny, minx, maxy, maxx = region_bias
        body["locationBias"] = {
            "rectangle": {
                "low": {"latitude": miny, "longitude": minx},
                "high": {"latitude": maxy, "longitude": maxx},
            }
        }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": FIELD_MASK + ",nextPageToken",
    }
    out: list[dict] = []
    next_token: str | None = None
    for _ in range(3):  # up to 3 pages = 60 results
        if next_token:
            body["pageToken"] = next_token
            time.sleep(2)  # nextPageToken needs a brief delay
        r = requests.post(TEXT_URL, json=body, headers=headers, timeout=30)
        if r.status_code != 200:
            sys.stderr.write(f"  [warn] textSearch {r.status_code} {r.text[:200]}\n")
            break
        data = r.json() or {}
        out.extend(data.get("places", []) or [])
        next_token = data.get("nextPageToken")
        if not next_token:
            break
    return out


def _normalize_place(p: dict) -> dict | None:
    pid = p.get("id")
    name = (p.get("displayName") or {}).get("text") or ""
    addr = p.get("formattedAddress") or ""
    loc = p.get("location") or {}
    lat = loc.get("latitude")
    lon = loc.get("longitude")
    if not pid or lat is None or lon is None or not name:
        return None
    if (p.get("businessStatus") or "OPERATIONAL") != "OPERATIONAL":
        return None
    return {
        "pharmacy_id": pid,
        "name": name,
        "brand": _infer_brand(name),
        "address": addr,
        "latitude": float(lat),
        "longitude": float(lon),
        "source": "google_places",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacing-km", type=float, default=5.0,
                        help="grid spacing for the nearby sweep (default 5)")
    parser.add_argument("--radius-km", type=float, default=5.0,
                        help="circle radius for the nearby sweep (default 5)")
    parser.add_argument("--skip-nearby", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    api_key = _load_google_api_key()
    if not api_key:
        sys.exit("ERROR: GOOGLE_MAPS_API_KEY not set (env var or .env file).")

    land = _johor_land_polygon()
    minx, miny, maxx, maxy = land.bounds
    region_bias = (miny, minx, maxy, maxx)

    excludes = _load_excludes()
    if excludes:
        print(f"Loaded {len(excludes)} exclude IDs from {EXCLUDE_FILE}")

    by_id: dict[str, dict] = {}

    if not args.skip_nearby:
        points = _grid_search_points(land, args.spacing_km)
        print(f"Phase 1: nearbySearch — {len(points)} circles "
              f"({args.radius_km:.0f} km radius)")
        for i, (lat, lon) in enumerate(points, 1):
            results = _nearby_search(api_key, lat, lon, args.radius_km * 1000)
            new = 0
            for p in results:
                rec = _normalize_place(p)
                if rec and rec["pharmacy_id"] not in by_id \
                        and rec["pharmacy_id"] not in excludes:
                    by_id[rec["pharmacy_id"]] = rec
                    new += 1
            if i % 25 == 0 or i == len(points):
                print(f"  {i}/{len(points)} circles processed, "
                      f"{len(by_id)} unique pharmacies so far "
                      f"(+{new} this circle)")

    if not args.skip_text:
        print(f"Phase 2: textSearch — {len(CHAIN_QUERIES)} chain queries")
        for q in CHAIN_QUERIES:
            results = _text_search(api_key, q, region_bias=region_bias)
            new = 0
            for p in results:
                rec = _normalize_place(p)
                if rec and rec["pharmacy_id"] not in by_id \
                        and rec["pharmacy_id"] not in excludes:
                    by_id[rec["pharmacy_id"]] = rec
                    new += 1
            print(f"  '{q}': {len(results)} results, +{new} new")

    rows = sorted(by_id.values(), key=lambda r: r["name"].lower())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "pharmacy_id", "name", "brand", "address",
            "latitude", "longitude", "source",
        ])
        writer.writeheader()
        writer.writerows(rows)

    by_brand: dict[str, int] = {}
    for r in rows:
        by_brand[r["brand"]] = by_brand.get(r["brand"], 0) + 1
    print()
    print(f"Wrote {args.output}: {len(rows)} unique pharmacies")
    for b, c in sorted(by_brand.items(), key=lambda x: -x[1]):
        print(f"  {b}: {c}")


if __name__ == "__main__":
    main()
