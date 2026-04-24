"""
geocode_scraped.py
------------------
Fill in missing latitude/longitude on a web-scraped store CSV (Watsons,
Guardian, etc.) using the same Google Geocoding pipeline we use for NPRA
and PMG.  Rows that already have coordinates pass through untouched.

Usage:
    python geocode_scraped.py data/pharmacies_watsons.csv
    python geocode_scraped.py data/pharmacies_guardian.csv --max 200
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from local_sources import geocode_addresses_google, _load_google_api_key


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to the scraped CSV (overwritten in place).")
    ap.add_argument("--cache", default="data/geocode_cache_google.json")
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()

    if not _load_google_api_key():
        print("ERROR: GOOGLE_MAPS_API_KEY not set (see .env.example).")
        return 2

    path = Path(args.csv)
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for alias, canon in [("lat", "latitude"), ("lng", "longitude"),
                          ("long", "longitude"), ("lon", "longitude")]:
        if alias in df.columns and canon not in df.columns:
            df[canon] = df[alias]
    for c in ["latitude", "longitude", "address", "state", "name"]:
        if c not in df.columns:
            df[c] = ""

    need = df["latitude"].isna() | (df["latitude"] == "")
    print(f"→ {int(need.sum())}/{len(df)} rows need geocoding")
    if need.sum() == 0:
        print("  nothing to do.")
        return 0

    targets = df.loc[need].copy()
    t0 = time.time()
    geo = geocode_addresses_google(
        targets, cache_path=args.cache, max_requests=args.max
    )
    done = int(geo["latitude"].notna().sum())
    print(f"→ Geocoded {done}/{len(geo)} in {time.time() - t0:.1f}s")

    df.loc[need, "latitude"] = geo["latitude"].values
    df.loc[need, "longitude"] = geo["longitude"].values
    df.to_csv(path, index=False)
    print(f"→ Saved back to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
