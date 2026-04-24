"""
geocode_watsons.py
------------------
Parse the Watsons Malaysia outlets Excel and Google-geocode all ~850
branches.  Shares the cache with NPRA + PMG so overlapping queries are
instant.

Usage:
    python geocode_watsons.py
    python geocode_watsons.py --max 200        # rate-cap one batch
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from local_sources import parse_watsons_excel, geocode_addresses_google, _load_google_api_key


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/source/Watson_Outlets_Malaysia.xlsx")
    ap.add_argument("--cache", default="data/geocode_cache_google.json")
    ap.add_argument("--out", default="data/pharmacies_watsons.csv")
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()

    if not _load_google_api_key():
        print("ERROR: GOOGLE_MAPS_API_KEY not set (see .env.example).")
        return 2

    print(f"→ Parsing {args.xlsx} ...")
    df = parse_watsons_excel(args.xlsx)
    print(f"  {len(df):,} Watsons outlets")
    print(f"  by state (top 6):\n{df['state'].value_counts().head(6).to_string()}")

    t0 = time.time()
    geo = geocode_addresses_google(df, cache_path=args.cache, max_requests=args.max)
    done = int(geo["latitude"].notna().sum())
    print(f"→ Geocoded {done}/{len(geo)} ({done / len(geo) * 100:.1f}%) "
          f"in {time.time() - t0:.1f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    geo.to_csv(args.out, index=False)
    print(f"→ Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
