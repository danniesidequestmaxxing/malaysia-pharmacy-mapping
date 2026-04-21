"""
geocode_pmg.py
--------------
Parse the PMG "outlets" Excel (branch-level sheet), filter to the Pharmacy
segment, and geocode each branch via Google.  Shares the same cache file
as geocode_npra.py so overlapping addresses (a branch already geocoded
from another source) are never re-fetched.

Usage:
    python geocode_pmg.py                    # default: google, all 187 branches
    python geocode_pmg.py --max 50           # stop after 50 new calls
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from local_sources import parse_pmg_excel, geocode_addresses_google, _load_google_api_key


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/source/PMG outlets (Pharmacy, Medical & Dental Clinics).xlsx")
    ap.add_argument("--cache", default="data/geocode_cache_google.json")
    ap.add_argument("--out", default="data/pharmacies_pmg_geocoded.csv")
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()

    if not _load_google_api_key():
        print("ERROR: GOOGLE_MAPS_API_KEY not set (see .env.example).")
        return 2

    print(f"→ Parsing {args.xlsx} ...")
    df = parse_pmg_excel(args.xlsx)
    print(f"  {len(df):,} pharmacy-segment branches to geocode")
    print(f"  by brand:\n{df['brand'].value_counts().to_string()}")

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
