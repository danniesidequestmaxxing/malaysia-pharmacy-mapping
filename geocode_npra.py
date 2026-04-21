"""
geocode_npra.py
---------------
Parse the NPRA 'Senarai Premis' PDF, dedupe addresses, and geocode them.

Two providers:
  * --provider google      Google Geocoding API (requires GOOGLE_MAPS_API_KEY
                           in env or .env).  ~95% hit rate, fast (~1-2 min).
  * --provider nominatim   OSM Nominatim (free, 1 req/sec, ~40-60% hit rate
                           with the postcode-cascade fallback).

Caches are provider-specific so results from both can coexist.  The default
is Google because it's dramatically more accurate for Malaysian shophouse
addresses.

Usage:
    python geocode_npra.py                 # google, resume from cache
    python geocode_npra.py --max 100       # stop after 100 new calls
    python geocode_npra.py --provider nominatim
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from local_sources import (
    parse_npra_pdf,
    geocode_addresses,           # Nominatim
    geocode_addresses_google,    # Google
    _load_google_api_key,
    detect_brand_from_name,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="data/source/farmasi-komuniti.pdf")
    ap.add_argument("--provider", choices=["google", "nominatim"], default="google",
                    help="Geocoder to use (default: google).")
    ap.add_argument("--cache", default=None,
                    help="Override cache path. Defaults per-provider.")
    ap.add_argument("--out", default="data/pharmacies_npra_geocoded.csv")
    ap.add_argument("--max", type=int, default=None,
                    help="Max network calls this run (rest retried next time).")
    args = ap.parse_args()

    print(f"→ Parsing {args.pdf} ...")
    pdf_df = parse_npra_pdf(args.pdf)
    print(f"  {len(pdf_df):,} rows")

    keyed = pdf_df.assign(
        _addr_key=pdf_df["address"].str.lower().str.strip()
        + "|"
        + pdf_df["state"].str.lower()
    )
    uniques = keyed.drop_duplicates("_addr_key").copy()
    print(f"  {len(uniques):,} unique (address, state) keys to geocode")

    t0 = time.time()
    if args.provider == "google":
        key = _load_google_api_key()
        if not key:
            print("ERROR: GOOGLE_MAPS_API_KEY not set. Add it to .env "
                  "(see .env.example) or export it in your shell.")
            return 2
        cache_path = args.cache or "data/geocode_cache_google.json"
        print(f"  provider: Google (cache: {cache_path})")
        geo = geocode_addresses_google(
            uniques, api_key=key, cache_path=cache_path, max_requests=args.max,
        )
    else:
        cache_path = args.cache or "data/geocode_cache.json"
        print(f"  provider: Nominatim (cache: {cache_path})")
        geo = geocode_addresses(
            uniques, cache_path=cache_path, max_requests=args.max,
        )
    done = int(geo["latitude"].notna().sum())
    print(f"→ Geocoded {done}/{len(geo)} ({done / len(geo) * 100:.1f}%)  "
          f"in {time.time() - t0:.1f}s")
    if "geocode_source" in geo.columns:
        print("  cascade breakdown:")
        print(geo["geocode_source"].value_counts().to_string())

    # Propagate lat/lon back to every original row (address may appear >1x
    # when a single premise has multiple preceptors listed).
    lookup_cols = ["latitude", "longitude"]
    if "geocode_source" in geo.columns:
        lookup_cols.append("geocode_source")
    lookup = geo.set_index("_addr_key")[lookup_cols]
    enriched = keyed.merge(lookup, left_on="_addr_key", right_index=True, how="left")
    enriched = enriched.drop(columns=["_addr_key"])
    # Detect the chain from the pharmacy name (e.g. 'Caring Pharmacy Sdn Bhd
    # - Bangsar Village' -> 'Caring').  Default to 'NPRA' so rows without a
    # recognised chain still get a distinct colour on the map.
    enriched["brand"] = enriched["name"].map(
        lambda n: detect_brand_from_name(n, default="NPRA")
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.out, index=False)
    print(f"→ Saved {args.out}: "
          f"{int(enriched['latitude'].notna().sum())}/{len(enriched)} rows have coordinates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
