"""
geocode_npra.py
---------------
Parse the NPRA 'Senarai Premis' PDF, dedupe addresses, and geocode via OSM
Nominatim with a cascade fallback (full address → postcode+town → state).

Results cache incrementally to `data/geocode_cache.json` so re-runs skip work
and interrupted runs resume cleanly.

Usage:
    python geocode_npra.py                 # geocode everything (takes ~15-20 min)
    python geocode_npra.py --max 50        # do 50 new calls at a time and stop
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from local_sources import parse_npra_pdf, geocode_addresses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="data/source/farmasi-komuniti.pdf")
    ap.add_argument("--cache", default="data/geocode_cache.json")
    ap.add_argument("--out", default="data/pharmacies_npra_geocoded.csv")
    ap.add_argument("--max", type=int, default=None,
                    help="Max network calls this run (rest retried on next run).")
    args = ap.parse_args()

    print(f"→ Parsing {args.pdf} ...")
    pdf_df = parse_npra_pdf(args.pdf)
    print(f"  {len(pdf_df):,} rows")

    # Dedup by (address, state) so duplicate addresses (common for chains with
    # multiple preceptors at the same premise) cost only one geocode.
    keyed = pdf_df.assign(
        _addr_key=pdf_df["address"].str.lower().str.strip()
        + "|"
        + pdf_df["state"].str.lower()
    )
    uniques = keyed.drop_duplicates("_addr_key").copy()
    print(f"  {len(uniques):,} unique (address, state) keys to geocode")

    t0 = time.time()
    geo = geocode_addresses(uniques, cache_path=args.cache, max_requests=args.max)
    done = int(geo["latitude"].notna().sum())
    print(f"→ Geocoded {done}/{len(geo)} ({done / len(geo) * 100:.1f}%)  "
          f"in {time.time() - t0:.1f}s")
    print("  breakdown by cascade step:")
    print(geo["geocode_source"].value_counts().to_string())

    # Propagate the geocoded lat/lon back to every original row via _addr_key.
    lookup = geo.set_index("_addr_key")[["latitude", "longitude", "geocode_source"]]
    enriched = keyed.merge(lookup, left_on="_addr_key", right_index=True, how="left")
    enriched = enriched.drop(columns=["_addr_key"])
    # Stamp a brand label so NPRA-only pharmacies render with their own colour.
    enriched["brand"] = "NPRA"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.out, index=False)
    have_coords = int(enriched["latitude"].notna().sum())
    print(f"→ Saved {args.out}: {have_coords}/{len(enriched)} rows have coordinates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
