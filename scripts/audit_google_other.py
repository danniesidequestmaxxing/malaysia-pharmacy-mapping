"""
audit_google_other.py
---------------------
One-shot helper: extracts the `Other` (independent) rows from
data/pharmacies_google_johor.csv into a separate audit CSV that's
easier to review on GitHub.

Run from the repo root:
    python scripts/audit_google_other.py
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path


SRC = Path("data/pharmacies_google_johor.csv")
OUT = Path("data/pharmacies_google_johor_audit_other.csv")


def main() -> None:
    df = pd.read_csv(SRC)
    other = df[df["brand"] == "Other"].copy().sort_values("name").reset_index(drop=True)
    cols = ["pharmacy_id", "name", "address", "latitude", "longitude", "source"]
    other = other[[c for c in cols if c in other.columns]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    other.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(other)} 'Other'-branded entries from Google Places")


if __name__ == "__main__":
    main()
