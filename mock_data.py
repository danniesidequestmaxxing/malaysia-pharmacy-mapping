"""
mock_data.py
------------
Generates realistic mock data so the dashboard runs out-of-the-box,
before you wire in the real data.gov.my / DOSM / NPRA datasets.

Produces:
    - districts_df    : one row per Malaysian daerah (population, strata, centroid)
    - pharmacies_df   : ~N pharmacy points jittered around each district centroid
    - districts_geojson: a minimal GeoJSON of square polygons around each centroid,
                        good enough for choropleth rendering during development.

Replace these loaders with the real-data loaders in `data_pipeline.py`
once you have the official CSVs / API access.
"""

from __future__ import annotations
import json
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Realistic Malaysian districts (daerah). lat/lon ≈ administrative-centre coordinates.
# Population numbers are illustrative (loosely based on DOSM 2020 census order-of-magnitude).
# Strata is a coarse Urban/Rural label commonly used in Malaysian planning.
# --------------------------------------------------------------------------------------
DISTRICTS: List[Dict] = [
    # Selangor
    {"district": "Petaling",       "state": "Selangor",         "lat": 3.1073, "lon": 101.6068, "population": 2_500_000, "strata": "Urban"},
    {"district": "Hulu Langat",    "state": "Selangor",         "lat": 3.0738, "lon": 101.7910, "population": 1_400_000, "strata": "Urban"},
    {"district": "Klang",          "state": "Selangor",         "lat": 3.0319, "lon": 101.4450, "population": 1_050_000, "strata": "Urban"},
    {"district": "Gombak",         "state": "Selangor",         "lat": 3.2680, "lon": 101.6886, "population":   720_000, "strata": "Urban"},
    {"district": "Sepang",         "state": "Selangor",         "lat": 2.6892, "lon": 101.7506, "population":   260_000, "strata": "Rural"},

    # WP Kuala Lumpur
    {"district": "Kuala Lumpur",   "state": "W.P. Kuala Lumpur","lat": 3.1390, "lon": 101.6869, "population": 1_980_000, "strata": "Urban"},

    # Johor
    {"district": "Johor Bahru",    "state": "Johor",            "lat": 1.4927, "lon": 103.7414, "population": 1_700_000, "strata": "Urban"},
    {"district": "Muar",           "state": "Johor",            "lat": 2.0442, "lon": 102.5689, "population":   270_000, "strata": "Rural"},
    {"district": "Batu Pahat",     "state": "Johor",            "lat": 1.8548, "lon": 102.9325, "population":   430_000, "strata": "Rural"},
    {"district": "Kluang",         "state": "Johor",            "lat": 2.0303, "lon": 103.3186, "population":   320_000, "strata": "Rural"},

    # Pulau Pinang
    {"district": "Timur Laut",     "state": "Pulau Pinang",     "lat": 5.4145, "lon": 100.3292, "population":   570_000, "strata": "Urban"},
    {"district": "Barat Daya",     "state": "Pulau Pinang",     "lat": 5.3300, "lon": 100.2000, "population":   230_000, "strata": "Rural"},
    {"district": "Seberang Perai Tengah", "state": "Pulau Pinang", "lat": 5.3950, "lon": 100.4880, "population":   400_000, "strata": "Urban"},

    # Perak
    {"district": "Kinta",          "state": "Perak",            "lat": 4.5841, "lon": 101.0829, "population":   780_000, "strata": "Urban"},
    {"district": "Manjung",        "state": "Perak",            "lat": 4.2105, "lon": 100.6700, "population":   250_000, "strata": "Rural"},
    {"district": "Larut Matang Selama", "state": "Perak",       "lat": 4.7660, "lon": 100.7370, "population":   380_000, "strata": "Rural"},

    # Kedah
    {"district": "Kuala Muda",     "state": "Kedah",            "lat": 5.6520, "lon": 100.4990, "population":   470_000, "strata": "Rural"},
    {"district": "Kota Setar",     "state": "Kedah",            "lat": 6.1184, "lon": 100.3685, "population":   370_000, "strata": "Urban"},

    # Kelantan
    {"district": "Kota Bharu",     "state": "Kelantan",         "lat": 6.1254, "lon": 102.2386, "population":   530_000, "strata": "Urban"},
    {"district": "Pasir Mas",      "state": "Kelantan",         "lat": 6.0490, "lon": 102.1390, "population":   210_000, "strata": "Rural"},

    # Terengganu
    {"district": "Kuala Terengganu","state": "Terengganu",      "lat": 5.3296, "lon": 103.1370, "population":   430_000, "strata": "Urban"},
    {"district": "Kemaman",        "state": "Terengganu",       "lat": 4.2350, "lon": 103.4190, "population":   180_000, "strata": "Rural"},

    # Pahang
    {"district": "Kuantan",        "state": "Pahang",           "lat": 3.8077, "lon": 103.3260, "population":   620_000, "strata": "Urban"},
    {"district": "Temerloh",       "state": "Pahang",           "lat": 3.4500, "lon": 102.4170, "population":   170_000, "strata": "Rural"},

    # Negeri Sembilan
    {"district": "Seremban",       "state": "Negeri Sembilan",  "lat": 2.7297, "lon": 101.9381, "population":   680_000, "strata": "Urban"},
    {"district": "Port Dickson",   "state": "Negeri Sembilan",  "lat": 2.5236, "lon": 101.7956, "population":   115_000, "strata": "Rural"},

    # Melaka
    {"district": "Melaka Tengah",  "state": "Melaka",           "lat": 2.1896, "lon": 102.2501, "population":   500_000, "strata": "Urban"},
    {"district": "Alor Gajah",     "state": "Melaka",           "lat": 2.3800, "lon": 102.2080, "population":   190_000, "strata": "Rural"},

    # Perlis
    {"district": "Kangar",         "state": "Perlis",           "lat": 6.4414, "lon": 100.1986, "population":   100_000, "strata": "Urban"},

    # Sabah
    {"district": "Kota Kinabalu",  "state": "Sabah",            "lat": 5.9804, "lon": 116.0735, "population":   500_000, "strata": "Urban"},
    {"district": "Sandakan",       "state": "Sabah",            "lat": 5.8402, "lon": 118.1179, "population":   430_000, "strata": "Urban"},
    {"district": "Tawau",          "state": "Sabah",            "lat": 4.2440, "lon": 117.8910, "population":   410_000, "strata": "Rural"},

    # Sarawak
    {"district": "Kuching",        "state": "Sarawak",          "lat": 1.5497, "lon": 110.3626, "population":   720_000, "strata": "Urban"},
    {"district": "Miri",           "state": "Sarawak",          "lat": 4.3995, "lon": 113.9914, "population":   360_000, "strata": "Urban"},
    {"district": "Sibu",           "state": "Sarawak",          "lat": 2.2870, "lon": 111.8307, "population":   260_000, "strata": "Urban"},
    {"district": "Bintulu",        "state": "Sarawak",          "lat": 3.1717, "lon": 113.0411, "population":   240_000, "strata": "Rural"},
]


def get_districts_df() -> pd.DataFrame:
    """Returns the canonical district-level reference table."""
    return pd.DataFrame(DISTRICTS)


def generate_mock_pharmacies(seed: int = 42, density_per_capita: int = 8_000) -> pd.DataFrame:
    """
    Generate a mock pharmacy registry. For each district, the number of pharmacies
    is roughly population / density_per_capita, with some Poisson noise so the
    population-per-pharmacy ratio varies realistically across districts.

    Each pharmacy is jittered ~0.05–0.12 deg around the district centroid (≈ 5–13 km),
    so points fall inside the mock square polygons we generate for the choropleth.
    """
    rng = np.random.default_rng(seed)
    rows = []
    pid = 1
    for d in DISTRICTS:
        n_expected = max(1, round(d["population"] / density_per_capita))
        n = int(rng.poisson(lam=n_expected))
        for i in range(n):
            # Jitter inside roughly the district bounding-box used by the mock GeoJSON.
            lat_jitter = rng.uniform(-0.12, 0.12)
            lon_jitter = rng.uniform(-0.12, 0.12)
            rows.append({
                "pharmacy_id": f"PH{pid:05d}",
                "name": f"Farmasi {d['district']} {i+1}",
                "address": f"Lot {rng.integers(1, 999)}, Jalan Utama, {d['district']}",
                "license_no": f"PRA{rng.integers(10000, 99999)}",
                "district": d["district"],
                "state": d["state"],
                "strata": d["strata"],
                "latitude": d["lat"] + lat_jitter,
                "longitude": d["lon"] + lon_jitter,
            })
            pid += 1
    return pd.DataFrame(rows)


def generate_mock_geojson(half_size_deg: float = 0.15) -> Dict:
    """
    Build a minimal FeatureCollection where each district is a square polygon
    centred on its (lat, lon). This is *not* an accurate boundary — it's just
    enough to render a working choropleth during development.

    Replace with the real DOSM/JUPEM admin-boundary GeoJSON in production
    (see `data_pipeline.load_district_geojson`).
    """
    features = []
    for d in DISTRICTS:
        lat, lon = d["lat"], d["lon"]
        h = half_size_deg
        polygon = [[
            [lon - h, lat - h],
            [lon + h, lat - h],
            [lon + h, lat + h],
            [lon - h, lat + h],
            [lon - h, lat - h],  # closed ring
        ]]
        features.append({
            "type": "Feature",
            "properties": {
                "district": d["district"],
                "state": d["state"],
            },
            "geometry": {"type": "Polygon", "coordinates": polygon},
        })
    return {"type": "FeatureCollection", "features": features}


if __name__ == "__main__":
    # Quick smoke-test: print sizes when run directly.
    df_d = get_districts_df()
    df_p = generate_mock_pharmacies()
    gj = generate_mock_geojson()
    print(f"Districts: {len(df_d)} | Pharmacies: {len(df_p)} | GeoJSON features: {len(gj['features'])}")
    print(df_p.head())
