"""
Klang Valley Focus — KL + Selangor core districts + Putrajaya.
Mukim-clipped 1 km grid.
"""
from metro_focus import render_metro_focus


CONFIG = {
    "name": "Klang Valley + Selangor",
    "icon": "🏙️",
    "center": [3.25, 101.45],            # roughly Shah Alam — centre of Selangor
    "zoom": 9,
    "sub_center": [3.15, 101.70],        # KLCC for the 1 km grid view
    "sub_zoom": 11,
    "state_filter": ["Selangor", "Kuala Lumpur", "Putrajaya"],
    "strategy": "mukim_districts",
    # All 10 Selangor districts + the 2 KL districts.  "Kuala Langat",
    # "Kuala Selangor", "Sabak Bernam" and "Ulu Selangor" extend coverage
    # north and west beyond the core metro.  geoBoundaries normalises
    # "Hulu" as "Ulu" for Selangor.
    "target_districts": (
        # Selangor — 10 districts.  Note: geoBoundaries ships "Ulu Langat"
        # but our name-canonicaliser rewrites it to the common "Hulu
        # Langat", so that's the string we need here to match.
        "Petaling", "Hulu Langat", "Gombak", "Klang", "Sepang",
        "Kuala Langat", "Kuala Selangor", "Sabak Bernam", "Ulu Selangor",
        # KL (state) contributes these district names
        "Kuala Lumpur",
    ),
    "cache_key": "klang_valley_v2_full_1km",
    "grid_path": "data/submukim_grid_klang_valley.geojson",
    "pop_path": "data/worldpop_per_submukim_klang_valley_v2_full_1km.csv",
    "intro": (
        "All metrics filtered to **Selangor + KL + Putrajaya** — the full "
        "10 Selangor districts + both KL districts + Putrajaya.  Covers "
        "the Klang Valley core plus rural-north (Sabak Bernam, Kuala "
        "Selangor) and west-coast (Kuala Langat) districts.  Sub-Mukim "
        "Grid gives 1 km resolution across the whole territory."
    ),
}


render_metro_focus(CONFIG)
