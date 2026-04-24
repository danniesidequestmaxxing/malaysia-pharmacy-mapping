"""
Klang Valley Focus — KL + Selangor core districts + Putrajaya.
Mukim-clipped 1 km grid.
"""
from metro_focus import render_metro_focus


CONFIG = {
    "name": "Klang Valley",
    "icon": "🏙️",
    "center": [3.10, 101.60],
    "zoom": 10,
    "sub_center": [3.15, 101.70],        # KLCC
    "sub_zoom": 11,
    "state_filter": ["Selangor", "Kuala Lumpur", "Putrajaya"],
    "strategy": "mukim_districts",
    # geoBoundaries normalises "Hulu" as "Ulu" for Selangor's districts; KL
    # and Putrajaya appear as both states and as ADM2 districts, so we list
    # the district names here.
    "target_districts": (
        "Petaling", "Ulu Langat", "Gombak", "Klang", "Sepang", "Kuala Lumpur",
    ),
    "cache_key": "klang_valley_v1_1km",
    "grid_path": "data/submukim_grid_klang_valley.geojson",
    "pop_path": "data/worldpop_per_submukim_klang_valley_v1_1km.csv",
    "intro": (
        "All metrics filtered to the **Klang Valley** — KL + core Selangor "
        "districts (Petaling, Ulu Langat, Gombak, Klang, Sepang) + Putrajaya. "
        "Switch to **Sub-Mukim Grid** for 1 km resolution across the whole belt."
    ),
}


render_metro_focus(CONFIG)
