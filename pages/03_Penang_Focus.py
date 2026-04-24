"""
Penang Focus — Penang Island + Seberang Perai.
District-clipped 1 km grid (geoBoundaries has no ADM3 for Penang).
"""
from metro_focus import render_metro_focus


CONFIG = {
    "name": "Penang",
    "icon": "🏝️",
    "center": [5.35, 100.35],
    "zoom": 10,
    "sub_center": [5.42, 100.33],         # Georgetown
    "sub_zoom": 11,
    "state_filter": ["Pulau Pinang"],
    "strategy": "adm2",                   # fall back to ADM2 districts
    "target_districts": (
        "Timur Laut", "Barat Daya",
        "Seberang Perai Tengah", "Seberang Perai Utara", "Seberang Perai Selatan",
    ),
    "cache_key": "penang_v1_1km",
    "grid_path": "data/submukim_grid_penang.geojson",
    "pop_path": "data/worldpop_per_submukim_penang_v1_1km.csv",
    "intro": (
        "All metrics filtered to **Penang** — Timur Laut + Barat Daya "
        "(the Island) plus the three Seberang Perai mainland districts. "
        "Sub-grid is clipped to ADM2 district polygons because geoBoundaries "
        "has no mukim (ADM3) layer for Penang."
    ),
}


render_metro_focus(CONFIG)
