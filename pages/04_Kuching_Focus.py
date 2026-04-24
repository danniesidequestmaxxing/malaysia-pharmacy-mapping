"""
Kuching Focus — Kuching + Samarahan (Sarawak's urban west coast).
Mukim-clipped 1 km grid.
"""
from metro_focus import render_metro_focus


CONFIG = {
    "name": "Kuching",
    "icon": "🌴",
    "center": [1.55, 110.35],
    "zoom": 10,
    "sub_center": [1.55, 110.36],
    "sub_zoom": 11,
    "state_filter": ["Sarawak"],
    "strategy": "mukim_districts",
    "target_districts": ("Kuching", "Samarahan"),
    "cache_key": "kuching_v1_1km",
    "grid_path": "data/submukim_grid_kuching.geojson",
    "pop_path": "data/worldpop_per_submukim_kuching_v1_1km.csv",
    "intro": (
        "All metrics filtered to the **Kuching metro** — Kuching district "
        "+ Samarahan.  Switch to Sub-Mukim Grid for 1 km resolution."
    ),
}


render_metro_focus(CONFIG)
