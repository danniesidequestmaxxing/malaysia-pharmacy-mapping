"""
Johor Focus — the original per-metro focus page.  All heavy lifting now
lives in `metro_focus.render_metro_focus`; this file only provides the
metro-specific config.
"""
from metro_focus import render_metro_focus


CONFIG = {
    "name": "Johor",
    "icon": "🔍",
    "center": [2.0, 103.3],
    "zoom": 9,
    "sub_center": [1.55, 103.7],   # JB Bahru city centre
    "sub_zoom": 11,
    "state_filter": ["Johor"],
    "strategy": "mukim_names",
    "target_names": (
        "Mukim Tebrau", "Mukim Plentong", "Mukim Pulai",
        "Mukim Kota Tinggi", "Mukim Senai", "Mukim Kulai",
        "Bandar Johor Bahru", "Bandar Kulai",
    ),
    "cache_key": "v3_8mukim_1km",
    "grid_path": "data/submukim_grid_johor.geojson",
    "pop_path": "data/worldpop_per_submukim_v3_8mukim_1km.csv",
    "intro": (
        "All metrics filtered to Johor state. Switch to **Sub-Mukim Grid** "
        "for 1 km resolution across the Bandar JB / Tebrau / Plentong / "
        "Pulai / Kota Tinggi / Senai / Kulai belt."
    ),
}


render_metro_focus(CONFIG)
