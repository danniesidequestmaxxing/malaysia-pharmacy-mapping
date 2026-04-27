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
    # State-level zoom on the grid view too — the grid now covers every
    # Johor mukim, not just the JB urban belt.
    "sub_center": [2.0, 103.3],
    "sub_zoom": 8,
    "state_filter": ["Johor"],
    # If the committed grid file is missing, the runtime falls back to
    # rebuilding from these districts so the rebuild still produces a
    # state-wide grid.
    "strategy": "mukim_districts",
    "target_districts": (
        "Batu Pahat", "Johor Bahru", "Kluang", "Kota Tinggi",
        "Kulaijaya", "Ledang", "Mersing", "Muar", "Pontian", "Segamat",
    ),
    "cache_key": "johor_full_1km",
    "grid_path": "data/submukim_grid_johor_full.geojson",
    "pop_path": "data/worldpop_per_submukim_johor_full_1km.csv",
    "intro": (
        "Sub-Mukim Grid now covers **every district in Johor** at 1 km "
        "resolution. The JB urban belt (Bandar JB / Tebrau / Plentong / "
        "Pulai / Kota Tinggi / Senai / Kulai) keeps its high-resolution "
        "1 km WorldPop population; the rest of the state distributes the "
        "per-mukim WorldPop total uniformly across cells."
    ),
}


render_metro_focus(CONFIG)
