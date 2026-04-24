"""
Kota Kinabalu Focus — KK + Penampang + Putatan + Tuaran + Papar.
District-clipped 1 km grid (geoBoundaries has no ADM3 for Sabah).
"""
from metro_focus import render_metro_focus


CONFIG = {
    "name": "Kota Kinabalu",
    "icon": "⛰️",
    "center": [5.98, 116.10],
    "zoom": 10,
    "sub_center": [5.98, 116.07],
    "sub_zoom": 11,
    "state_filter": ["Sabah"],
    "strategy": "adm2",
    "target_districts": (
        "Kota Kinabalu", "Penampang", "Putatan", "Tuaran", "Papar",
    ),
    "cache_key": "kk_v1_1km",
    "grid_path": "data/submukim_grid_kk.geojson",
    "pop_path": "data/worldpop_per_submukim_kk_v1_1km.csv",
    "intro": (
        "All metrics filtered to the **Kota Kinabalu metro** — KK + "
        "Penampang + Putatan + Tuaran + Papar.  Sub-grid is clipped to "
        "ADM2 district polygons because geoBoundaries has no mukim "
        "(ADM3) layer for Sabah."
    ),
}


render_metro_focus(CONFIG)
