"""
data_pipeline.py
----------------
Real-data loaders + the spatial-join / metric-computation pipeline.

Data sources (verify the latest dataset IDs on https://data.gov.my):
  * Pharmacies (NPRA / KKM):
        Registered community pharmacies + premise lat/lon. If a clean lat/lon
        feed is not yet on data.gov.my, fall back to the NPRA premise list CSV
        and geocode addresses (e.g. with OSM Nominatim) once.
  * Population by district (DOSM):
        Catalogue id is typically `population_district` (or the latest variant).
        Endpoint: https://api.data.gov.my/data-catalogue?id=<dataset_id>
  * District boundaries (GeoJSON):
        DOSM publishes administrative boundaries; `dosm-malaysia/data-open` on
        GitHub mirrors them. JUPEM is the authoritative source.

The functions below are designed to be swapped in for the mock loaders without
touching the dashboard code in `app.py`.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from shapely.geometry import shape, Point
from shapely.strtree import STRtree


# --------------------------------------------------------------------------------------
# 1. Fetchers
# --------------------------------------------------------------------------------------

DATAGOVMY_BASE = "https://api.data.gov.my/data-catalogue"


def fetch_datagovmy(dataset_id: str, params: Optional[Dict] = None, timeout: int = 30) -> pd.DataFrame:
    """
    Generic data.gov.my OpenAPI fetcher. Returns the dataset as a DataFrame.
    Pass extra `params` (e.g. {"filter": "state@Selangor"}) per the catalogue docs.
    """
    p = {"id": dataset_id}
    if params:
        p.update(params)
    r = requests.get(DATAGOVMY_BASE, params=p, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    # data.gov.my returns either a list of rows directly, or {"data": [...]}
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    return pd.DataFrame(payload)


def load_population_from_api(dataset_id: str = "population_district") -> pd.DataFrame:
    """
    Returns columns: district, state, population (latest year only).
    Adjust the column rename map if DOSM changes its schema.
    """
    df = fetch_datagovmy(dataset_id)
    # Defensive renames — DOSM uses snake_case Malay/English mixes.
    rename = {
        "daerah": "district", "district": "district",
        "negeri": "state",   "state": "state",
        "population_total": "population",
        "pop_total": "population",
        "value": "population",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "date" in df.columns:
        df = df.sort_values("date").drop_duplicates(["state", "district"], keep="last")
    keep = [c for c in ["district", "state", "population", "strata"] if c in df.columns]
    return df[keep].reset_index(drop=True)


def load_pharmacies_from_csv(path: str | Path) -> pd.DataFrame:
    """
    Expected columns (rename your source CSV to match):
        pharmacy_id, name, address, license_no, district, state, latitude, longitude
    """
    df = pd.read_csv(path)
    required = {"latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pharmacy CSV missing required columns: {missing}")
    df = df.dropna(subset=["latitude", "longitude"])
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    return df


def load_district_geojson(path_or_url: str | Path) -> Dict:
    """Load a district-boundary GeoJSON from disk or HTTP(S)."""
    s = str(path_or_url)
    if s.startswith("http://") or s.startswith("https://"):
        r = requests.get(s, timeout=60)
        r.raise_for_status()
        return r.json()
    return json.loads(Path(s).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# 1b. Live-data fetchers (no API key required)
# --------------------------------------------------------------------------------------
#
# These three loaders together make the dashboard runnable against real, current
# data without any manual CSV wrangling:
#   * `fetch_pharmacies_from_osm()`        — OpenStreetMap Overpass, amenity=pharmacy
#   * `load_population_district_dosm()`    — DOSM storage CSV (api.data.gov.my fallback)
#   * `load_malaysia_districts_geojson()`  — geoBoundaries ADM2 + ADM1 spatial join
#
# All three cache to disk (default: ./data/) and reuse the cache within `ttl_hours`.

OVERPASS_ENDPOINTS: List[str] = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]

# amenity=pharmacy covers retail/community pharmacies in OSM. We include node,
# way, and relation so mall-based and multi-building pharmacies are captured,
# and `out center` ensures ways/relations get a representative lat/lon.
OVERPASS_PHARMACY_QUERY_MY = """
[out:json][timeout:90];
area["ISO3166-1"="MY"][admin_level=2]->.my;
(
  node["amenity"="pharmacy"](area.my);
  way["amenity"="pharmacy"](area.my);
  relation["amenity"="pharmacy"](area.my);
);
out center tags;
""".strip()

DOSM_POPULATION_CSV = "https://storage.dosm.gov.my/population/population_district.csv"
GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/{level}/"


def _cache_is_fresh(path: Path, ttl_hours: float) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) < ttl_hours * 3600


def _overpass_post(endpoint: str, query: str, timeout: int = 180) -> Dict:
    r = requests.post(endpoint, data={"data": query}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _normalize_osm_elements(elements: List[Dict]) -> pd.DataFrame:
    """Flatten Overpass elements into the canonical pharmacy schema.

    The resulting DataFrame has no `district` / `state` yet — those are filled
    in later by `spatial_join_pharmacies_to_districts`.
    """
    rows = []
    for el in elements:
        tags = el.get("tags", {}) or {}
        if el.get("type") == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            center = el.get("center") or {}
            lat, lon = center.get("lat"), center.get("lon")
        if lat is None or lon is None:
            continue

        name = tags.get("name") or tags.get("brand") or "Pharmacy"
        addr_parts = [
            tags.get("addr:housenumber"),
            tags.get("addr:street"),
            tags.get("addr:city"),
            tags.get("addr:state"),
            tags.get("addr:postcode"),
        ]
        address = ", ".join(p for p in addr_parts if p)

        rows.append({
            "pharmacy_id": f"OSM{el['type'][0]}{el['id']}",
            "name": name,
            "address": address,
            "license_no": tags.get("ref", ""),
            "brand": tags.get("brand", ""),
            "latitude": float(lat),
            "longitude": float(lon),
        })
    return pd.DataFrame(rows)


def fetch_pharmacies_from_osm(
    cache_path: str | Path = "data/pharmacies_osm.json",
    ttl_hours: float = 24,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Query OSM Overpass for `amenity=pharmacy` inside Malaysia.

    Tries the primary Overpass endpoint, then two mirrors, then falls back to
    any cached payload on disk so the app still boots offline.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if _cache_is_fresh(cache_path, ttl_hours) and not force_refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        payload = None
        last_err: Optional[Exception] = None
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                payload = _overpass_post(endpoint, OVERPASS_PHARMACY_QUERY_MY)
                break
            except Exception as e:  # network, 429, 504, JSON decode, etc.
                last_err = e
                continue
        if payload is None:
            # Every live endpoint failed — fall back to stale cache rather than crash.
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                raise RuntimeError(f"All Overpass endpoints failed: {last_err}")
        else:
            cache_path.write_text(json.dumps(payload), encoding="utf-8")

    return _normalize_osm_elements(payload.get("elements", []))


def load_population_district_dosm(
    cache_path: str | Path = "data/population_district.csv",
    ttl_hours: float = 24,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch DOSM administrative-district population (latest year per district).

    Output columns: `district`, `state`, `population` (int).

    DOSM's `population_district` CSV has one row per (date, state, district,
    sex, age, ethnicity). We keep only the total slice and the latest date.
    Population values are published in thousands — we detect that heuristically
    and rescale so the rest of the pipeline can treat `population` as a raw
    head count.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not _cache_is_fresh(cache_path, ttl_hours) or force_refresh:
        r = requests.get(DOSM_POPULATION_CSV, timeout=60)
        r.raise_for_status()
        cache_path.write_bytes(r.content)

    df = pd.read_csv(cache_path)

    # Keep only the "total" slice per district.
    for col, target in [("sex", "both"), ("age", "overall"), ("ethnicity", "overall")]:
        if col in df.columns:
            df = df[df[col].astype(str).str.lower() == target]

    pop_col = "population" if "population" in df.columns else "value"
    raw = pd.to_numeric(df[pop_col], errors="coerce")

    # DOSM publishes population in thousands. Heuristic guard in case the
    # schema ever flips to raw counts: if the biggest value is still under
    # 100k (e.g. Kangar ~100 thousand), we're clearly in thousands.
    multiplier = 1000 if raw.max(skipna=True) < 100_000 else 1

    out = df.assign(population=(raw * multiplier).round().astype("Int64"))
    if "date" in out.columns:
        out = out.sort_values("date").drop_duplicates(["state", "district"], keep="last")

    return (
        out[["district", "state", "population"]]
        .dropna(subset=["population"])
        .reset_index(drop=True)
    )


def _fetch_geoboundaries(iso3: str, level: str, timeout: int = 60) -> Dict:
    """Two-step fetch: metadata → direct GeoJSON URL → GeoJSON."""
    meta_url = GEOBOUNDARIES_API.format(iso3=iso3, level=level)
    meta = requests.get(meta_url, timeout=timeout)
    meta.raise_for_status()
    info = meta.json()
    # geoBoundaries exposes both full and simplified geometries; prefer full.
    gj_url = info.get("gjDownloadURL") or info.get("simplifiedGeometryGeoJSON")
    if not gj_url:
        raise RuntimeError(f"No GeoJSON URL in geoBoundaries response for {iso3}/{level}")
    gj = requests.get(gj_url, timeout=timeout)
    gj.raise_for_status()
    return gj.json()


def load_malaysia_mukim_geojson(
    cache_path: str | Path = "data/mukim_my_adm3.geojson",
    ttl_hours: float = 24 * 30,
    force_refresh: bool = False,
) -> Dict:
    """Fetch Malaysia ADM3 (Mukim) boundaries from geoBoundaries.

    1,859 mukim across Malaysia — 12× the ADM2 district granularity.
    Each output feature has properties `{mukim, district, state}` where:
      * `mukim`    = geoBoundaries `shapeName` (sub-district name)
      * `district` = parent ADM2 name, stamped via centroid-in-ADM2 lookup
      * `state`    = parent ADM1 name, stamped via centroid-in-ADM1 lookup
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if _cache_is_fresh(cache_path, ttl_hours) and not force_refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    adm3 = _fetch_geoboundaries("MYS", "ADM3")
    adm2 = _fetch_geoboundaries("MYS", "ADM2")
    adm1 = _fetch_geoboundaries("MYS", "ADM1")

    state_polys = [shape(f["geometry"]) for f in adm1["features"]]
    state_names = [f["properties"].get("shapeName") for f in adm1["features"]]
    state_tree = STRtree(state_polys)

    district_polys = [shape(f["geometry"]) for f in adm2["features"]]
    district_names = [f["properties"].get("shapeName") for f in adm2["features"]]
    district_tree = STRtree(district_polys)

    def _parent(geom, tree, polys, names) -> Optional[str]:
        pt = geom.representative_point()
        for idx in tree.query(pt):
            if polys[idx].contains(pt):
                return names[idx]
        best_idx, best_area = None, 0.0
        for idx in tree.query(geom):
            inter = polys[idx].intersection(geom).area
            if inter > best_area:
                best_area, best_idx = inter, idx
        return names[best_idx] if best_idx is not None else None

    out_features = []
    for feat in adm3["features"]:
        geom = shape(feat["geometry"])
        out_features.append({
            "type": "Feature",
            "geometry": feat["geometry"],
            "properties": {
                "mukim":    feat["properties"].get("shapeName"),
                "district": _parent(geom, district_tree, district_polys, district_names),
                "state":    _parent(geom, state_tree, state_polys, state_names),
            },
        })

    out = {"type": "FeatureCollection", "features": out_features}
    cache_path.write_text(json.dumps(out), encoding="utf-8")
    return out


# KKM / Pharmacy Board Malaysia coarse pharmacy zones. States are grouped
# by the same regional buckets MOH uses for public-pharmacy logistics.
KKM_ZONES: Dict[str, List[str]] = {
    "Zon Utara":           ["Perlis", "Kedah", "Pulau Pinang", "Perak"],
    "Zon Tengah":          ["Selangor", "Kuala Lumpur", "Putrajaya", "Negeri Sembilan"],
    "Zon Selatan":         ["Melaka", "Johor"],
    "Zon Timur":           ["Kelantan", "Terengganu", "Pahang"],
    "Zon Sabah/Sarawak":   ["Sabah", "Sarawak", "Labuan"],
}


def zone_for_state(state: str) -> Optional[str]:
    """Return the KKM pharmacy zone for a state, or None if unmapped."""
    if not isinstance(state, str):
        return None
    for zone, states in KKM_ZONES.items():
        if state in states:
            return zone
    return None


def build_kkm_zones_geojson(adm1_geojson: Dict) -> Dict:
    """Union each zone's member-state polygons into a single multi-polygon.

    We don't call geoBoundaries for zones (KKM doesn't publish a zone layer)
    — we derive them by grouping ADM1 state polygons with the KKM_ZONES map.
    State names are normalized through `_canonical_state` so that "Penang"
    and "Pulau Pinang" (geoBoundaries vs DOSM/KKM) map to the same bucket.
    """
    from shapely.ops import unary_union
    state_geoms = {
        _canonical_state(f["properties"].get("shapeName")): shape(f["geometry"])
        for f in adm1_geojson["features"]
    }
    features = []
    for zone, states in KKM_ZONES.items():
        geoms = [state_geoms[s] for s in states if s in state_geoms]
        if not geoms:
            continue
        merged = unary_union(geoms)
        features.append({
            "type": "Feature",
            "geometry": merged.__geo_interface__,
            "properties": {"zone": zone, "state_count": len(geoms)},
        })
    return {"type": "FeatureCollection", "features": features}


def load_kkm_zones_geojson(
    cache_path: str | Path = "data/kkm_zones.geojson",
    force_refresh: bool = False,
) -> Dict:
    """Build (or load from cache) the KKM-zone GeoJSON."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    adm1 = _fetch_geoboundaries("MYS", "ADM1")
    gj = build_kkm_zones_geojson(adm1)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(gj), encoding="utf-8")
    return gj


def build_voronoi_catchments(
    pharmacies: pd.DataFrame,
    boundary_geojson: Dict,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> Dict:
    """Build a Voronoi polygon per pharmacy, clipped to a boundary.

    Each cell = "the area closer to this pharmacy than to any other" — a clean
    analytic interpretation of a pharmacy's catchment zone. The boundary (the
    union of all ADM1 states, usually) prevents cells from extending into the
    sea.
    """
    from shapely.geometry import MultiPoint
    from shapely.ops import voronoi_diagram, unary_union

    coords = pharmacies[[lon_col, lat_col]].dropna().to_numpy()
    if len(coords) < 3:
        raise ValueError("Need at least 3 points to build a Voronoi diagram.")

    pts = MultiPoint(coords)
    # Extract the boundary polygon (union of all boundary features).
    boundary = unary_union([shape(f["geometry"]) for f in boundary_geojson["features"]])

    diagram = voronoi_diagram(pts, envelope=boundary, edges=False)
    cells = list(diagram.geoms)

    # Match each Voronoi cell back to its generating pharmacy by point-in-cell.
    # With a large Voronoi we'd use a spatial index; 1-2k pharmacies is fine.
    pharm_rows = pharmacies.reset_index(drop=True)
    features = []
    for cell in cells:
        clipped = cell.intersection(boundary)
        if clipped.is_empty:
            continue
        # Find the pharmacy whose point lies inside this cell.
        name, pid, brand = "Unknown", "", "Other"
        for _, row in pharm_rows.iterrows():
            p = Point(row[lon_col], row[lat_col])
            if cell.contains(p):
                name = row.get("name") or name
                pid = row.get("pharmacy_id") or ""
                brand = row.get("brand") or "Other"
                break
        features.append({
            "type": "Feature",
            "geometry": clipped.__geo_interface__,
            "properties": {
                "catchment_id": pid or f"VC{len(features):05d}",
                "pharmacy":     name,
                "brand":        brand,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def load_malaysia_districts_geojson(
    cache_path: str | Path = "data/districts_my_adm2.geojson",
    ttl_hours: float = 24 * 30,  # boundaries change rarely — cache for a month
    force_refresh: bool = False,
) -> Dict:
    """Return a Malaysia district (ADM2) GeoJSON with normalized properties.

    geoBoundaries ships each feature with `shapeName` (district) but no parent
    state name. We fetch ADM1 separately and do a point-in-polygon lookup on
    each district's representative point to stamp in the state. The resulting
    feature properties are `{district, state}` — what `enrich_geojson_with_metrics`
    and the Folium tooltip expect.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if _cache_is_fresh(cache_path, ttl_hours) and not force_refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    adm2 = _fetch_geoboundaries("MYS", "ADM2")
    adm1 = _fetch_geoboundaries("MYS", "ADM1")

    # Build the state spatial index once.
    state_polys = [shape(f["geometry"]) for f in adm1["features"]]
    state_names = [f["properties"].get("shapeName") for f in adm1["features"]]
    state_tree = STRtree(state_polys)

    def _state_for(geom) -> Optional[str]:
        # Fast path: centroid containment.
        pt = geom.representative_point()
        for idx in state_tree.query(pt):
            if state_polys[idx].contains(pt):
                return state_names[idx]
        # Border fallback: whichever state has the greatest overlap area.
        best_idx, best_area = None, 0.0
        for idx in state_tree.query(geom):
            inter = state_polys[idx].intersection(geom).area
            if inter > best_area:
                best_area, best_idx = inter, idx
        return state_names[best_idx] if best_idx is not None else None

    out_features = []
    for feat in adm2["features"]:
        geom = shape(feat["geometry"])
        out_features.append({
            "type": "Feature",
            "geometry": feat["geometry"],
            "properties": {
                "district": feat["properties"].get("shapeName"),
                "state": _state_for(geom),
            },
        })

    out = {"type": "FeatureCollection", "features": out_features}
    cache_path.write_text(json.dumps(out), encoding="utf-8")
    return out


# --------------------------------------------------------------------------------------
# 1c. District-name normalization
# --------------------------------------------------------------------------------------
#
# DOSM, geoBoundaries, and NPRA do not agree on state / district spellings.
# Normalize both sides of the merge so `compute_district_metrics` doesn't
# silently drop rows to NaN.

_STATE_ALIASES: Dict[str, str] = {
    # All keys are the output of `_normalize_token`; values are canonical.
    "penang": "Pulau Pinang",
    "pulau pinang": "Pulau Pinang",
    "w.p. kuala lumpur": "Kuala Lumpur",
    "wp kuala lumpur": "Kuala Lumpur",
    "wp. kuala lumpur": "Kuala Lumpur",
    "kuala lumpur": "Kuala Lumpur",
    "federal territory of kuala lumpur": "Kuala Lumpur",
    "w.p. labuan": "Labuan",
    "wp labuan": "Labuan",
    "wp. labuan": "Labuan",
    "labuan": "Labuan",
    "federal territory of labuan": "Labuan",
    "w.p. putrajaya": "Putrajaya",
    "wp putrajaya": "Putrajaya",
    "wp. putrajaya": "Putrajaya",
    "putrajaya": "Putrajaya",
    "federal territory of putrajaya": "Putrajaya",
    "n. sembilan": "Negeri Sembilan",
    "negeri sembilan": "Negeri Sembilan",
    "malacca": "Melaka",
    "melaka": "Melaka",
}

_DISTRICT_ALIASES: Dict[str, str] = {
    "central melaka": "Melaka Tengah",
    "melaka tengah": "Melaka Tengah",
    "spt": "Seberang Perai Tengah",
    "seberang perai tengah": "Seberang Perai Tengah",
    "spu": "Seberang Perai Utara",
    "seberang perai utara": "Seberang Perai Utara",
    "sps": "Seberang Perai Selatan",
    "seberang perai selatan": "Seberang Perai Selatan",
    # Federal Territories: DOSM labels them "W.P. ...", geoBoundaries uses plain names.
    "w.p. kuala lumpur": "Kuala Lumpur",
    "wp kuala lumpur": "Kuala Lumpur",
    "wp. kuala lumpur": "Kuala Lumpur",
    "federal territory of kuala lumpur": "Kuala Lumpur",
    "w.p. labuan": "Labuan",
    "wp labuan": "Labuan",
    "wp. labuan": "Labuan",
    "federal territory of labuan": "Labuan",
    "w.p. putrajaya": "Putrajaya",
    "wp putrajaya": "Putrajaya",
    "wp. putrajaya": "Putrajaya",
    "federal territory of putrajaya": "Putrajaya",
    # DOSM / geoBoundaries drift on Selangor's largest district.
    "hulu langat": "Hulu Langat",
    "ulu langat": "Hulu Langat",
    # Common Perak variants.
    "larut dan matang": "Larut Matang Selama",
    "larut matang selama": "Larut Matang Selama",
    "larut, matang dan selama": "Larut Matang Selama",
    # Sarawak "Samarahan" administrative rename (older data says "Asajaya").
    "kota samarahan": "Kota Samarahan",
    "samarahan": "Kota Samarahan",
}


def _normalize_token(s) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip().lower()
    if t.startswith("daerah "):
        t = t[len("daerah "):]
    return " ".join(t.split())


def _canonical_district(s) -> str:
    if not isinstance(s, str):
        return s
    norm = _normalize_token(s)
    if norm in _DISTRICT_ALIASES:
        return _DISTRICT_ALIASES[norm]
    # Preserve Malaysian multi-word names with proper capitalization.
    return " ".join(w.capitalize() for w in norm.split()) if norm else s


def _canonical_state(s) -> str:
    if not isinstance(s, str):
        return s
    return _STATE_ALIASES.get(_normalize_token(s), s)


def normalize_district_names(
    df: pd.DataFrame,
    district_col: str = "district",
    state_col: str = "state",
) -> pd.DataFrame:
    """Return a copy of `df` with state/district spellings canonicalized."""
    out = df.copy()
    if state_col in out.columns:
        out[state_col] = out[state_col].map(_canonical_state)
    if district_col in out.columns:
        out[district_col] = out[district_col].map(_canonical_district)
    return out


def normalize_geojson_names(
    gj: Dict,
    district_key: str = "district",
    state_key: str = "state",
) -> Dict:
    """Apply the same canonicalization to a FeatureCollection's properties."""
    out = json.loads(json.dumps(gj))  # deep copy
    for feat in out["features"]:
        props = feat.setdefault("properties", {})
        if district_key in props:
            props[district_key] = _canonical_district(props[district_key])
        if state_key in props:
            props[state_key] = _canonical_state(props[state_key])
    return out


# --------------------------------------------------------------------------------------
# 2. Spatial join — assign each pharmacy to the district polygon it sits in
# --------------------------------------------------------------------------------------

def stamp_polygon_props(
    points_df: pd.DataFrame,
    polygons_geojson: Dict,
    prop_keys: List[str],
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    prefix: str = "",
) -> pd.DataFrame:
    """Point-in-polygon stamp — generic replacement for the district-specific
    join. For every row, find the polygon that contains (lon, lat) and copy
    the named property keys into new columns on the output DataFrame.

    If a column of the same name already exists on the input, it's preserved
    as `<name>_raw` so downstream joins don't silently overwrite source data.
    """
    polys: List = []
    props: List[Dict] = []
    for feat in polygons_geojson["features"]:
        polys.append(shape(feat["geometry"]))
        props.append(feat.get("properties", {}))
    tree = STRtree(polys)

    out = points_df.copy()
    # Preserve any existing columns that would collide.
    for k in prop_keys:
        col = f"{prefix}{k}"
        if col in out.columns:
            out = out.rename(columns={col: f"{col}_raw"})

    # Bulk query where possible.
    lons = out[lon_col].to_numpy()
    lats = out[lat_col].to_numpy()
    n = len(out)
    pts = np.empty(n, dtype=object)
    for i in range(n):
        pts[i] = Point(lons[i], lats[i])

    # `within` is the predicate for "input point within tree polygon".
    try:
        input_idx, tree_idx = tree.query(pts, predicate="within")
        # Each point matches at most one polygon — build a point→polygon map.
        match_map = np.full(n, -1, dtype=np.int64)
        match_map[input_idx] = tree_idx
    except Exception:
        match_map = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            for idx in tree.query(pts[i]):
                if polys[idx].contains(pts[i]):
                    match_map[i] = idx
                    break

    for key in prop_keys:
        out[f"{prefix}{key}"] = [
            props[match_map[i]].get(key) if match_map[i] >= 0 else None
            for i in range(n)
        ]
    return out


def spatial_join_pharmacies_to_districts(
    pharmacies: pd.DataFrame,
    districts_geojson: Dict,
    district_name_field: str = "district",
    state_name_field: str = "state",
) -> pd.DataFrame:
    """
    Returns the input DataFrame with `district` and `state` columns overwritten
    based on point-in-polygon containment. Uses an STRtree for O(log N) lookups.

    If the input already has district/state columns (e.g. mock data), those are
    preserved as `district_raw`/`state_raw` for QA.
    """
    polys: List = []
    props: List[Dict] = []
    for feat in districts_geojson["features"]:
        polys.append(shape(feat["geometry"]))
        props.append(feat.get("properties", {}))
    tree = STRtree(polys)

    out = pharmacies.copy()
    if "district" in out.columns:
        out = out.rename(columns={"district": "district_raw"})
    if "state" in out.columns:
        out = out.rename(columns={"state": "state_raw"})

    matched_district, matched_state = [], []
    for lon, lat in zip(out["longitude"].to_numpy(), out["latitude"].to_numpy()):
        pt = Point(lon, lat)
        # STRtree.query returns candidate indices; verify with .contains
        cand_idx = tree.query(pt)
        d_name, s_name = None, None
        for idx in cand_idx:
            if polys[idx].contains(pt):
                d_name = props[idx].get(district_name_field)
                s_name = props[idx].get(state_name_field)
                break
        matched_district.append(d_name)
        matched_state.append(s_name)
    out["district"] = matched_district
    out["state"] = matched_state
    return out


# --------------------------------------------------------------------------------------
# 3. Metric computation + GeoJSON enrichment
# --------------------------------------------------------------------------------------

def compute_district_metrics(
    pharmacies_with_district: pd.DataFrame,
    population_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the per-district metric table used everywhere in the dashboard.

    Output columns:
        district, state, population, strata (if available),
        pharmacy_count, pop_per_pharmacy, pharmacies_per_100k
    """
    counts = (
        pharmacies_with_district
        .dropna(subset=["district"])
        .groupby(["district", "state"], dropna=False)
        .size()
        .reset_index(name="pharmacy_count")
    )
    merged = population_df.merge(counts, on=["district", "state"], how="left")
    merged["pharmacy_count"] = merged["pharmacy_count"].fillna(0).astype(int)

    # Avoid division-by-zero — use NaN where there are no pharmacies.
    merged["pop_per_pharmacy"] = np.where(
        merged["pharmacy_count"] > 0,
        merged["population"] / merged["pharmacy_count"],
        np.nan,
    )
    merged["pharmacies_per_100k"] = merged["pharmacy_count"] / merged["population"] * 100_000
    merged["pharmacies_per_1000"] = merged["pharmacy_count"] / merged["population"] * 1_000
    return merged


def compute_polygon_metrics(
    pharmacies_with_keys: pd.DataFrame,
    population_df: pd.DataFrame,
    on: List[str],
) -> pd.DataFrame:
    """Generic metric builder — works for district, mukim, zone, catchment.

    `on` is the list of columns that identify a polygon (e.g. ["district","state"]
    or ["mukim","district","state"] or ["zone"] or ["catchment_id"]).

    Output columns: *on, population, pharmacy_count, pop_per_pharmacy,
    pharmacies_per_100k, pharmacies_per_1000.
    """
    counts = (
        pharmacies_with_keys
        .dropna(subset=on)
        .groupby(on, dropna=False)
        .size()
        .reset_index(name="pharmacy_count")
    )
    merged = population_df.merge(counts, on=on, how="left")
    merged["pharmacy_count"] = merged["pharmacy_count"].fillna(0).astype(int)
    merged["pop_per_pharmacy"] = np.where(
        merged["pharmacy_count"] > 0,
        merged["population"] / merged["pharmacy_count"],
        np.nan,
    )
    pop = merged["population"].replace(0, np.nan)
    merged["pharmacies_per_100k"] = merged["pharmacy_count"] / pop * 100_000
    merged["pharmacies_per_1000"] = merged["pharmacy_count"] / pop * 1_000
    return merged


def enrich_geojson_with_polygon_metrics(
    geojson: Dict,
    metrics_df: pd.DataFrame,
    on: List[str],
) -> Dict:
    """Generic enrichment for mukim/zone/catchment layers.

    Joins each feature to `metrics_df` by a tuple of property keys, injecting
    the metric columns into `feature.properties` so Folium tooltips read them.
    """
    lookup = metrics_df.set_index(on).to_dict(orient="index") if on else {}
    out = json.loads(json.dumps(geojson))
    for feat in out["features"]:
        props = feat.setdefault("properties", {})
        key = tuple(props.get(k) for k in on) if len(on) > 1 else props.get(on[0])
        m = lookup.get(key, {}) if lookup else {}
        props["population"] = int(m.get("population", 0) or 0)
        props["pharmacy_count"] = int(m.get("pharmacy_count", 0) or 0)
        ratio = m.get("pop_per_pharmacy")
        props["pop_per_pharmacy"] = (
            f"1 : {int(ratio):,}" if pd.notna(ratio) else "N/A"
        )
        props["pharmacies_per_100k"] = round(float(m.get("pharmacies_per_100k", 0) or 0), 2)
        props["pharmacies_per_1000"] = round(float(m.get("pharmacies_per_1000", 0) or 0), 3)
    return out


def enrich_geojson_with_metrics(
    districts_geojson: Dict,
    metrics_df: pd.DataFrame,
    join_key: str = "district",
) -> Dict:
    """
    Inject the metric columns into each feature's `properties` so Folium
    tooltips can read them directly.
    """
    lookup = metrics_df.set_index(join_key).to_dict(orient="index")
    out = json.loads(json.dumps(districts_geojson))  # deep copy
    for feat in out["features"]:
        key = feat["properties"].get(join_key)
        m = lookup.get(key, {})
        feat["properties"]["population"] = int(m.get("population", 0))
        feat["properties"]["pharmacy_count"] = int(m.get("pharmacy_count", 0))
        ratio = m.get("pop_per_pharmacy")
        feat["properties"]["pop_per_pharmacy"] = (
            f"1 : {int(ratio):,}" if pd.notna(ratio) else "N/A"
        )
        feat["properties"]["pharmacies_per_100k"] = round(
            float(m.get("pharmacies_per_100k", 0) or 0), 2
        )
        if "strata" in m:
            feat["properties"]["strata"] = m["strata"]
    return out
