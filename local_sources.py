"""
local_sources.py
----------------
Parsers + loaders for the three user-supplied authoritative sources:

  * `Project Pharma.kmz`         — pre-geocoded pharmacy placemarks (chain map).
  * `farmasi-komuniti.pdf`       — NPRA / Pharmacy Board Malaysia "Senarai Premis"
                                   list of ROPA-1951 registered community pharmacies
                                   with full addresses (needs geocoding).
  * `mys_general_2020.csv/.tif`  — WorldPop gridded population (~100m cells).

All three are merged into the same canonical schema the rest of the pipeline
expects:

    Pharmacies:   pharmacy_id, name, address, latitude, longitude, brand, source
    Population:   district, state, population

Privacy note:
    The NPRA PDF contains preceptor names and email addresses for each premise.
    Those columns are DROPPED at load time — we never pass them to the dashboard,
    never cache them to disk, and never expose them in the map tooltip.
"""

from __future__ import annotations
import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, shape
from shapely.strtree import STRtree


# --------------------------------------------------------------------------------------
# 1. KMZ — pre-geocoded placemarks (fastest path: no geocoding needed)
# --------------------------------------------------------------------------------------

_KML_NS = {"k": "http://www.opengis.net/kml/2.2"}


def _style_to_brand(style_url: str) -> str:
    """Map the KMZ's styleUrl hex colour to its chain name.

    The "Project Pharma" KMZ organises placemarks into six folders, one per
    chain, and assigns a distinct icon colour per folder:
        Caring Pharmacy          #097138  green
        Alpro Pharmacy           #9C27B0  purple
        BIG Pharmacy             #FF5252  red
        Healthlane Pharmacy      #757575  grey
        Sunway Multicare         #1A237E  dark blue
        AA Pharmacy              #FFEA00  yellow
    """
    code = (style_url or "").split("-")[-2] if "-" in (style_url or "") else ""
    return {
        "097138": "Caring",
        "9C27B0": "Alpro",
        "FF5252": "BIG Pharmacy",
        "757575": "Healthlane",
        "1A237E": "Sunway Multicare",
        "FFEA00": "AA Pharmacy",
    }.get(code, "Other")


# Name-based chain detection — works on KMZ sub-brands (e.g. Georgetown
# Pharmacy inside the Caring folder), NPRA PDF rows, and PMG Excel rows.
# Ordered: more-specific patterns first so "farmasi alpro" hits "Alpro" not
# the generic "farmasi" fallback.
_BRAND_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bcaring\s+pharmacy|farmasi\s+caring", re.I),        "Caring"),
    (re.compile(r"\b(farmasi\s+alpro|alpro)\b", re.I),                 "Alpro"),
    (re.compile(r"\bbig\s+pharmacy|aeon\s*big", re.I),                 "BIG Pharmacy"),
    (re.compile(r"\bhealth\s*lane\b", re.I),                           "Healthlane"),
    (re.compile(r"sunway\s+multicare|\bmulticare\s+pharmacy", re.I),   "Sunway Multicare"),
    (re.compile(r"\baa\s+pharmacy\b", re.I),                           "AA Pharmacy"),
    (re.compile(r"\bguardian\b", re.I),                                "Guardian"),
    (re.compile(r"\bwatsons?\b", re.I),                                "Watsons"),
    (re.compile(r"\bpmg\s+pharmacy\b|^pmg\s+|\(pmg\)", re.I),          "PMG"),
    (re.compile(r"am\s*pm\s+pharmacy", re.I),                          "AM PM"),
    (re.compile(r"\bnazen\b",     re.I),                               "Nazen"),
    (re.compile(r"\bgeorgetown\s+pharmacy", re.I),                     "Georgetown"),
    (re.compile(r"\bsiang\s+pharmacy|siang.*kulim", re.I),             "Siang"),
    (re.compile(r"\balliance\s+pharmacy", re.I),                       "Alliance"),
    (re.compile(r"\bmega\s+kulim", re.I),                              "Mega Kulim"),
    (re.compile(r"\bwellings\s+pharmacy", re.I),                       "Wellings"),
    (re.compile(r"\bstraits\s+pharmacy", re.I),                        "Straits"),
    (re.compile(r"\brx\s+pharmacy|rx\s+drug", re.I),                   "Rx"),
    (re.compile(r"\bconstant\s+pharmacy", re.I),                       "Constant"),
    (re.compile(r"\bmediq\b", re.I),                                   "MediQ"),
    (re.compile(r"\brejoice\s+pharmacy", re.I),                        "Rejoice"),
    (re.compile(r"\bbemed|be\s+pharmacy", re.I),                       "Be Pharmacy"),
]


def detect_brand_from_name(name: str, default: str = "Independent") -> str:
    """Identify the chain a pharmacy belongs to by its name string."""
    if not isinstance(name, str) or not name.strip():
        return default
    for pat, brand in _BRAND_PATTERNS:
        if pat.search(name):
            return brand
    return default


def parse_kmz(path: str | Path) -> pd.DataFrame:
    """Extract pharmacy placemarks from a Google-Earth KMZ.

    Returns columns:
        pharmacy_id, name, latitude, longitude, brand, source
    """
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        # Pick the first .kml inside the zip (KMZ convention: doc.kml).
        kml_name = next(n for n in zf.namelist() if n.lower().endswith(".kml"))
        kml_bytes = zf.read(kml_name)

    root = ET.fromstring(kml_bytes)
    rows: List[Dict] = []
    for placemark in root.iter("{http://www.opengis.net/kml/2.2}Placemark"):
        name_el = placemark.find("k:name", _KML_NS)
        coord_el = placemark.find(".//k:coordinates", _KML_NS)
        style_el = placemark.find("k:styleUrl", _KML_NS)
        if coord_el is None or not (coord_el.text or "").strip():
            continue
        # KML coords are "lon,lat[,alt]" strings, possibly whitespace-padded.
        parts = coord_el.text.strip().split(",")
        if len(parts) < 2:
            continue
        try:
            lon, lat = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        name = (name_el.text or "").strip() if name_el is not None else ""
        style = (style_el.text or "").strip() if style_el is not None else ""

        pid = "KMZ" + hashlib.md5(f"{name}|{lon:.6f}|{lat:.6f}".encode()).hexdigest()[:10]
        # Prefer name-based detection (catches sub-brands like Georgetown
        # Pharmacy nested inside the Caring folder) and fall back to the
        # folder-colour mapping for records where the name is a sentinel like
        # "Jalan Sultan" (a KMZ artefact where a row has no business name).
        brand = detect_brand_from_name(name, default="") or _style_to_brand(style)
        rows.append({
            "pharmacy_id": pid,
            "name": name,
            "address": "",
            "latitude": lat,
            "longitude": lon,
            "brand": brand,
            "source": "KMZ",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# PMG Excel — Pharmacy, Medical & Dental Clinics branch list
# --------------------------------------------------------------------------------------

def parse_pmg_excel(path: str | Path,
                    segment_filter: str = "pharmacy") -> pd.DataFrame:
    """Parse the PMG 'outlets' Excel into the canonical pharmacy schema.

    The sheet has one row per branch with columns:
        Entity Legal Name | Branch Name | Segment | Region

    We filter to the 'Pharmacy' segment (dropping Medical / Dental / SCM / HQ /
    Lab / P&M / Physio / Wholesale / E-Commerce / Others), then build a
    geocode-friendly query by extracting the location hint from the branch
    name's parentheses.

    No lat/lon yet — `geocode_addresses_google` fills those in.
    """
    raw = pd.read_excel(path, sheet_name=0, header=1)
    raw = raw.dropna(axis=1, how="all")  # drop the empty left columns
    raw.columns = [str(c).strip() for c in raw.columns]
    df = raw[raw["Segment"].astype(str).str.lower().str.strip() == segment_filter.lower()].copy()

    def _extract_location(branch: str) -> str:
        """Pull the location hint out of parentheses, e.g.
        'AM PM Pharmacy (Ayer Hitam)' -> 'Ayer Hitam'.  Falls back to the
        whole branch name if no parens."""
        if not isinstance(branch, str):
            return ""
        m = re.search(r"\(([^)]+)\)", branch)
        return m.group(1).strip() if m else branch.strip()

    def _region_state(region: str) -> str:
        """Translate the coarse region column into a state hint Google can use."""
        if not isinstance(region, str):
            return ""
        r = region.lower()
        if "johor" in r:           return "Johor"
        if "east malaysia" in r:   return ""  # Sabah vs Sarawak — let Google guess
        if "north peninsula" in r: return ""  # Perlis/Kedah/Penang/Perak — let Google guess
        if "central" in r:         return "Selangor"
        return ""

    df = df.copy()
    df["name"] = df["Branch Name"].fillna(df["Entity Legal Name"]).astype(str).str.strip()
    df["chain"] = df["Entity Legal Name"].astype(str).str.strip()
    df["location_hint"] = df["Branch Name"].map(_extract_location)
    df["state"] = df["Region"].map(_region_state)

    # The `address` column is what the geocoder concatenates with state + Malaysia.
    # Use "<Branch name cleaned> <location hint>" so Google can lock onto the
    # business even when branch and hint overlap.
    df["address"] = df.apply(
        lambda r: (r["name"] + ", " + r["location_hint"]).strip(", "),
        axis=1,
    )

    df = df.reset_index(drop=True)

    def _brand_for(branch: str, entity: str) -> str:
        """Two-pass: detect on branch name first (most specific), then entity
        legal name (catches e.g. entity='PMG PHARMACY SDN BHD', branch='PR').
        Last fallback: first word of the entity, preserving short ACRONYMS."""
        hit = detect_brand_from_name(branch, default="")
        if hit:
            return hit
        hit = detect_brand_from_name(entity, default="")
        if hit:
            return hit
        first = (entity or "").split()[0] if entity else "Independent"
        # 'PMG' stays 'PMG', 'AM' stays 'AM'; longer words get title-cased.
        return first if first.isupper() and len(first) <= 5 else first.title()

    out = pd.DataFrame({
        "pharmacy_id": [
            "PMG" + hashlib.md5(f"{n}|{c}".encode()).hexdigest()[:10]
            for n, c in zip(df["name"], df["chain"])
        ],
        "name":    df["name"].values,
        "address": df["address"].values,
        "state":   df["state"].values,
        "brand":   [_brand_for(n, c) for n, c in zip(df["name"], df["chain"])],
        "source":  "PMG",
    })
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 2. NPRA PDF — "Senarai Premis" ROPA-1951 registered pharmacies
# --------------------------------------------------------------------------------------
#
# The PDF has one section per state (Perlis, Kedah, …) and within each section
# a six-column table:  No, Nama Premis, Alamat Premis, No. Tel, Nama Preseptor,
# Alamat Email Preseptor.  We drop the last two columns on load for privacy.

MALAYSIAN_STATES = [
    "Perlis", "Kedah", "Pulau Pinang", "Perak", "Selangor",
    "Wilayah Persekutuan Kuala Lumpur", "W.P Kuala Lumpur", "Kuala Lumpur",
    "Wilayah Persekutuan Putrajaya", "W.P Putrajaya", "Putrajaya",
    "Wilayah Persekutuan Labuan", "W.P Labuan", "Labuan",
    "Negeri Sembilan", "Melaka", "Johor",
    "Pahang", "Terengganu", "Kelantan",
    "Sabah", "Sarawak",
]


def _is_state_heading(row_text: str) -> Optional[str]:
    """Return a state tag if `row_text` matches a state heading row.

    Handles three shapes seen in the NPRA PDF:
      * plain state name ("Perlis", "Selangor", ...)
      * "Wilayah Persekutuan <FT>" (Kuala Lumpur / Labuan / Putrajaya)
      * combined heading "Wilayah Persekutuan Kuala Lumpur & Putrajaya" —
        returned as a special tag that the row loop disambiguates per-address.
    """
    t = (row_text or "").strip().lower()
    if not t:
        return None
    # Combined WP heading — we'll pick KL vs Putrajaya per-address later.
    if "kuala lumpur" in t and "putrajaya" in t:
        return "__WP_KL_OR_PUTRAJAYA__"
    for s in MALAYSIAN_STATES:
        sl = s.lower()
        if t == sl or t == f"daerah {sl}":
            return s
    # "Wilayah Persekutuan <X>" collapses to X if X is a state we know.
    if t.startswith("wilayah persekutuan "):
        tail = t[len("wilayah persekutuan "):].strip()
        for s in MALAYSIAN_STATES:
            if tail == s.lower():
                return s
    return None


def _resolve_wp_state(address: str) -> str:
    """For a row under the combined WP KL/Putrajaya heading, pick one."""
    a = (address or "").lower()
    if "putrajaya" in a:
        return "Putrajaya"
    return "Kuala Lumpur"


def parse_npra_pdf(path: str | Path) -> pd.DataFrame:
    """Parse the NPRA 'Senarai Premis' PDF into the canonical schema.

    Returns columns:
        pharmacy_id, name, address, phone, state, source
    (latitude/longitude are filled in later by `geocode_addresses`.)
    """
    import pdfplumber  # deferred: only needed when this loader is used

    rows: List[Dict] = []
    current_state: Optional[str] = None
    path = Path(path)
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                for raw_row in table:
                    # Normalize None → ""
                    cells = [(c or "").strip() for c in raw_row]
                    # State section headings look like a single-cell row whose
                    # non-empty cell matches a state name.
                    nonempty = [c for c in cells if c]
                    if len(nonempty) == 1:
                        maybe = _is_state_heading(nonempty[0])
                        if maybe:
                            current_state = maybe
                            continue
                    # Header row inside each state repeats the column titles.
                    if cells and cells[0].lower() == "no":
                        continue
                    # Entry rows start with a numeric No.
                    if not cells or not cells[0].strip().isdigit():
                        continue
                    if len(cells) < 4:
                        continue
                    no, name, address, phone = cells[0], cells[1], cells[2], cells[3]
                    # PII (preceptor name / email at cells[4], cells[5]) intentionally dropped.
                    if not name or not address:
                        continue
                    # Collapse line-wrapped whitespace introduced by pdfplumber.
                    name = re.sub(r"\s+", " ", name)
                    address = re.sub(r"\s+", " ", address)
                    # Resolve the shared WP heading to KL or Putrajaya per-row.
                    row_state = current_state or ""
                    if row_state == "__WP_KL_OR_PUTRAJAYA__":
                        row_state = _resolve_wp_state(address)
                    pid = "NPRA" + hashlib.md5(f"{no}|{name}|{address}".encode()).hexdigest()[:10]
                    rows.append({
                        "pharmacy_id": pid,
                        "name": name,
                        "address": address,
                        "phone": re.sub(r"\s+", " ", phone),
                        "state": row_state,
                        "source": "NPRA",
                    })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 3. Nominatim geocoding (free, rate-limited to 1 req/sec)
# --------------------------------------------------------------------------------------

NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "malaysia-pharmacy-mapping/1.0 (https://github.com/danniesidequestmaxxing/malaysia-pharmacy-mapping)"

GOOGLE_GEOCODE_ENDPOINT = "https://maps.googleapis.com/maps/api/geocode/json"

# Strip Malaysian state suffixes we already know, so we don't duplicate them
# when we append ", <state>, Malaysia" to the query. Also a few commercial
# prefixes that confuse Nominatim (it tends to lock onto the specific lot).
_STATE_SUFFIX_RE = re.compile(
    r",?\s*(perlis|kedah|pulau pinang|penang|perak|selangor|negeri sembilan|"
    r"melaka|malacca|johor|pahang|terengganu|kelantan|sabah|sarawak|"
    r"labuan|putrajaya|kuala lumpur|wilayah persekutuan[^,]*)\s*\.?$",
    flags=re.IGNORECASE,
)
_POSTCODE_TOWN_RE = re.compile(r"(\d{5})[,\s]+([A-Za-z][A-Za-z\s\.]+?)(?:,|$)")


def _geocode_one(query: str, timeout: int = 20) -> Optional[Tuple[float, float]]:
    """Single Nominatim request. Returns (lat, lon) or None."""
    params = {"q": query, "format": "json", "limit": 1, "countrycodes": "my"}
    r = requests.get(
        NOMINATIM_ENDPOINT,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    r.raise_for_status()
    hits = r.json()
    if not hits:
        return None
    try:
        return float(hits[0]["lat"]), float(hits[0]["lon"])
    except (KeyError, ValueError, TypeError):
        return None


def _build_query_cascade(address: str, state: str) -> List[str]:
    """Produce progressively-simpler queries for Nominatim.

    Strategy: try the full address first (highest precision); if that misses,
    fall back to the postcode + town extracted from the tail of the address
    (hits ~90% of Malaysian addresses with a ~500m accuracy); finally, just
    the state name as a last resort.
    """
    addr = address.strip()
    # Strip trailing state name/ASCII dot so we don't duplicate it.
    addr_no_state = _STATE_SUFFIX_RE.sub("", addr).strip().rstrip(",")
    queries: List[str] = []

    full = f"{addr_no_state}, {state}, Malaysia" if state else f"{addr_no_state}, Malaysia"
    queries.append(full)

    # Postcode + town from the tail of the address, e.g.
    # "... 05000 Alor Setar, Kedah"  ->  "05000 Alor Setar, Kedah, Malaysia"
    m = _POSTCODE_TOWN_RE.search(addr)
    if m:
        postcode, town = m.group(1), m.group(2).strip().rstrip(".")
        coarse = f"{postcode} {town}, {state}, Malaysia" if state else f"{postcode} {town}, Malaysia"
        if coarse not in queries:
            queries.append(coarse)

    # State-only fallback for rows that refuse to resolve — better to plot
    # them at the state capital than drop them entirely.
    if state:
        state_only = f"{state}, Malaysia"
        if state_only not in queries:
            queries.append(state_only)

    return queries


def _load_google_api_key() -> Optional[str]:
    """Resolve the Google Maps API key from env or a local .env file.

    Looked up in this order:
      1. `GOOGLE_MAPS_API_KEY` environment variable
      2. `.env` file in the CWD (`KEY=VALUE` format, shell-style comments OK)

    Never logged or printed.  Returns None if not found.
    """
    import os
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if key:
        return key.strip()
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            if k.strip() == "GOOGLE_MAPS_API_KEY":
                return v.strip().strip('"').strip("'")
    return None


def _geocode_one_google(query: str, api_key: str, timeout: int = 20) -> Optional[Tuple[float, float]]:
    """Single Google Geocoding API request. Returns (lat, lon) or None.

    Uses `components=country:MY` to bias results to Malaysia, and `region=my`
    for ccTLD hints. Errors (over-quota, invalid key, etc.) raise so the
    caller can decide whether to bail or continue.
    """
    params = {
        "address": query,
        "key": api_key,
        "components": "country:MY",
        "region": "my",
    }
    r = requests.get(GOOGLE_GEOCODE_ENDPOINT, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    status = payload.get("status")
    if status == "ZERO_RESULTS":
        return None
    if status != "OK":
        # Surface everything actionable (quota, key restrictions) to the caller.
        msg = payload.get("error_message") or status or "unknown error"
        raise RuntimeError(f"Google Geocoding returned {status}: {msg}")
    results = payload.get("results") or []
    if not results:
        return None
    loc = results[0].get("geometry", {}).get("location") or {}
    try:
        return float(loc["lat"]), float(loc["lng"])
    except (KeyError, ValueError, TypeError):
        return None


def geocode_addresses_google(
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    cache_path: str | Path = "data/geocode_cache_google.json",
    address_col: str = "address",
    state_col: str = "state",
    max_requests: Optional[int] = None,
    sleep_seconds: float = 0.05,   # Google QPS is generous; 20 req/s is safe
    verbose: bool = True,
) -> pd.DataFrame:
    """Geocode each row with Google Geocoding API, cached per-query on disk.

    Cache file is separate from the Nominatim cache so the two can coexist;
    when both exist, Google wins. Addresses are tried with the full address,
    falling back to postcode + town if the full address returns zero hits.
    """
    api_key = api_key or _load_google_api_key()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY not set. Put it in the shell env or in a "
            "local .env file (git-ignored).  See README.md."
        )

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, Optional[List[float]]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    out = df.copy()
    lats: List[Optional[float]] = []
    lons: List[Optional[float]] = []
    sources: List[str] = []
    new_calls = 0

    for _, row in out.iterrows():
        address = (row.get(address_col) or "").strip()
        state = (row.get(state_col) or "").strip()
        queries = _build_query_cascade(address, state)

        hit: Optional[List[float]] = None
        hit_kind = "miss"
        for i, q in enumerate(queries):
            if q in cache:
                if cache[q]:
                    hit = cache[q]
                    hit_kind = ["full", "postcode", "state"][i] if i < 3 else "other"
                    break
                continue  # cached miss — try next cascade step
            if max_requests is not None and new_calls >= max_requests:
                break
            try:
                result = _geocode_one_google(q, api_key)
            except Exception as e:
                if verbose:
                    # Don't print the query (may contain the address) at error level;
                    # print only the kind of failure.
                    print(f"  google geocode error: {type(e).__name__}: {e}")
                # Hard errors (bad key, over quota) are fatal — abort so we don't
                # waste more requests.
                msg = str(e).lower()
                if "api_key" in msg or "over_query_limit" in msg or "request_denied" in msg:
                    cache_path.write_text(json.dumps(cache), encoding="utf-8")
                    raise
                result = None
            cache[q] = list(result) if result else None
            new_calls += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            if new_calls % 50 == 0:
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
                if verbose:
                    hits = sum(1 for v in cache.values() if v)
                    print(f"  {new_calls} new calls, {hits}/{len(cache)} cache hits")
            if result:
                hit = result
                hit_kind = ["full", "postcode", "state"][i] if i < 3 else "other"
                break

        if hit:
            lats.append(hit[0]); lons.append(hit[1])
        else:
            lats.append(np.nan); lons.append(np.nan)
        sources.append(hit_kind)

    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    out["latitude"] = lats
    out["longitude"] = lons
    out["geocode_source"] = sources
    return out


def geocode_addresses(
    df: pd.DataFrame,
    cache_path: str | Path = "data/geocode_cache.json",
    address_col: str = "address",
    state_col: str = "state",
    max_requests: Optional[int] = None,
    sleep_seconds: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Geocode each row's address. Results cached per-query on disk.

    Uses a cascade: full address → postcode+town → state. Each cascade step
    consumes the rate-limit budget separately, so `max_requests` is the total
    new network calls across all rows and all cascade steps.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, Optional[List[float]]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    out = df.copy()
    lats: List[Optional[float]] = []
    lons: List[Optional[float]] = []
    sources: List[str] = []  # "full" | "postcode" | "state" | "miss"
    new_calls = 0

    for _, row in out.iterrows():
        address = (row.get(address_col) or "").strip()
        state = (row.get(state_col) or "").strip()
        queries = _build_query_cascade(address, state)

        hit: Optional[List[float]] = None
        hit_kind = "miss"
        for i, q in enumerate(queries):
            if q in cache:
                if cache[q]:
                    hit = cache[q]
                    hit_kind = ["full", "postcode", "state"][i] if i < 3 else "other"
                    break
                continue  # cached miss — try next cascade step
            if max_requests is not None and new_calls >= max_requests:
                break
            try:
                result = _geocode_one(q)
            except Exception as e:
                if verbose:
                    print(f"  geocode error: {e}")
                result = None
            cache[q] = list(result) if result else None
            new_calls += 1
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            if new_calls % 25 == 0:
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
                if verbose:
                    hits = sum(1 for v in cache.values() if v)
                    print(f"  {new_calls} new calls, {hits}/{len(cache)} cache hits")
            if result:
                hit = result
                hit_kind = ["full", "postcode", "state"][i] if i < 3 else "other"
                break

        if hit:
            lats.append(hit[0])
            lons.append(hit[1])
        else:
            lats.append(np.nan)
            lons.append(np.nan)
        sources.append(hit_kind)

    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    out["latitude"] = lats
    out["longitude"] = lons
    out["geocode_source"] = sources
    return out


# --------------------------------------------------------------------------------------
# 4. Merge multiple pharmacy sources with dedup
# --------------------------------------------------------------------------------------

def merge_pharmacy_sources(
    *dfs: pd.DataFrame,
    coord_precision: int = 4,     # ~11m at the equator; enough to collapse duplicates
) -> pd.DataFrame:
    """Union multiple pharmacy DataFrames, drop rows missing lat/lon, dedupe.

    Deduplication key: lowercased first word of name + rounded lat + rounded lon.
    That absorbs minor spelling / suffix differences between sources
    (e.g. 'Sunlight Pharmacy Sdn Bhd' vs 'Sunlight Pharmacy (KK)').
    """
    merged = pd.concat([d for d in dfs if d is not None and len(d) > 0],
                       ignore_index=True, sort=False)
    # Ensure canonical columns exist.
    for c in ["pharmacy_id", "name", "address", "brand", "source",
              "latitude", "longitude"]:
        if c not in merged.columns:
            merged[c] = "" if c not in ("latitude", "longitude") else np.nan

    merged = merged.dropna(subset=["latitude", "longitude"]).copy()
    merged["_dedup_key"] = (
        merged["name"].fillna("").str.lower().str.split().str[0].fillna("") + "|" +
        merged["latitude"].round(coord_precision).astype(str) + "|" +
        merged["longitude"].round(coord_precision).astype(str)
    )
    # Keep first occurrence (KMZ preferred over NPRA if listed first in dfs).
    merged = merged.drop_duplicates(subset="_dedup_key", keep="first")
    merged = merged.drop(columns=["_dedup_key"])

    # Fill any blank brand labels with "Other" so the map legend is clean.
    merged["brand"] = merged["brand"].replace("", "Other").fillna("Other")
    return merged.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 5. WorldPop → per-district aggregation
# --------------------------------------------------------------------------------------

def compute_worldpop_per_polygons(
    csv_path: str | Path,
    polygons_geojson: Dict,
    cache_path: str | Path,
    id_properties: Optional[List[str]] = None,
    chunksize: int = 1_000_000,
    force_refresh: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Aggregate a WorldPop raster CSV (lon,lat,value) to any polygon set.

    Generic version — originally written for ADM2 districts, now also used
    for ADM3 mukim and any arbitrary polygon set (e.g. Voronoi catchments).

    `id_properties` is the list of property keys from each feature to copy
    into the output DataFrame.  Defaults to whatever keys the first feature
    has (minus geometry internals).
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_refresh:
        if verbose:
            print(f"  using cached aggregation: {cache_path}")
        return pd.read_csv(cache_path)

    features = polygons_geojson["features"]
    if id_properties is None:
        id_properties = list(features[0].get("properties", {}).keys())

    polys = [shape(f["geometry"]) for f in features]
    props = [f["properties"] for f in features]
    tree = STRtree(polys)

    totals = np.zeros(len(polys), dtype=np.float64)
    unmatched = 0
    total_rows = 0
    t0 = time.time()
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        lons = chunk["longitude"].to_numpy()
        lats = chunk["latitude"].to_numpy()
        vals = chunk.iloc[:, 2].to_numpy()

        try:
            pts = np.empty(len(chunk), dtype=object)
            for i in range(len(chunk)):
                pts[i] = Point(lons[i], lats[i])
            input_idx, tree_idx = tree.query(pts, predicate="within")
            np.add.at(totals, tree_idx, vals[input_idx])
            matched = np.zeros(len(chunk), dtype=bool)
            matched[input_idx] = True
            unmatched += int((~matched).sum())
        except Exception:
            for lon, lat, v in zip(lons, lats, vals):
                pt = Point(lon, lat)
                found = False
                for idx in tree.query(pt):
                    if polys[idx].contains(pt):
                        totals[idx] += v
                        found = True
                        break
                if not found:
                    unmatched += 1

        total_rows += len(chunk)
        if verbose:
            print(f"  processed {total_rows:,} rows  ({time.time() - t0:.1f}s)")

    out = pd.DataFrame({key: [p.get(key) for p in props] for key in id_properties})
    out["population"] = totals.round().astype(int)
    if verbose:
        print(f"  done. {unmatched:,} cells outside all polygons. "
              f"Total pop: {int(out['population'].sum()):,}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return out


def compute_worldpop_per_district(
    csv_path: str | Path,
    districts_geojson: Dict,
    cache_path: str | Path = "data/worldpop_per_district.csv",
    chunksize: int = 500_000,
    force_refresh: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Back-compat shim — delegates to compute_worldpop_per_polygons.

    Output columns: district, state, population (int).
    """
    return compute_worldpop_per_polygons(
        csv_path=csv_path,
        polygons_geojson=districts_geojson,
        cache_path=cache_path,
        id_properties=["district", "state"],
        chunksize=chunksize,
        force_refresh=force_refresh,
        verbose=verbose,
    )
