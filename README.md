# Malaysia Pharmacy Mapping Dashboard

Interactive dashboard visualising registered pharmacies across Malaysia and the
population-to-pharmacy ratio per district (daerah).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app launches at <http://localhost:8501>. It defaults to the **Local
(NPRA PDF + KMZ + WorldPop)** data source, which serves the pre-processed
authoritative data committed in `./data/`.

To refresh any of the data sources, see [Refreshing data](#refreshing-data).

## Project layout

```
app.py               Streamlit dashboard (UI, map, charts, filters)
data_pipeline.py     Live-data loaders (OSM/DOSM/geoBoundaries) + spatial join + metrics
local_sources.py     KMZ / NPRA-PDF / WorldPop loaders + geocoding cascade
mock_data.py         Bundled sample pharmacies / districts / GeoJSON
refresh_data.py      One-shot CLI: pre-caches OSM + DOSM + geoBoundaries
geocode_npra.py      One-shot CLI: parses NPRA PDF and geocodes via Nominatim
requirements.txt     Python dependencies
data/                Cache directory + committed processed artefacts
  source/            User-provided raw files (KMZ committed; PDF git-ignored for privacy)
```

## Data sources

Four modes, switchable in the sidebar:

| Mode | Pharmacies | Population | District polygons |
|---|---|---|---|
| **Local (NPRA + KMZ + WorldPop)** (default) | Project Pharma KMZ (~957, chain-tagged) ∪ geocoded NPRA "Senarai Premis" (~594 ROPA-1951 premises) | WorldPop 2020 aggregated per ADM2 (100m grid → district sums) | geoBoundaries MYS ADM2 |
| **Live (OSM + DOSM)** | OSM Overpass `amenity=pharmacy` in Malaysia | DOSM `population_district` (latest year) | geoBoundaries MYS ADM2 |
| **Mock** | Bundled 300+ jittered points around 35 district centroids | Illustrative figures per district | Square polygons |
| **Custom CSV + GeoJSON** | `data/pharmacies.csv` you supply | `api.data.gov.my?id=population_district` | GeoJSON path/URL you supply |

**Privacy:** the raw NPRA PDF contains preceptor names and email addresses.
Those columns are stripped at parse time in `local_sources.parse_npra_pdf`
and never leave that function. The raw PDF is git-ignored; only the
pre-processed CSV (addresses + coordinates, no PII) is committed.

## Refreshing data

```bash
# Live-mode caches (OSM + DOSM + geoBoundaries)
python refresh_data.py              # respects 24h TTL
python refresh_data.py --force

# Re-geocode the NPRA PDF — two providers available
# 1) Google (recommended; ~95-100% hit rate, ~3 min at 50ms sleep)
cp .env.example .env                # then paste your GOOGLE_MAPS_API_KEY
python geocode_npra.py --provider google
# 2) OSM Nominatim (free, no key; ~40-60% hit rate with cascade, ~15-20 min)
python geocode_npra.py --provider nominatim --max 100   # rate-cap one batch

# Re-aggregate WorldPop from the raw CSV (requires the 543 MB file on disk)
python -c "
from local_sources import compute_worldpop_per_district
from data_pipeline import load_malaysia_districts_geojson
compute_worldpop_per_district(
    '/path/to/mys_general_2020.csv',
    load_malaysia_districts_geojson(),
    force_refresh=True)
"

# Pull pharmacies from Google Places API (New) for Johor and write
# data/pharmacies_google_johor.csv (loaded automatically by load_pharmacies).
# Cost: ~$25 in Places API calls. Requires Places API enabled in your GCP
# project on the same key already used for Geocoding.
python scripts/fetch_google_places_johor.py
```

### Caveats

* **OSM coverage** undercounts the NPRA registry. Use Local mode for the
  authoritative view; Live mode is the zero-setup fallback.
* **WorldPop 2020** is the latest public WorldPop release; Malaysia's total
  is ~32.3 M (2020 census baseline), vs. DOSM's 2024 estimate of ~34.0 M.
* **Nominatim geocoding** only resolves ~60-70% of Malaysian shophouse addresses
  precisely; the cascade falls back to postcode + town centroid for the rest,
  which is accurate to ~500m — fine for district-level aggregation, not for
  pin-point navigation.  Use the Google provider (above) for pin-point accuracy.

## Basemap tiles (Google Maps)

The sidebar's **Basemap** selector switches between Google Maps (Roadmap /
Satellite / Hybrid / Terrain), CartoDB Positron, and OpenStreetMap. The
Google entries hit Google's public `mt{0-3}.google.com/vt/lyrs=…` tile
endpoint — the same URLs the Google Maps site itself serves from. Strictly
speaking, Google's Maps ToS only licenses those tiles for use inside Google's
own products; they're widely used in open-source geospatial dashboards but
are not ToS-clean for a public commercial deploy. Pick **CartoDB Positron**
(OSM-derived, CC-BY) for a license-clean alternative. A fully-compliant
Google deploy would move to the Map Tiles API (session-token REST flow), a
larger rework out of scope here.

## Security notes

* `.env` is **git-ignored** — your `GOOGLE_MAPS_API_KEY` never enters the repo.
  Always restrict the key in the [GCP console](https://console.cloud.google.com/google/maps-apis/credentials)
  with an HTTP-referrer or IP allow-list before using it on shared infra.
* The raw NPRA PDF (`data/source/farmasi-komuniti.pdf`) is **git-ignored**
  because it contains preceptor names and email addresses. Only the
  PII-stripped `data/pharmacies_npra_geocoded.csv` (addresses + coords only)
  is committed.
* Deploy on Streamlit Cloud: set `GOOGLE_MAPS_API_KEY` in the app's
  **Settings → Secrets** panel; the app reads `st.secrets` before falling
  back to `.env`.

## Tech stack

- **Streamlit** — UI, sidebar filters, caching.
- **Folium + streamlit-folium** — Choropleth, `GeoJsonTooltip`, `MarkerCluster`.
- **Shapely (STRtree)** — Fast point-in-polygon spatial join.
- **Pandas / NumPy** — Tabular data and metric computation.
- **Plotly Express** — Top-N bar chart.
- **Requests** — data.gov.my + Overpass + geoBoundaries HTTPS calls.

## Known gotchas

- District-name spellings differ across sources (e.g. *Petaling* vs *Daerah Petaling*,
  *Pulau Pinang* vs *Penang*). `data_pipeline.normalize_district_names` canonicalizes
  both sides of the merge — extend the alias tables if you find new variants.
- DOSM publishes `population_district` values in **thousands**; the loader
  auto-detects and rescales so downstream code sees raw head counts.
- Streamlit reruns on every widget interaction — the `@st.cache_data` decorators
  on `load_all` and `build_metrics` keep the spatial join from re-running.
- Overpass public endpoints rate-limit heavily during peak hours. The loader
  tries three mirrors in order and falls back to a stale cache if all fail.
