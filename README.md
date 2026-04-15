# Malaysia Pharmacy Mapping Dashboard

Interactive dashboard visualising registered pharmacies across Malaysia and the
population-to-pharmacy ratio per district (daerah).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Option A — run immediately on live data (recommended)
python refresh_data.py       # pre-caches pharmacies, population, boundaries
streamlit run app.py         # defaults to "Live (OSM + DOSM)" in the sidebar

# Option B — run offline on bundled mock data
streamlit run app.py         # then flip the sidebar radio to "Mock"
```

The app launches at <http://localhost:8501>.

## Project layout

```
app.py            Streamlit dashboard (UI, map, charts, filters)
data_pipeline.py  Data loaders (live + CSV) + spatial join + metrics
mock_data.py      Bundled sample pharmacies / districts / GeoJSON
refresh_data.py   One-shot CLI: pre-caches all live sources into ./data/
requirements.txt  Python dependencies
data/             Cache directory for the live-data sources
```

## Data sources

Three modes, switchable in the sidebar:

| Mode | Pharmacies | Population | District polygons |
|---|---|---|---|
| **Live (OSM + DOSM)** (default) | OSM Overpass `amenity=pharmacy` in Malaysia | DOSM `population_district` (latest year) | geoBoundaries `MYS/ADM2` + spatial join to `ADM1` for state names |
| **Mock** | Bundled 300+ jittered points around 35 real district centroids | Illustrative figures per district | Square polygons around district centroids |
| **Custom CSV + GeoJSON** | `data/pharmacies.csv` you supply | `api.data.gov.my?id=population_district` | GeoJSON path/URL you supply |

Live mode needs **no API keys**. First boot: `python refresh_data.py` populates
`./data/` and subsequent Streamlit loads use the cache (default TTL: 24h for
OSM + DOSM, 30 days for boundaries). The app still boots offline if cached
files exist.

### Caveats on OSM pharmacy coverage

OSM's community-sourced `amenity=pharmacy` tags cover most chains (Guardian,
Watsons, Alpro, Caring, CARiNG, etc.) and many independents, but under-count
NPRA's authoritative registry (~3,500 retail pharmacies). Treat ratios as
directional. To use the official NPRA list, drop it into
`data/pharmacies.csv` with columns
`pharmacy_id,name,address,license_no,district,state,latitude,longitude` and
switch the sidebar to **Custom CSV + GeoJSON**.

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
