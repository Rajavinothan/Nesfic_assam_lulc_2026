# LandWatch Assam

Web tool for land use / land cover (LULC) change detection in Assam, built for NESFIC-D-21.
The user picks a district, the satellite data (Sentinel-2, Sentinel-1 or fused) and two years.
The backend runs a season-stacked Random Forest on Google Earth Engine, flags low-confidence areas,
detects significant change and returns map tiles plus a transition matrix in hectares.

```
backend/    FastAPI + Earth Engine (processing, tile URLs, statistics)
frontend/   Static page (Leaflet map, controls, results table)
```

## Run locally
1. Register a Google Cloud project for Earth Engine. Create a service account and key, or run `earthengine authenticate`.
2. Prepare a FeatureCollection asset of labelled points with an integer property `class`
   (values must match `CLASSES` in `backend/main.py`).
3. Backend:
   ```
   cd backend && python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env     # then fill in the values
   uvicorn main:app --reload --port 8000
   ```
4. Frontend (new terminal): `cd frontend && python -m http.server 8080`, then open http://localhost:8080.

To host the frontend on GitHub Pages, set `window.API_BASE` to your deployed API URL and add that Pages origin to `CORS_ORIGINS`.
The backend must run on a server, because it holds the Earth Engine credentials.

## Demo mode (no Earth Engine needed)
Open `frontend/index.html?demo` (or run the static server and visit http://localhost:8080/?demo).
Every control works with invented sample data and a visible "Demo mode" banner. Use it to explore the interface
or capture screenshots, and label any screenshot as demo data. Remove `?demo` to use the real backend.

## Status
| Implemented | Not yet implemented |
|---|---|
| District selector (FAO GAUL boundaries) | Tehsil and SOI sheet boundaries (needs PostGIS data from ASSAC) |
| Dataset registry: Sentinel-2, Sentinel-1, fused | Bhoonidhi and ASSAC imagery ingestion |
| Cloud masking, four seasonal composites, indices, SAR fusion | Object-based step (SNIC) and texture features |
| Level-I Random Forest with confidence fallback to "Uncertain" | Hierarchical Level-II models, benchmarking against other models |
| Dual-evidence change (class change + spectral magnitude + confidence) | Significant-change polygons, prioritisation scoring |
| Transition matrix, T1/T2/change map layers | Review queue, PDF report, job queue, login |

This code has not been run against a live Earth Engine project. Test it with your own credentials before relying on it.
The classes in `CLASSES` are placeholders and must be aligned with the NRSC scheme agreed with ASSAC.

## Licensing
Earth Engine requires a commercial licence for government operational use. This repository is a pilot.
See Google's Earth Engine noncommercial and commercial terms.
