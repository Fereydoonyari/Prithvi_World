# PrithviWORLD Workbench

**This is the main product in the repository.** Standalone web app — everything it needs lives in this folder.

```
software/
  engine/          scientific engine (HLS, Prithvi, dynamics)
  app/             FastAPI, SQLite CRUD, background jobs
  frontend/        Leaflet satellite dashboard
  config/          default AOIs and model ids
  data/            runtime: DB, HLS, embeddings, overlays (gitignored)
```

## Run

From this folder (`software/`):

```powershell
python -m pip install -r requirements.txt
python run.py
```

Open http://127.0.0.1:8000

Default login: `admin@workbench.local` / `admin` (local demo only).

NASA Earthdata credentials are required only for **Ingest HLS**. Encoding
downloads the Prithvi weights into `data/models/prithvi/` on first encode job.

Put HLSS30 GeoTIFFs in `data/hls/<aoi-slug>/HLS/` and click **Index HLS** to
register scenes without downloading.

## What the dashboard does

- Click the map to place a study area (CRUD).
- Index local HLS, encode frozen Prithvi tokens, run a multi-seed forecast
  evaluation as background jobs.
- Overlay a 14×14 token or NDVI change map on the 6.7 km HLS footprint.
- Export experiment metrics as CSV.

## Categories

| Category | In this folder |
|---|---|
| CRUD | users (register/login), AOIs, observations, embeddings, experiments, jobs, results |
| Third-party | Earthaccess, frozen Prithvi, rasterio, Leaflet, Chart.js |
| Complex | Fmask protocol, window builder, climatology ladder, nuisance floor, from-scratch dynamics, evaluation orchestrator, overlay composer |
