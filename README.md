# PrithviWORLD

Interactive workbench that uses a **frozen** Earth-observation foundation model as the state layer of a land-surface world model.

```
HLS scene  →  frozen Prithvi-EO-2.0  →  z_t (14×14×768 tokens)
           →  dynamics model  →  ẑ_{t+Δ}
```

Prithvi stays frozen. The training target is the **next real embedding**, not a land-cover class.

**Primary deliverable:** [`software/`](software/) — FastAPI + Leaflet dashboard  
**Secondary:** [`notebooks/`](notebooks/) — exploratory analysis (not required to run the app)

---

## Quick start

```powershell
cd software
python -m pip install -r requirements.txt
copy .env.example .env   # optional: OpenAI key for the Explain feature
python run.py
```

Open http://127.0.0.1:8000  
Demo login: `admin@workbench.local` / `admin` (local demo only).

Full details: [`software/README.md`](software/README.md).

HLS imagery, embeddings, and model weights are **not** in git (too large). Prithvi downloads from Hugging Face on first encode; NASA Earthdata login is only needed for new HLS downloads.

---

## What the software does

| Capability | How |
|---|---|
| Study areas | Click the map to place / edit / delete AOIs (CRUD) |
| Imagery | Index local HLSS30 GeoTIFFs or ingest via Earthaccess |
| Encoding | Frozen Prithvi-EO-2.0 → 14×14×768 token grids |
| Forecasting | Multi-seed residual models (linear, token-LSTM, ConvLSTM, S4D) vs persistence, seasonal copy, and causal climatology |
| Evaluation | Nuisance floor, seed spread, CSV export |
| Visualization | Token / NDVI change overlays on the ~6.7 km footprint; decode previews |

### Layout

```
software/
  app/         FastAPI, auth, SQLite CRUD, background jobs
  engine/      HLS, Fmask, Prithvi, window builder, dynamics, overlays
  frontend/    Leaflet + Chart.js single-page UI
  config/      default AOIs and model ids
  data/        runtime only (gitignored)
```

---

## Stack

FastAPI · SQLite · Leaflet · Chart.js · PyTorch · rasterio · NASA HLS (HLSS30) · Earthaccess · Hugging Face (Prithvi-EO-2.0-100M-TL)

---

## Notebooks (optional)

Exploratory notebooks that mirror parts of the pipeline. The supported product is **`software/`** — you do not need these to use the app.

| Notebook | Notes |
|----------|--------|
| [`notebooks/prithvi_world.ipynb`](notebooks/prithvi_world.ipynb) | End-to-end encode → forecast → baselines |
| [`notebooks/wildfire_casestudy.ipynb`](notebooks/wildfire_casestudy.ipynb) | Latent-state response to a wildfire event |

---

## License

MIT — see [`LICENSE`](LICENSE).
