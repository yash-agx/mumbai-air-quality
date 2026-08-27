# Master Prompt — Air Quality Dashboard

*Streamlit + spatial ML · Project Option A*

> Paste everything below into a fresh Claude Code session, in an empty project folder. Run this one first — it is the larger build.

---

I'm building an interactive web dashboard that predicts and visualises PM2.5 across Mumbai by interpolating between sparse ground sensors. This is a portfolio project for university applications, so it needs to be deployed and publicly usable, not just a notebook.

## Stack (do not substitute)

- Python 3.11, pandas for data handling
- scikit-learn for the interpolation model
- Streamlit for the UI
- Plotly for charts, pydeck or streamlit-folium for the map
- Deployed to Streamlit Community Cloud

## Data

Ground truth comes from CPCB (Central Pollution Control Board) and SAFAR monitoring stations in the Mumbai metropolitan region. There are only a handful of stations, so the core problem is estimating pollution at locations *between* them.

## Build this in phases

**Stop after each phase and show me what you've got before continuing.**

### Phase 1 — Data pipeline

Write `data/fetch.py` and `data/clean.py`.

- Ingest station-level readings into a tidy long-format DataFrame: `station_id, lat, lon, timestamp, pollutant, value`
- Handle the things that will actually break: missing hours, stations that go offline for days, negative or absurd sensor values, inconsistent station naming across sources
- Cache cleaned output to parquet so the app doesn't refetch on every run
- Print a data quality report: coverage per station, % missing, date range

Do not write the model yet.

### Phase 2 — Interpolation model

Write `model/interpolate.py`.

- Start with inverse distance weighting as a baseline — I need something to beat
- Then fit a proper spatial model. Gaussian process regression with a spatial kernel, or random forest with engineered features (distance to nearest station, distance to coast, hour of day, day of week, distance to major roads if I can source that)
- **Validation must be spatial cross-validation, not random k-fold.** Hold out entire stations, not random rows. Random splits leak spatial information and will give me a fake-good score.
- Report RMSE and MAE per held-out station, and compare against the IDW baseline
- Expose a function `predict_surface(timestamp, pollutant, grid_resolution)` returning a grid of predictions plus per-cell uncertainty

### Phase 3 — Streamlit app

Write `app.py`.

**Sidebar:** date picker, hour slider, pollutant dropdown (PM2.5 / PM10 / NO2), toggle for "show uncertainty", toggle for "show sensor locations".

**Main panel, top:** map of Mumbai with the interpolated pollution surface as a coloured overlay. Use pydeck or folium — Streamlit's native `st.map` cannot render a continuous surface and will not work here.

**Main panel, middle:** Plotly time-series for whichever point the user clicks on the map, showing predicted values with a confidence band.

**Main panel, bottom:** small model card — which model is running, spatial CV scores, how it compares to the IDW baseline, and an honest note about what the model does badly.

Use `@st.cache_data` on the data loading and `@st.cache_resource` on the model so it doesn't refit on every interaction.

### Phase 4 — Deploy

- `requirements.txt` with pinned versions
- README with a screenshot, a one-paragraph explanation of the interpolation approach, and the spatial CV results table
- Walk me through pushing to GitHub and connecting Streamlit Community Cloud

## Constraints

- Do not scaffold folders or files we haven't reached yet
- Do not write a config system, plugin architecture, or abstract base classes. This is one dashboard.
- If a phase is taking more than a couple of files, tell me and we simplify
- Comments only where the logic is non-obvious. No docstrings on one-line functions.
