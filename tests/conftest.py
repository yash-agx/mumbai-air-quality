"""Shared fixtures.

The three modules under test are loaded by path rather than imported: `data`
and `model` are plain directories with no __init__.py, and scripts/02_features.py
starts with a digit so its name is not a legal identifier. model/interpolate.py
already loads the features module this way, so this is the same trick.

Nothing here touches the network or data/, data/interim or data/processed. Every
test builds its own panel in memory, and the two functions that would read a
parquet off disk (station_cells) or an OSM point cloud (features_for) are
pointed at fixtures instead.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    # Registered so dataclasses/pickle-style lookups inside the module resolve,
    # and so a second load in the same session is cheap.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def clean_mod():
    return _load("aq_clean", "data/clean.py")


@pytest.fixture(scope="session")
def features_mod():
    return _load("aq_features", "scripts/02_features.py")


@pytest.fixture(scope="session")
def interp_mod():
    return _load("aq_interpolate", "model/interpolate.py")


# --------------------------------------------------------------------------
# Raw-panel builder, shaped like what data/fetch.py writes.
# --------------------------------------------------------------------------

RAW_COLUMNS = ["station_id", "station_name", "lat", "lon", "timestamp",
               "pollutant", "value", "flagged"]


def raw_rows(station_id, station_name, lat, lon, values, start="2024-01-01T00:00:00Z",
             freq="1h", flagged=None, timestamps=None):
    """Hourly rows for one station. `values` may contain None for a null reading."""
    n = len(values)
    ts = (pd.to_datetime(timestamps, utc=True) if timestamps is not None
          else pd.date_range(start, periods=n, freq=freq, tz="UTC"))
    return pd.DataFrame({
        "station_id": station_id,
        "station_name": station_name,
        "lat": lat,
        "lon": lon,
        "timestamp": ts,
        "pollutant": "pm25",
        "value": pd.array(values, dtype="float64"),
        "flagged": [False] * n if flagged is None else flagged,
    })[RAW_COLUMNS]


@pytest.fixture
def make_raw():
    """Concatenate per-station frames into one raw pull."""
    def _make(*frames):
        return pd.concat(frames, ignore_index=True)
    return _make


@pytest.fixture
def two_station_raw():
    """Two ungrouped stations, 48 clean hours each, ~2.9 km apart."""
    a = raw_rows(1001, "Alpha, Mumbai - MPCB", 19.10, 72.85,
                 list(20.0 + np.arange(48) % 7))
    b = raw_rows(1002, "Beta, Mumbai - IITM", 19.12, 72.87,
                 list(30.0 + np.arange(48) % 5))
    return pd.concat([a, b], ignore_index=True)


# --------------------------------------------------------------------------
# Station panel for the model, matching what model/interpolate.load() returns.
# --------------------------------------------------------------------------

@pytest.fixture
def panel(interp_mod):
    """(wide, feat): 4 stations in 4 CV groups over 200 fully-reported hours.

    200 hours clears fit_variogram's 100-shared-hour floor for every pair, and
    four one-station groups make each fold a real leave-one-group-out split.
    """
    rng = np.random.default_rng(0)
    ids = [1, 2, 3, 4]
    lat = [19.00, 19.05, 19.12, 19.20]
    lon = [72.82, 72.88, 72.85, 72.90]
    hours = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")

    # A shared city-wide level plus a per-station offset: the structure the
    # models are meant to separate, so a fold has something to find.
    level = 40.0 + 15.0 * np.sin(np.arange(200) / 6.0)
    offsets = np.array([-6.0, -2.0, 3.0, 8.0])
    values = (level[:, None] + offsets[None, :]
              + rng.normal(0, 2.0, size=(200, 4)))

    wide = pd.DataFrame(values, index=hours, columns=ids)
    wide.index.name = "timestamp"
    wide.columns.name = "station_id"

    feat = pd.DataFrame({
        "station_id": ids,
        "lat": lat,
        "lon": lon,
        "site": ["Alpha", "Beta", "Gamma", "Delta"],
        "cv_group": ["g1", "g2", "g3", "g4"],
        "dist_major_road_m": [120.0, 300.0, 55.0, 900.0],
        "road_density_500m": [18.0, 12.0, 25.0, 6.0],
        "road_density_1km": [16.0, 11.0, 22.0, 5.0],
        "dist_coast_m": [1500.0, 6000.0, 800.0, 12000.0],
        "dist_industrial_m": [400.0, 0.0, 2500.0, 5000.0],
        "building_density_500m": [900.0, 600.0, 1400.0, 200.0],
    }).set_index("station_id")
    return wide, feat
