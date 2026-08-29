"""Mumbai PM2.5 dashboard - interpolated surface over the metropolitan region.

Run: streamlit run app.py

Pass 1: the map only. No model card, no time series.

Reads the artefacts model/interpolate.py writes. If they are missing the app
says which script to run rather than failing on an import.
"""

import importlib.util
import time
from datetime import datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent
TZ = "Asia/Kolkata"

# Sequential blue, light -> dark, from the project palette. One hue: magnitude
# is a single quantity, so it gets a single ramp rather than a rainbow.
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

# Fixed across hours on purpose: a colour has to mean the same concentration at
# 3am as at 3pm, or the slider turns into a lie. Values above clamp to the top.
SCALE_MAX = 100.0

# Masked cells come off the hue ramp entirely rather than going pale. Pale means
# "a low number" on a sequential ramp, and these cells are not low, they are
# unvouched-for. Neutral grey says "no reading of this at all".
MASK_RGB = [176, 174, 168]
MASK_ALPHA = 70

GRID_DEFAULT = 25

st.set_page_config(page_title="Mumbai PM2.5", layout="wide")


def hex_to_rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


RAMP_RGB = np.array([hex_to_rgb(h) for h in RAMP], dtype=float)


def ramp_colors(values):
    """Map concentrations onto the sequential ramp, NaN-safe."""
    v = np.clip(np.nan_to_num(values, nan=0.0) / SCALE_MAX, 0, 1)
    pos = v * (len(RAMP_RGB) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(RAMP_RGB) - 1)
    t = (pos - lo)[:, None]
    return (RAMP_RGB[lo] * (1 - t) + RAMP_RGB[hi] * t).round().astype(int)


@st.cache_resource(show_spinner="Loading OSM layers and model...")
def engine():
    """The model module, with its KD-trees and model card already warm.

    cache_resource, not cache_data: this holds live scipy KD-trees and a loaded
    module, none of which survive a pickle round trip. Built once per process,
    so the OSM point clouds are not re-read on every widget change.
    """
    spec = importlib.util.spec_from_file_location(
        "interpolate", ROOT / "model" / "interpolate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.production()          # station panel, tuned power, calibrations
    mod.osm_features()._index()   # KD-trees behind the extrapolation mask
    return mod


@st.cache_data(show_spinner=False)
def panel():
    """Hourly readings keyed by station, plus each station's coordinates."""
    mod = engine()
    wide, feat, _ = mod.load()
    stations = feat.reset_index()[["station_id", "site", "lat", "lon"]]
    return wide, stations


@st.cache_data(show_spinner=False)
def surface(ts_iso, resolution):
    """One hour's surface. Keyed by the hour, so revisiting one is free."""
    s = engine().predict_surface(pd.Timestamp(ts_iso), grid_resolution=resolution)
    return s.lats, s.lons, s.values, s.sigma, s.mask, s.meta


def cell_polygons(lats, lons):
    """Rectangles for each grid cell, as pydeck wants them: rings of [lon, lat]."""
    dlat = (lats[1] - lats[0]) / 2 if len(lats) > 1 else 0.01
    dlon = (lons[1] - lons[0]) / 2 if len(lons) > 1 else 0.01
    glon, glat = np.meshgrid(lons, lats)
    a, o = glat.ravel(), glon.ravel()
    return [[[x - dlon, y - dlat], [x + dlon, y - dlat],
             [x + dlon, y + dlat], [x - dlon, y + dlat]] for x, y in zip(o, a)]


# ---------------------------------------------------------------- sidebar
t_start = time.perf_counter()

try:
    wide, stations = panel()
except SystemExit as e:
    st.error(str(e))
    st.stop()

local_index = wide.index.tz_convert(TZ)
first, last = local_index.min(), local_index.max()

# Readings are stamped on the UTC hour and IST is UTC+5:30, so every reading
# lands at half past in local time. The slider picks a whole IST hour and we
# floor to the UTC hour that contains it, which is what predict_surface does
# internally -- doing it here too keeps the station dots on the same hour as the
# surface. The caption prints the timestamp actually drawn, half past and all,
# rather than the whole hour the slider suggests.
live_utc = wide.index[wide.notna().any(axis=1).to_numpy()]
# Open on the newest hour that has readings. Nudging past the half hour first
# means flooring lands back on that exact stamp rather than an hour short.
latest = (live_utc.max().tz_convert(TZ) + pd.Timedelta(minutes=30))

st.sidebar.title("Mumbai PM2.5")
st.sidebar.caption(f"{len(stations)} stations - {first:%d %b %Y} to {last:%d %b %Y} (IST)")

date = st.sidebar.date_input("Date", value=latest.date(),
                             min_value=first.date(), max_value=last.date())
hour = st.sidebar.slider("Hour (IST)", 0, 23, int(latest.hour))
show_sigma = st.sidebar.toggle("Show uncertainty", value=False,
                               help="Fade cells by the model's own error bar")
resolution = st.sidebar.select_slider("Grid", [15, 25, 40, 60], value=GRID_DEFAULT)

ts = pd.Timestamp(datetime.combine(date, dtime(hour)), tz=TZ).tz_convert("UTC").floor("h")

# ---------------------------------------------------------------- surface
t_surface = time.perf_counter()
lats, lons, values, sigma, mask, meta = surface(ts.isoformat(), resolution)
t_surface = time.perf_counter() - t_surface

if meta["n_stations"] == 0:
    same_day = local_index[(local_index.date == date)
                           & wide.notna().any(axis=1).to_numpy()]
    hint = (f"Hours with data on {date:%d %b}: "
            + ", ".join(f"{h:02d}" for h in sorted({t.hour for t in same_day}))
            if len(same_day) else f"No station reported at all on {date:%d %b}.")
    st.warning(f"No station reported for {date:%d %b %Y} {hour:02d}:00 IST. {hint}")
    st.stop()

flat_v, flat_s, flat_m = values.ravel(), sigma.ravel(), mask.ravel()
colors = ramp_colors(flat_v)

if show_sigma:
    # Uncertainty as an alpha channel, so hue keeps meaning concentration.
    lo, hi = np.nanmin(flat_s), np.nanmax(flat_s)
    conf = 1.0 - (flat_s - lo) / (hi - lo) if hi > lo else np.ones_like(flat_s)
    alpha = (90 + 110 * conf).round().astype(int)
else:
    alpha = np.full(len(flat_v), 175)

fill = np.column_stack([colors, alpha])
fill[flat_m] = MASK_RGB + [MASK_ALPHA]

cells = pd.DataFrame({
    "polygon": cell_polygons(lats, lons),
    "fill": [list(map(int, c)) for c in fill],
    "tip": [f"<b>{v:.1f} ug/m3</b><br/>+/- {s:.1f} (1 sigma)"
            + ("<br/><i>outside training range</i>" if m else "")
            for v, s, m in zip(flat_v, flat_s, flat_m)],
})

# ---------------------------------------------------------------- stations
readings = wide.loc[ts] if ts in wide.index else pd.Series(dtype=float)
assert ts in wide.index or meta["n_stations"] == 0, "surface and dots disagree"
pts = stations.copy()
pts["value"] = pts["station_id"].map(readings).astype(float)
live = pts["value"].notna()

pts["fill"] = [list(map(int, c)) + [255] for c in ramp_colors(pts["value"].to_numpy())]
pts.loc[~live, "fill"] = pd.Series([[140, 138, 132, 90]] * (~live).sum(),
                                   index=pts.index[~live])
pts["tip"] = [
    f"<b>{s}</b><br/>{v:.1f} ug/m3" if np.isfinite(v) else f"<b>{s}</b><br/>not reporting"
    for s, v in zip(pts["site"], pts["value"])]

# ---------------------------------------------------------------- map
layers = [
    pdk.Layer("PolygonLayer", cells, get_polygon="polygon", get_fill_color="fill",
              stroked=False, pickable=True),
    # White ring so a station stays legible against the darkest ramp steps.
    pdk.Layer("ScatterplotLayer", pts, get_position=["lon", "lat"],
              get_fill_color="fill", get_line_color=[255, 255, 255],
              line_width_min_pixels=2, stroked=True, radius_min_pixels=6,
              radius_max_pixels=9, get_radius=200, pickable=True),
]

st.pydeck_chart(pdk.Deck(
    layers=layers,
    initial_view_state=pdk.ViewState(latitude=19.12, longitude=72.95, zoom=9.2),
    map_style="light",
    tooltip={"html": "{tip}", "style": {"backgroundColor": "#0b0b0b",
                                        "color": "#fff", "fontSize": "12px"}},
), height=620)

# ---------------------------------------------------------------- legend
swatches = "".join(
    f'<span style="display:inline-block;width:26px;height:12px;background:{c}"></span>'
    for c in RAMP)
st.markdown(
    f'<div style="font-size:13px">{swatches} &nbsp;0 &rarr; {SCALE_MAX:.0f}+ ug/m3'
    f' &nbsp;&nbsp;<span style="display:inline-block;width:26px;height:12px;'
    f'background:rgb({MASK_RGB[0]},{MASK_RGB[1]},{MASK_RGB[2]});opacity:.5"></span>'
    f' outside training range</div>', unsafe_allow_html=True)

total = time.perf_counter() - t_start
masked_pct = 100 * meta["masked_fraction"]
st.caption(
    f"{ts.tz_convert(TZ):%d %b %Y %H:%M} IST - {meta['n_stations']} stations "
    f"reporting - {meta['cell_km']:.1f} km cells - "
    f"{masked_pct:.0f}% masked - surface {1000 * t_surface:.0f} ms, "
    f"page {1000 * total:.0f} ms")
