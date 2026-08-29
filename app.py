"""Mumbai PM2.5 dashboard - interpolated surface over the metropolitan region.

Run: streamlit run app.py

Pass 2: map, mask explanation, model card. No time series yet.

Reads the artefacts model/interpolate.py writes. If they are missing the app
says which script to run rather than failing on an import.
"""

import importlib.util
import json
import time
from datetime import datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent
TZ = "Asia/Kolkata"

# Diverging blue <-> red about a neutral grey. Concentrations at one hour sit in
# a narrow band, so an absolute 0-100 ramp renders the whole city as one shade;
# centring on the hour's own median spends the full ramp on the differences that
# actually exist. Blue is cleaner than the median, red is dirtier. The two arms
# are lightness-matched (max dL 0.04) so neither side shouts louder.
DIV_LOW = ["#0d366b", "#1c5cab", "#2a78d6", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
DIV_MID = "#f0efec"
DIV_HIGH = ["#fbd5d4", "#f4a9a8", "#ec7f7e", "#e34948", "#c73332", "#a02524", "#6b1717"]

# One hue, light to dark, for the absolute scale.
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_MAX = 100.0

# Masked cells come off the value ramp entirely rather than going pale. Pale
# means "a low number" on a ramp; these cells are not low, they are unvouched-for.
MASK_GREY = [176, 174, 168]
MASK_ALPHA = 70

# Mask-reason layer. Validated as a categorical pair: CVD dE 9.2, normal-vision
# 27.6. Aqua is sub-3:1 on a light surface, so the legend labels both in text.
REASON_COLORS = {1: [27, 175, 122], 2: [235, 104, 52], 3: [140, 138, 132]}
REASON_LABELS = {1: "empty - fewer roads or buildings than any station",
                 2: "remote - farther from a road or industry than any station",
                 3: "other - off the band on some other feature"}

# The grid is a rectangle over a coastal city, and a hard rectangular edge of
# colour reads as a rendering bug rather than as data. Fade the outer band of
# cells to nothing so the surface dissolves instead of stopping.
FEATHER = 0.10

GRID_DEFAULT = 25

st.set_page_config(page_title="Mumbai PM2.5", layout="wide")


def hex_to_rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


DIV_RGB = np.array([hex_to_rgb(h) for h in DIV_LOW + [DIV_MID] + DIV_HIGH], float)
SEQ_RGB = np.array([hex_to_rgb(h) for h in SEQ], float)


def sample(ramp, t):
    """Interpolate a 0-1 position along an RGB ramp."""
    pos = np.clip(t, 0, 1) * (len(ramp) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(ramp) - 1)
    f = (pos - lo)[:, None]
    return (ramp[lo] * (1 - f) + ramp[hi] * f).round().astype(int)


def diverging_colors(values, centre, half):
    """Signed distance from the hour's median, mapped onto the diverging ramp."""
    t = np.clip((np.nan_to_num(values, nan=centre) - centre) / half, -1, 1)
    return sample(DIV_RGB, (t + 1) / 2)


def absolute_colors(values):
    return sample(SEQ_RGB, np.nan_to_num(values, nan=0.0) / SEQ_MAX)


def feather_alpha(n):
    """1 in the interior, tapering to 0 over the outer FEATHER of each axis."""
    d = np.minimum(np.arange(n), n - 1 - np.arange(n)) / max(n - 1, 1)
    f = np.clip(d / FEATHER, 0, 1)
    return np.minimum.outer(f, f).ravel()


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
    mod.production()
    mod.osm_features()._index()
    return mod


@st.cache_data(show_spinner=False)
def panel():
    mod = engine()
    wide, feat, _ = mod.load()
    return wide, feat.reset_index()[["station_id", "site", "lat", "lon"]]


@st.cache_data(show_spinner=False)
def model_card():
    return json.loads(engine().CARD_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def surface(ts_iso, resolution):
    """One hour's surface. Keyed by the hour, so revisiting one is free."""
    s = engine().predict_surface(pd.Timestamp(ts_iso), grid_resolution=resolution)
    return s.lats, s.lons, s.values, s.sigma, s.mask, s.mask_kind, s.meta


def cell_polygons(lats, lons):
    """Rectangles for each grid cell, as pydeck wants them: rings of [lon, lat]."""
    dlat = (lats[1] - lats[0]) / 2 if len(lats) > 1 else 0.01
    dlon = (lons[1] - lons[0]) / 2 if len(lons) > 1 else 0.01
    glon, glat = np.meshgrid(lons, lats)
    return [[[x - dlon, y - dlat], [x + dlon, y - dlat],
             [x + dlon, y + dlat], [x - dlon, y + dlat]]
            for x, y in zip(glon.ravel(), glat.ravel())]


def swatch(color, label):
    c = f"rgb({color[0]},{color[1]},{color[2]})"
    return (f'<span style="display:inline-block;width:14px;height:14px;'
            f'background:{c};vertical-align:-2px"></span> {label}')


# ---------------------------------------------------------------- sidebar
t_start = time.perf_counter()

try:
    wide, stations = panel()
    card = model_card()
except SystemExit as e:
    st.error(str(e))
    st.stop()

local_index = wide.index.tz_convert(TZ)
first, last = local_index.min(), local_index.max()

# Readings are stamped on the UTC hour and IST is UTC+5:30, so every reading
# lands at half past in local time. The slider picks a whole IST hour and we
# floor to the UTC hour containing it, which is what predict_surface does
# internally -- doing it here too keeps the station dots on the same hour as the
# surface. The caption prints the timestamp actually drawn, half past and all.
live_utc = wide.index[wide.notna().any(axis=1).to_numpy()]
latest = live_utc.max().tz_convert(TZ) + pd.Timedelta(minutes=30)

st.sidebar.title("Mumbai PM2.5")
st.sidebar.caption(f"{len(stations)} stations - {first:%d %b %Y} to {last:%d %b %Y} (IST)")

date = st.sidebar.date_input("Date", value=latest.date(),
                             min_value=first.date(), max_value=last.date())
hour = st.sidebar.slider("Hour (IST)", 0, 23, int(latest.hour))
scale = st.sidebar.radio(
    "Colour scale", ["Relative to this hour", "Absolute"],
    help="Relative centres the ramp on the hour's median, so the spread across "
         "the city is visible. Absolute fixes it at 0-100 ug/m3.")
explain = st.sidebar.toggle("Explain the mask", value=False,
                            help="Colour masked cells by why they are masked")
resolution = st.sidebar.select_slider("Grid", [15, 25, 40, 60], value=GRID_DEFAULT)

ts = pd.Timestamp(datetime.combine(date, dtime(hour)), tz=TZ).tz_convert("UTC").floor("h")

# ---------------------------------------------------------------- surface
t_surface = time.perf_counter()
lats, lons, values, sigma, mask, kind, meta = surface(ts.isoformat(), resolution)
t_surface = time.perf_counter() - t_surface

if meta["n_stations"] == 0:
    same_day = local_index[(local_index.date == date)
                           & wide.notna().any(axis=1).to_numpy()]
    hint = (f"Hours with data on {date:%d %b}: "
            + ", ".join(f"{h:02d}" for h in sorted({t.hour for t in same_day}))
            if len(same_day) else f"No station reported at all on {date:%d %b}.")
    st.warning(f"No station reported for {date:%d %b %Y} {hour:02d}:00 IST. {hint}")
    st.stop()

flat_v, flat_s = values.ravel(), sigma.ravel()
flat_m, flat_k = mask.ravel(), kind.ravel()
readings = wide.loc[ts].dropna()

# Centre on the median of the cells actually drawn, not on the median of the
# reporting stations. IDW at power 0.5 is a broad weighted average, so the
# surface tracks the station *mean*, and PM2.5 is right-skewed enough that the
# station median can leave 98% of the map on one side of the ramp -- measured at
# 0-2% of cells below it on three hours out of four. The surface median splits
# the map ~50/50 by construction, which is what makes the two arms readable.
inrange = flat_v[~flat_m]
centre = float(np.median(inrange)) if inrange.size else float(readings.median())
half = max(float(np.percentile(np.abs(inrange - centre), 95)) if inrange.size else 0.0, 2.0)
station_median = float(readings.median())

relative = scale.startswith("Relative")
colors = diverging_colors(flat_v, centre, half) if relative else absolute_colors(flat_v)

alpha = (175 * feather_alpha(len(lats))).round().astype(int)
fill = np.column_stack([colors, alpha])

if explain:
    # Mute the values so the reasons read; the map is answering a different
    # question in this mode.
    fill[:, :3] = 232
    fill[:, 3] = (alpha * 0.45).astype(int)
    for k, rgb in REASON_COLORS.items():
        hit = flat_k == k
        fill[hit, :3] = rgb
        fill[hit, 3] = (alpha[hit] * 0.85).astype(int)
else:
    fill[flat_m, :3] = MASK_GREY
    fill[flat_m, 3] = (alpha[flat_m] * (MASK_ALPHA / 175)).astype(int)

reason_of = {0: "", 1: "empty", 2: "remote from roads/industry", 3: "other"}
cells = pd.DataFrame({
    "polygon": cell_polygons(lats, lons),
    "fill": [list(map(int, c)) for c in fill],
    "tip": [f"<b>{v:.1f} ug/m3</b> ({v - centre:+.1f} vs map median)"
            f"<br/>+/- {s:.1f} (1 sigma)"
            + (f"<br/><i>masked: {reason_of[int(k)]}</i>" if m else "")
            for v, s, m, k in zip(flat_v, flat_s, flat_m, flat_k)],
})

# ---------------------------------------------------------------- stations
pts = stations.copy()
pts["value"] = pts["station_id"].map(readings).astype(float)
live = pts["value"].notna()
vals = pts["value"].to_numpy()
pts["fill"] = [list(map(int, c)) + [255] for c in
               (diverging_colors(vals, centre, half) if relative else absolute_colors(vals))]
pts.loc[~live, "fill"] = pd.Series([[140, 138, 132, 90]] * int((~live).sum()),
                                   index=pts.index[~live])
pts["tip"] = [f"<b>{s}</b><br/>{v:.1f} ug/m3 ({v - centre:+.1f} vs map median)"
              if np.isfinite(v) else f"<b>{s}</b><br/>not reporting"
              for s, v in zip(pts["site"], vals)]

# ---------------------------------------------------------------- map
st.pydeck_chart(pdk.Deck(
    layers=[
        pdk.Layer("PolygonLayer", cells, get_polygon="polygon",
                  get_fill_color="fill", stroked=False, pickable=True),
        # White ring so a station stays legible against the darkest ramp steps.
        pdk.Layer("ScatterplotLayer", pts, get_position=["lon", "lat"],
                  get_fill_color="fill", get_line_color=[255, 255, 255],
                  line_width_min_pixels=2, stroked=True, radius_min_pixels=6,
                  radius_max_pixels=9, get_radius=200, pickable=True),
    ],
    initial_view_state=pdk.ViewState(latitude=19.12, longitude=72.95, zoom=9.2),
    map_style="light",
    tooltip={"html": "{tip}", "style": {"backgroundColor": "#0b0b0b",
                                        "color": "#fff", "fontSize": "12px"}},
), height=600)

# ---------------------------------------------------------------- legend
if explain:
    counts = meta["mask_reasons"]
    total = values.size
    parts = [swatch(REASON_COLORS[k], f"{REASON_LABELS[k]} "
                    f"({100 * counts[n] / total:.0f}%)")
             for k, n in ((1, "_empty"), (2, "_remote"), (3, "_other")) if counts[n]]
    st.markdown(" &nbsp;&nbsp; ".join(parts) + "<br/><span style='color:#52514e'>"
                "Pale cells are inside the training range. <b>Remote</b> is the one "
                "that matters: inhabited ground the model has no analogue for.</span>",
                unsafe_allow_html=True)
elif relative:
    bar = "".join(f'<span style="display:inline-block;width:19px;height:12px;'
                  f'background:{c}"></span>' for c in DIV_LOW + [DIV_MID] + DIV_HIGH)
    st.markdown(
        f'<div style="font-size:13px">{bar}<br/>'
        f'<span style="color:#52514e">{centre - half:.0f}'
        f' &nbsp;&larr; cleaner &nbsp;&nbsp; <b>map median {centre:.0f} ug/m3</b>'
        f' &nbsp;&nbsp; dirtier &rarr; &nbsp;{centre + half:.0f}</span></div>',
        unsafe_allow_html=True)
else:
    bar = "".join(f'<span style="display:inline-block;width:20px;height:12px;'
                  f'background:{c}"></span>' for c in SEQ)
    st.markdown(f'<div style="font-size:13px">{bar} &nbsp;0 &rarr; {SEQ_MAX:.0f}+ ug/m3'
                f' &nbsp;&nbsp;{swatch(MASK_GREY, "outside training range")}</div>',
                unsafe_allow_html=True)

total_ms = 1000 * (time.perf_counter() - t_start)
st.caption(
    f"{ts.tz_convert(TZ):%d %b %Y %H:%M} IST - {meta['n_stations']} stations "
    f"reporting - {meta['cell_km']:.1f} km cells - "
    f"{100 * meta['masked_fraction']:.0f}% masked - "
    f"station median {station_median:.1f} ug/m3 - "
    f"surface {1000 * t_surface:.0f} ms, page {total_ms:.0f} ms")

# ---------------------------------------------------------------- model card
st.divider()
st.subheader("Model card")

cv = card["cv"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Production model", card["model"].upper())
c2.metric("Spatial CV RMSE", f"{cv['rmse']:.2f}", help="ug/m3, leave-one-group-out")
c3.metric("Spatial CV MAE", f"{cv['mae']:.2f}")
c4.metric("Held out", f"{cv['n_stations']} stations",
          help=f"{cv['n_folds']} folds, {cv['n_station_hours']:,} station-hours")

left, right = st.columns([1, 1.35])

comp = (pd.DataFrame(card["comparison"]).T
          .rename_axis("model").reset_index().sort_values("RMSE"))
base = comp.loc[comp["model"] == "idw", "RMSE"].iloc[0]
comp["vs IDW"] = ((comp["RMSE"] - base) / base * 100).map("{:+.2f}%".format)
left.dataframe(comp, hide_index=True, width="stretch")

right.markdown(f"""
**No method beat city averaging by more than 0.5%.** Simply averaging every
reporting station scores {card['comparison']['regional_mean']['RMSE']:.2f} RMSE;
the winner, IDW, scores {base:.2f}. That gap is the entire value the spatial
model adds.

Why: once each hour's city-wide level is removed, spatial correlation is flat
beyond about 2 km, while the stations sit 1-40 km apart. The correlation is gone
before the second-nearest station, so there is very little for any interpolation
method to exploit. Kriging and gradient-boosted trees on land-use features both
lost to plain distance weighting.

**Read the map as the city average, tilted slightly** - not as a resolved
pollution field. Uncertainty is +/- {card['uncertainty']['a']:.1f} +
{card['uncertainty']['b']:.2f} x the prediction, calibrated on held-out
stations; it tracks how dirty the air is, not how far you are from a monitor.
""")
