"""Mumbai PM2.5 dashboard - interpolated air quality across the metro region.

Run: streamlit run app.py

Pass 3: readable for a resident, not a data scientist. Map, AQI tooltips, a
provenance panel on click, model card. No time series yet.

Reads the artefacts model/interpolate.py writes. If they are missing the app
says which script to run rather than failing on an import.
"""

import dataclasses
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

# --------------------------------------------------------------------------
# CPCB National Air Quality Index - PM2.5 sub-index.
#
# Breakpoints as published by the Central Pollution Control Board: each band is
# a concentration range in ug/m3 mapped linearly onto an AQI range, with the
# category name and CPCB's own wording for the health impact.
#
# One honest caveat, surfaced in the technical details: the official index is
# computed on a 24-hour average. This dashboard is hourly, so what is shown is
# the AQI those breakpoints give for a single hour's concentration -- the right
# scale and the right categories, but not the official daily number.
#
# The top band has no published upper concentration. 380 ug/m3 is the widely
# used convention for pinning AQI 500; anything above simply reads 500.
# --------------------------------------------------------------------------
CPCB_PM25 = [
    #  conc_lo conc_hi  aqi_lo aqi_hi  category
    (0.0, 30.0, 0, 50, "Good"),
    (30.0, 60.0, 51, 100, "Satisfactory"),
    (60.0, 90.0, 101, 200, "Moderate"),
    (90.0, 120.0, 201, 300, "Poor"),
    (120.0, 250.0, 301, 400, "Very Poor"),
    (250.0, 380.0, 401, 500, "Severe"),
]

# CPCB's published "Associated Health Impacts", quoted rather than paraphrased.
# Source: Central Pollution Control Board, National Air Quality Index (2014),
# the table of AQI categories and their associated health impacts.
CPCB_HEALTH = {
    "Good": "Minimal impact.",
    "Satisfactory": "May cause minor breathing discomfort to sensitive people.",
    "Moderate": "May cause breathing discomfort to people with lung disease such "
                "as asthma, and discomfort to people with heart disease, children "
                "and older adults.",
    "Poor": "May cause breathing discomfort to people on prolonged exposure, and "
            "discomfort to people with heart disease.",
    "Very Poor": "May cause respiratory illness to the people on prolonged "
                 "exposure. Effect may be more pronounced in people with lung and "
                 "heart diseases.",
    "Severe": "May cause respiratory impact even on healthy people, and serious "
              "health impacts on people with lung/heart disease. The health "
              "impacts may be experienced even during light physical activity.",
}
CPCB_SOURCE = ("Health guidance quoted from the Central Pollution Control Board, "
               "National Air Quality Index (2014), table of AQI categories and "
               "associated health impacts. CPCB calls the 101-200 band "
               "“Moderately Polluted”; shortened to Moderate here.")

# How long a live snapshot is reused before refetching. CPCB publishes roughly
# hourly, so a shorter window would just spend API calls on the same numbers.
LIVE_TTL = 900


def aqi_from_pm25(pm25):
    """CPCB AQI and category for a PM2.5 concentration in ug/m3.

    Linear interpolation inside the band that contains the concentration, which
    is the standard sub-index formula:

        AQI = aqi_lo + (aqi_hi - aqi_lo) * (C - conc_lo) / (conc_hi - conc_lo)

    Returns (aqi, category). Concentrations above the top band return 500.
    """
    if pm25 is None or not np.isfinite(pm25):
        return None, None
    c = max(float(pm25), 0.0)
    for lo, hi, alo, ahi, name in CPCB_PM25:
        if c <= hi:
            return int(round(alo + (ahi - alo) * (c - lo) / (hi - lo))), name
    return 500, CPCB_PM25[-1][4]


# Diverging blue <-> red about a neutral grey. Concentrations at one hour sit in
# a narrow band, so an absolute ramp renders the whole city as one shade;
# centring on the hour's own median spends the full ramp on real differences.
# The two arms are lightness-matched (max dL 0.04) so neither side shouts.
DIV_LOW = ["#0d366b", "#1c5cab", "#2a78d6", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
DIV_MID = "#f0efec"
DIV_HIGH = ["#fbd5d4", "#f4a9a8", "#ec7f7e", "#e34948", "#c73332", "#a02524", "#6b1717"]

SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_MAX = 100.0

MASK_GREY = [176, 174, 168]
MASK_ALPHA = 70

# Mask-reason layer. Validated as a categorical pair: CVD dE 9.2, normal-vision
# 27.6. Aqua is sub-3:1 on a light surface, so the legend labels both in text.
REASON_COLORS = {1: [27, 175, 122], 2: [235, 104, 52], 3: [140, 138, 132]}

# Plain-language mask reasons. Deliberately NOT "too far from a monitor":
# Phase 2 measured error against distance to the nearest station and found no
# relationship (r = -0.02). These cells are refused because the place is unlike
# anywhere we measure, which is a different claim from being far away.
REASON_SHORT = {1: "Open land, water or forest",
                2: "Away from the roads and industry our monitors sit near",
                3: "Unlike anywhere we have a monitor"}
REASON_LONG = {
    1: "Every monitor sits in built-up Mumbai. This cell has fewer roads and "
       "buildings around it than any of them - it is sea, forest or open "
       "ground - so there is nothing comparable to estimate from.",
    2: "This is inhabited ground, but it sits further from a main road or an "
       "industrial area than any of our monitors do. Monitors cluster near "
       "traffic and industry, so quieter neighbourhoods like this one have no "
       "close counterpart in the data.",
    3: "One of the land-use measurements here falls outside the range covered "
       "by all 39 monitors, so an estimate would be a guess.",
}

FEATHER = 0.10
GRID_DEFAULT = 25

# Session keys for the location feature. The browser's coordinates live only in
# this session's memory for as long as the tab is open; nothing is written to
# disk, logged, or put in the URL. They do reach the server, because the
# interpolation runs there -- the UI says so rather than implying otherwise.
GEO_KEY = "geo_position"
LOCATE_FLAG = "locate_requested"

st.set_page_config(page_title="Mumbai air quality", layout="wide")


def hex_to_rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


DIV_RGB = np.array([hex_to_rgb(h) for h in DIV_LOW + [DIV_MID] + DIV_HIGH], float)
SEQ_RGB = np.array([hex_to_rgb(h) for h in SEQ], float)


def sample(ramp, t):
    pos = np.clip(t, 0, 1) * (len(ramp) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(ramp) - 1)
    f = (pos - lo)[:, None]
    return (ramp[lo] * (1 - f) + ramp[hi] * f).round().astype(int)


def diverging_colors(values, centre, half):
    t = np.clip((np.nan_to_num(values, nan=centre) - centre) / half, -1, 1)
    return sample(DIV_RGB, (t + 1) / 2)


def absolute_colors(values):
    return sample(SEQ_RGB, np.nan_to_num(values, nan=0.0) / SEQ_MAX)


def feather_alpha(n):
    """1 in the interior, tapering to 0 over the outer FEATHER of each axis.

    A hard rectangular edge of colour reads as a rendering bug rather than data.
    """
    d = np.minimum(np.arange(n), n - 1 - np.arange(n)) / max(n - 1, 1)
    f = np.clip(d / FEATHER, 0, 1)
    return np.minimum.outer(f, f).ravel()


@st.cache_resource(show_spinner="Loading map data...")
def engine():
    """The model module, with its KD-trees and model card already warm.

    cache_resource, not cache_data: this holds live scipy KD-trees and a loaded
    module, none of which survive a pickle round trip.
    """
    spec = importlib.util.spec_from_file_location(
        "interpolate", ROOT / "model" / "interpolate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.production()
    # No OSM warm-up: the extrapolation masks are baked into
    # data/processed/grid_masks.npz by `python model/interpolate.py --masks`,
    # so the 41 MB of point clouds are a pipeline input, not a runtime one.
    if mod.grid_masks() is None:
        raise FileNotFoundError(
            f"{mod.GRID_MASK_PATH.name} is missing - run "
            f"'python model/interpolate.py --masks' to generate it.")
    return mod


@st.cache_data(show_spinner=False)
def panel():
    mod = engine()
    wide, feat, _ = mod.load()
    return wide, feat.reset_index()[["station_id", "site", "lat", "lon"]]


@st.cache_data(show_spinner=False)
def model_card():
    return json.loads(engine().CARD_PATH.read_text(encoding="utf-8"))


def datagov_key():
    """The live-feed credential: Streamlit secrets first, then .env.

    st.secrets raises rather than returning empty when no secrets file exists,
    which is the normal case locally, so the lookup is guarded and simply falls
    through to the environment. Only this one key is needed in the cloud --
    OPENAQ_API_KEY belongs to the offline pipeline, not the app.
    """
    try:
        value = st.secrets.get("DATAGOV_API_KEY")
    except Exception:
        value = None
    return str(value) if value else None


@st.cache_data(ttl=LIVE_TTL, show_spinner="Fetching current readings...")
def live_snapshot():
    """Current CPCB readings, refetched at most every LIVE_TTL seconds."""
    spec = importlib.util.spec_from_file_location("live", ROOT / "data" / "live.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _, st_df = panel()
    # A plain dict, not the dataclass: st.cache_data pickles what it stores, and
    # a class defined in a module loaded by path is not importable by name.
    return dataclasses.asdict(mod.fetch_live(st_df, key=datagov_key()))


def _unpack(s):
    return s.lats, s.lons, s.values, s.sigma, s.mask, s.mask_kind, s.meta


@st.cache_data(show_spinner=False)
def surface(ts_iso, resolution):
    return _unpack(engine().predict_surface(pd.Timestamp(ts_iso),
                                            grid_resolution=resolution))


@st.cache_data(show_spinner=False)
def surface_from(items, resolution, ts_iso):
    """Surface built from an explicit set of readings (the live path).

    `items` is a tuple of (station_id, value) pairs rather than a Series so the
    cache can hash it.
    """
    return _unpack(engine().predict_surface(
        pd.Timestamp(ts_iso), grid_resolution=resolution,
        readings=pd.Series(dict(items), dtype=float)))


def cell_polygons(lats, lons):
    dlat = (lats[1] - lats[0]) / 2 if len(lats) > 1 else 0.01
    dlon = (lons[1] - lons[0]) / 2 if len(lons) > 1 else 0.01
    glon, glat = np.meshgrid(lons, lats)
    return [[[x - dlon, y - dlat], [x + dlon, y - dlat],
             [x + dlon, y + dlat], [x - dlon, y + dlat]]
            for x, y in zip(glon.ravel(), glat.ravel())]


def contributions(lat, lon, readings, stations, power, top_n=5):
    """Which monitors actually produced this cell's number, and how much.

    These are the IDW weights themselves, renormalised to percentages - the
    arithmetic that made the estimate, not an interpretation of it. Nothing here
    says why the air is dirty; it says where the number came from.
    """
    mod = engine()
    live = stations[stations["station_id"].isin(readings.index)].copy()
    d = mod.haversine_grid_km(np.array([lat]), np.array([lon]),
                              live["lat"].to_numpy(), live["lon"].to_numpy())[0]
    w = 1.0 / np.maximum(d, mod.MIN_DIST_KM) ** power
    live["km"] = d
    live["weight"] = 100 * w / w.sum()
    live["reading"] = live["station_id"].map(readings).astype(float)
    ranked = live.sort_values("weight", ascending=False)
    top = ranked.head(top_n)
    # How much of the answer the shown rows actually accounts for. At power 0.5
    # the weighting is deliberately flat, so the top five carry only ~25-35%
    # and no small group of monitors "makes" the number. Saying so is the
    # difference between provenance and a misleading highlight reel.
    return top, float(top["weight"].sum()), len(ranked)


def cell_index(lat, lon, lats, lons):
    """Flat index of the grid cell containing a point, matching the ravel order."""
    i = int(np.argmin(np.abs(np.asarray(lats) - lat)))
    j = int(np.argmin(np.abs(np.asarray(lons) - lon)))
    return i * len(lons) + j


def km_outside(lat, lon, bbox):
    """How far a point sits outside the mapped rectangle, in km. 0 when inside."""
    lon_min, lat_min, lon_max, lat_max = bbox
    dlat = max(lat_min - lat, 0.0, lat - lat_max)
    dlon = max(lon_min - lon, 0.0, lon - lon_max)
    return float(np.hypot(dlat * 111.0,
                          dlon * 111.0 * np.cos(np.radians(lat))))


def swatch(color, label):
    c = f"rgb({color[0]},{color[1]},{color[2]})"
    return (f'<span style="display:inline-block;width:14px;height:14px;'
            f'background:{c};vertical-align:-2px;border-radius:2px"></span> {label}')


# ---------------------------------------------------------------- load
t_start = time.perf_counter()

try:
    wide, stations = panel()
    card = model_card()
except (SystemExit, FileNotFoundError) as e:
    st.error(str(e))
    st.stop()

local_index = wide.index.tz_convert(TZ)
first, last = local_index.min(), local_index.max()

# Readings are stamped on the UTC hour and IST is UTC+5:30, so every reading
# lands at half past in local time. The slider picks a whole IST hour and we
# floor to the UTC hour containing it, which is what predict_surface does
# internally -- doing it here too keeps the monitor dots on the same hour as the
# surface. The caption prints the timestamp actually drawn, half past and all.
live_utc = wide.index[wide.notna().any(axis=1).to_numpy()]
latest = live_utc.max().tz_convert(TZ) + pd.Timedelta(minutes=30)

st.sidebar.title("Mumbai air quality")
st.sidebar.caption(f"{len(stations)} monitors across the metro region")

history = st.sidebar.toggle(
    "Browse history", value=False,
    help="Off shows current conditions. On lets you pick any past date and hour.")

if history:
    date = st.sidebar.date_input("Date", value=latest.date(),
                                 min_value=first.date(), max_value=last.date())
    hour = st.sidebar.slider("Hour (IST)", 0, 23, int(latest.hour))
else:
    date, hour = latest.date(), int(latest.hour)

scale = st.sidebar.radio(
    "Colour scale", ["Compare across the city", "Absolute AQI scale"],
    help="Comparing centres the colours on this hour's citywide middle, so you "
         "can see which areas are better or worse than the rest. Absolute fixes "
         "the scale so colours mean the same thing at every hour.")
explain = st.sidebar.toggle("Show why grey areas are grey", value=False)
resolution = st.sidebar.select_slider("Map detail", [15, 25, 40, 60],
                                      value=GRID_DEFAULT)

st.sidebar.divider()
if st.session_state.get(LOCATE_FLAG):
    if st.sidebar.button("Clear my location", width="stretch"):
        # Drop the flag and the component's stored value; nothing else held it.
        st.session_state.pop(LOCATE_FLAG, None)
        st.session_state.pop(GEO_KEY, None)
        st.rerun()
elif st.sidebar.button("Use my location", width="stretch"):
    st.session_state[LOCATE_FLAG] = True
    st.rerun()

# ---------------------------------------------------------------- location
# Only asked for when the button is pressed, so the browser never prompts for
# permission on page load. The component resolves rather than rejects on
# denial, so a refusal arrives as data and the app carries on unchanged.
geo = None
if st.session_state.get(LOCATE_FLAG):
    from streamlit_js_eval import get_geolocation

    pos = get_geolocation(component_key=GEO_KEY)
    if pos is None:
        geo = {"state": "waiting"}
    elif isinstance(pos, dict) and pos.get("error"):
        err = pos["error"]
        geo = {"state": "denied" if err.get("code") == 1 else "unavailable",
               "message": str(err.get("message") or "")}
    elif isinstance(pos, dict) and pos.get("coords"):
        c = pos["coords"]
        geo = {"state": "ok", "lat": float(c["latitude"]),
               "lon": float(c["longitude"]),
               "accuracy_m": float(c.get("accuracy") or 0)}
    else:
        geo = {"state": "unavailable", "message": "no position returned"}

# ---------------------------------------------------------------- source
# Live by default; history only when asked for. When the live feed cannot be
# reached the app does not go blank -- it drops to the newest hour we hold and
# says so, because a dashboard that renders nothing is worse than one that is
# honest about being behind.
snap, live_ok, fallback_reason = None, False, None
if not history:
    snap = live_snapshot()
    live_ok = snap["ok"]
    if not live_ok:
        fallback_reason = snap["error"]

t_surface = time.perf_counter()
if live_ok:
    ts = snap["observed_at"] or pd.Timestamp.now(tz="UTC").floor("h")
    lats, lons, values, sigma, mask, kind, meta = surface_from(
        tuple(sorted(snap["readings"].items())), resolution, ts.isoformat())
else:
    ts = pd.Timestamp(datetime.combine(date, dtime(hour)),
                      tz=TZ).tz_convert("UTC").floor("h")
    lats, lons, values, sigma, mask, kind, meta = surface(ts.isoformat(), resolution)
t_surface = time.perf_counter() - t_surface

if meta["n_stations"] == 0:
    same_day = local_index[(local_index.date == date)
                           & wide.notna().any(axis=1).to_numpy()]
    hint = (f"Hours with readings on {date:%d %b}: "
            + ", ".join(f"{h:02d}" for h in sorted({t.hour for t in same_day}))
            if len(same_day) else f"No monitor reported at all on {date:%d %b}.")
    st.warning(f"No monitor reported for {date:%d %b %Y} {hour:02d}:00. {hint}")
    st.stop()

flat_v, flat_s = values.ravel(), sigma.ravel()
flat_m, flat_k = mask.ravel(), kind.ravel()
readings = (snap["readings"].dropna() if live_ok
            else (wide.loc[ts].dropna() if ts in wide.index
                  else pd.Series(dtype=float)))

# Centre on the median of the cells actually drawn, not on the median of the
# reporting monitors. IDW at power 0.5 is a broad weighted average, so the
# surface tracks the monitor *mean*, and PM2.5 is right-skewed enough that the
# monitor median can leave 98% of the map on one side of the ramp. The surface
# median splits the map ~50/50, which is what makes the two arms readable.
inrange = flat_v[~flat_m]
centre = float(np.median(inrange)) if inrange.size else float(readings.median())
half = max(float(np.percentile(np.abs(inrange - centre), 95)) if inrange.size else 0.0, 2.0)
station_median = float(readings.median())
centre_aqi, centre_cat = aqi_from_pm25(centre)

relative = scale.startswith("Compare")
colors = diverging_colors(flat_v, centre, half) if relative else absolute_colors(flat_v)
alpha = (175 * feather_alpha(len(lats))).round().astype(int)
fill = np.column_stack([colors, alpha])

if explain:
    fill[:, :3] = 232
    fill[:, 3] = (alpha * 0.45).astype(int)
    for k, rgb in REASON_COLORS.items():
        hit = flat_k == k
        fill[hit, :3] = rgb
        fill[hit, 3] = (alpha[hit] * 0.85).astype(int)
else:
    fill[flat_m, :3] = MASK_GREY
    fill[flat_m, 3] = (alpha[flat_m] * (MASK_ALPHA / 175)).astype(int)

# deck.gl escapes values interpolated into a tooltip template, so any HTML put
# in the data comes back out as literal tags. Structure lives in the template
# below; these fields are plain text.
glon, glat = np.meshgrid(lons, lats)
rows = []
for v, sd, m, k in zip(flat_v, flat_s, flat_m, flat_k):
    aqi, cat = aqi_from_pm25(v)
    if m:
        rows.append(("No estimate here", REASON_SHORT[int(k)], "",
                     "Click the square to see why"))
    else:
        rows.append((f"AQI {aqi} - {cat}", f"{v:.0f} ug/m3 PM2.5",
                     f"could be {sd:.0f} higher or lower",
                     "Click the square for where this comes from"))

cells = pd.DataFrame({
    "polygon": cell_polygons(lats, lons),
    "fill": [list(map(int, c)) for c in fill],
    "idx": np.arange(len(flat_v)),
    "lat": glat.ravel(), "lon": glon.ravel(),
    "title": [r[0] for r in rows], "line2": [r[1] for r in rows],
    "line3": [r[2] for r in rows], "line4": [r[3] for r in rows],
})

# Resolve the position to a grid cell now that the grid exists. Anything that
# is not a usable in-bbox point becomes a status the panel explains in words.
located = None
if geo is not None:
    if geo["state"] == "ok":
        bbox = engine().grid_masks()["bbox"]
        out_km = km_outside(geo["lat"], geo["lon"], bbox)
        if out_km > 0:
            located = {"state": "outside", "km": out_km,
                       "lat": geo["lat"], "lon": geo["lon"]}
        else:
            located = {"state": "inside",
                       "idx": cell_index(geo["lat"], geo["lon"], lats, lons),
                       "lat": geo["lat"], "lon": geo["lon"],
                       "accuracy_m": geo["accuracy_m"]}
    else:
        located = dict(geo)

# ---------------------------------------------------------------- monitors
pts = stations.copy()
pts["value"] = pts["station_id"].map(readings).astype(float)
live = pts["value"].notna()
vals = pts["value"].to_numpy()
pts["fill"] = [list(map(int, c)) + [255] for c in
               (diverging_colors(vals, centre, half) if relative else absolute_colors(vals))]
pts.loc[~live, "fill"] = pd.Series([[140, 138, 132, 90]] * int((~live).sum()),
                                   index=pts.index[~live])
p_title, p2 = [], []
for v in vals:
    if np.isfinite(v):
        aqi, cat = aqi_from_pm25(v)
        p_title.append(f"AQI {aqi} - {cat}")
        p2.append(f"{v:.0f} ug/m3 PM2.5 - measured here")
    else:
        p_title.append("Monitor offline")
        p2.append("no reading this hour")
pts["title"], pts["line2"] = p_title, p2
pts["line3"] = pts["site"]
pts["line4"] = "Government monitoring station"

# ---------------------------------------------------------------- status
if live_ok:
    age_min = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60
    st.success(
        f"**Live** - {snap["n_reporting"]} of {snap["n_total"]} monitors reporting, "
        f"updated {ts.tz_convert(TZ):%H:%M} "
        f"({'just now' if age_min < 5 else f'{age_min:.0f} min ago'}). "
        f"Source: {snap["source"]}.")
elif history:
    st.info(f"**Browsing history** - showing "
            f"{ts.tz_convert(TZ):%d %b %Y, %H:%M}. Turn off *Browse history* in "
            f"the sidebar to return to current conditions.")
else:
    st.warning(
        f"**Live readings unavailable.** {fallback_reason}. Showing the most "
        f"recent readings we hold instead, from "
        f"{ts.tz_convert(TZ):%d %b %Y, %H:%M}."
        + (f"\n\n{snap["hint"]}" if snap and snap.get("hint") else ""))

# ---------------------------------------------------------------- explainer
_when = ("right now" if live_ok
         else f"{ts.tz_convert(TZ):%d %b %Y, %H:%M}")
st.markdown(f"#### Air quality across Mumbai, {_when}")
if relative:
    # Plain markdown, no inline HTML. A <span> in the middle of a paragraph is
    # inline HTML, which the markdown renderer escapes and prints as literal
    # tags; the block-level <div> in the legend below is treated as an HTML
    # block and does render. The colour bar under the map carries the mapping
    # anyway, so the words alone are enough here.
    st.markdown(
        f"Colours compare each area with the citywide middle for this hour "
        f"(**AQI {centre_aqi}, {centre_cat}**). **Blue is cleaner** than the "
        f"rest of the city, **red is dirtier**. Grey areas are ones we cannot "
        f"put a number on, because they look nothing like anywhere we have a "
        f"monitor.")
else:
    st.markdown(
        "Darker blue means more PM2.5 on a fixed scale, so a colour means the "
        "same thing at every hour. Grey areas are ones we cannot put a number "
        "on, because they look nothing like anywhere we have a monitor.")

# ---------------------------------------------------------------- map
map_layers = []
if located is not None and located["state"] == "inside":
    here = pd.DataFrame([{"lat": located["lat"], "lon": located["lon"],
                          "title": "Your location", "line2": "",
                          "line3": "", "line4": ""}])
    map_layers.append(
        pdk.Layer("ScatterplotLayer", here, id="here",
                  get_position=["lon", "lat"], get_fill_color=[255, 255, 255],
                  get_line_color=[11, 11, 11], line_width_min_pixels=3,
                  stroked=True, radius_min_pixels=8, radius_max_pixels=11,
                  get_radius=250, pickable=False))

deck = pdk.Deck(
    layers=[
        pdk.Layer("PolygonLayer", cells, id="cells", get_polygon="polygon",
                  get_fill_color="fill", stroked=False, pickable=True,
                  auto_highlight=True),
        # White ring so a monitor stays legible against the darkest ramp steps.
        pdk.Layer("ScatterplotLayer", pts, id="monitors", get_position=["lon", "lat"],
                  get_fill_color="fill", get_line_color=[255, 255, 255],
                  line_width_min_pixels=2, stroked=True, radius_min_pixels=6,
                  radius_max_pixels=9, get_radius=200, pickable=True,
                  auto_highlight=True),
    ] + map_layers,
    initial_view_state=pdk.ViewState(latitude=19.12, longitude=72.95, zoom=9.2),
    map_style="light",
    tooltip={"html": "<div style='font-weight:600;margin-bottom:2px'>{title}</div>"
                     "<div>{line2}</div><div>{line3}</div>"
                     "<div style='opacity:.7;margin-top:3px'>{line4}</div>",
             "style": {"backgroundColor": "#0b0b0b", "color": "#fff",
                       "fontSize": "12px", "padding": "8px", "borderRadius": "4px"}},
)
event = st.pydeck_chart(deck, height=560, on_select="rerun",
                        selection_mode="single-object", key="map")

# ---------------------------------------------------------------- legend
if explain:
    counts, total = meta["mask_reasons"], values.size
    parts = [swatch(REASON_COLORS[k], f"{REASON_SHORT[k]} ({100 * counts[n] / total:.0f}%)")
             for k, n in ((1, "_empty"), (2, "_remote"), (3, "_other")) if counts[n]]
    st.markdown(" &nbsp;&nbsp; ".join(parts)
                + "<br/><span style='color:#52514e'>Pale squares are ones we can "
                  "estimate. The orange group is the one worth knowing about: "
                  "ordinary neighbourhoods that simply have no monitor like "
                  "them.</span>", unsafe_allow_html=True)
elif relative:
    bar = "".join(f'<span style="display:inline-block;width:19px;height:12px;'
                  f'background:{c}"></span>' for c in DIV_LOW + [DIV_MID] + DIV_HIGH)
    lo_a, _ = aqi_from_pm25(max(centre - half, 0))
    hi_a, _ = aqi_from_pm25(centre + half)
    st.markdown(
        f'<div style="font-size:13px">{bar}<br/><span style="color:#52514e">'
        f'AQI {lo_a} &nbsp;&larr; cleaner &nbsp;&nbsp;'
        f'<b>citywide middle: AQI {centre_aqi}</b>'
        f'&nbsp;&nbsp; dirtier &rarr;&nbsp; AQI {hi_a}</span> &nbsp;&nbsp;'
        + swatch(MASK_GREY, "No estimate") + '</div>', unsafe_allow_html=True)
else:
    bar = "".join(f'<span style="display:inline-block;width:20px;height:12px;'
                  f'background:{c}"></span>' for c in SEQ)
    st.markdown(f'<div style="font-size:13px">{bar} &nbsp;cleaner &rarr; dirtier'
                f' &nbsp;&nbsp;{swatch(MASK_GREY, "No estimate")}</div>',
                unsafe_allow_html=True)

# ---------------------------------------------------------------- why panel
st.divider()
# Streamlit hands back an attribute-dict; accept plain-dict access too so the
# panel does not depend on which of the two a given version returns.
picked = None
sel = getattr(event, "selection", None)
if sel is None and isinstance(event, dict):
    sel = event.get("selection")
if sel:
    objs = (sel.get("objects") if hasattr(sel, "get") else None) or {}
    for layer in ("cells", "monitors"):
        if objs.get(layer):
            picked = (layer, objs[layer][0])
            break

def cell_estimate(i, cell_lat, cell_lon):
    """The estimate, health guidance and provenance for one grid cell.

    Shared by the map-click path and the my-location path so both give the same
    answer in the same words.
    """
    v, sd = flat_v[i], flat_s[i]
    masked, k = bool(flat_m[i]), int(flat_k[i])
    aqi, cat = aqi_from_pm25(v)

    if masked:
        st.warning(f"**No estimate here - {REASON_SHORT[k].lower()}.**\n\n"
                   f"{REASON_LONG[k]}")
        # A refusal on its own is unhelpful when a real monitor is close by, so
        # offer the nearest actual measurement. It is a reading, not an estimate
        # for this square, and is labelled as such.
        near, _, _ = contributions(cell_lat, cell_lon, readings, stations,
                                   card["idw_power"], top_n=1)
        if len(near):
            r = near.iloc[0]
            n_aqi, n_cat = aqi_from_pm25(r["reading"])
            st.markdown(
                f"The nearest monitor is **{r['site']}**, {r['km']:.1f} km away, "
                f"currently reading **AQI {n_aqi} - {n_cat}** "
                f"({r['reading']:.0f} ug/m3). That is a measurement at the "
                f"monitor, not an estimate for this square.")
            st.caption(CPCB_SOURCE)
        return

    diff = v - centre
    if abs(diff) < 1:
        comparison = "about the same as the citywide middle this hour"
    else:
        comparison = (f"{abs(diff):.0f} ug/m3 "
                      f"{'dirtier' if diff > 0 else 'cleaner'} than the "
                      f"citywide middle this hour")
    st.markdown(
        f"**AQI {aqi} - {cat}** &nbsp;·&nbsp; {v:.0f} ug/m3 PM2.5, "
        f"could be {sd:.0f} higher or lower.\n\n"
        f"That is {comparison} (AQI {centre_aqi}, {centre_cat}).\n\n"
        f"**Health guidance ({cat}):** {CPCB_HEALTH[cat]}")
    st.caption(CPCB_SOURCE)

    con, share, n_all = contributions(cell_lat, cell_lon, readings, stations,
                                      card["idw_power"])
    st.markdown("**Which monitors this number came from**")
    st.dataframe(pd.DataFrame({
        "Monitor": con["site"],
        "Distance": con["km"].map("{:.1f} km".format),
        "Its reading": [f"{r:.0f} ug/m3 (AQI {aqi_from_pm25(r)[0]})"
                        for r in con["reading"]],
        "Share of this estimate": con["weight"].map("{:.1f}%".format),
    }), hide_index=True, width="stretch")
    st.markdown(
        f"These {len(con)} monitors together account for **{share:.0f}%** of "
        f"the estimate. The other {n_all - len(con)} reporting monitors make "
        f"up the remaining {100 - share:.0f}% in smaller shares - the "
        f"calculation spreads weight deliberately widely, which is why this "
        f"map stays close to a citywide average.")
    st.caption(
        "These shares are the actual weights used in the calculation. They "
        "show where the number came from, not what caused the pollution: "
        "our own testing found road, building and industry data do not "
        "explain why one area differs from another.")


# The located cell takes the panel when there is one; a map click still works
# and is what the panel falls back to.
if located is not None:
    st.markdown("#### Air quality where you are")
    state = located["state"]
    if state == "waiting":
        st.info("Waiting for your browser to share a location. If nothing "
                "happens, your browser may be blocking it - the map and "
                "everything else still work.")
    elif state == "denied":
        st.info("Location permission was declined, so nothing has changed. "
                "Click any square on the map to get the same estimate for a "
                "place of your choosing.")
    elif state == "unavailable":
        st.info(f"Your browser could not provide a location"
                f"{' (' + located['message'] + ')' if located.get('message') else ''}. "
                f"This happens on desktops without GPS or on an insecure "
                f"connection. Click any square on the map instead.")
    elif state == "outside":
        st.warning(
            f"**You are about {located['km']:.0f} km outside the area this map "
            f"covers.** It spans the Mumbai metropolitan region only, so there "
            f"is no estimate for where you are - a number here would be made "
            f"up rather than interpolated.")
    else:
        acc = located.get("accuracy_m") or 0
        cell_lat = float(lats[located["idx"] // len(lons)])
        cell_lon = float(lons[located["idx"] % len(lons)])
        cell_estimate(located["idx"], cell_lat, cell_lon)
        st.caption(
            f"Matched to the {meta['cell_km']:.1f} km square containing your "
            f"position"
            + (f", located to about {acc:.0f} m" if acc else "") + ". Your "
            f"coordinates are used to pick that square and nothing else: they "
            f"are not saved, logged, or put in the page address, and they are "
            f"gone when you close the tab. They do reach this app's server, "
            f"because the estimate is computed there.")
    st.divider()

if picked is None:
    if located is None:
        st.markdown("#### Why this number?")
        st.markdown("Click any square or monitor on the map to see exactly "
                    "which monitors produced its estimate.")
elif picked[0] == "monitors":
    obj = picked[1]
    st.markdown(f"#### {obj.get('line3', 'This monitor')}")
    v = obj.get("value")
    if v is None or not np.isfinite(float(v)):
        st.info("This monitor did not report a reading for this hour.")
    else:
        aqi, cat = aqi_from_pm25(float(v))
        st.markdown(f"**AQI {aqi} - {cat}** &nbsp;·&nbsp; {float(v):.0f} ug/m3 PM2.5")
        st.markdown(f"This is a **measured** reading, not an estimate. "
                    f"**Health guidance ({cat}):** {CPCB_HEALTH[cat]}")
        st.caption(CPCB_SOURCE)
else:
    obj = picked[1]
    st.markdown("#### Why this number?")
    cell_estimate(int(obj["idx"]), float(obj["lat"]), float(obj["lon"]))

# ---------------------------------------------------------------- model card
st.divider()
st.subheader("How this map is made")

cv = card["cv"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Method", "Distance weighting")
c2.metric("Typical error", f"{cv['rmse']:.0f} ug/m3",
          help="Root-mean-square error when a whole monitor is hidden from the "
               "model and it has to predict that location")
c3.metric("Average miss", f"{cv['mae']:.0f} ug/m3", help="Mean absolute error")
c4.metric("Tested on", f"{cv['n_stations']} monitors",
          help=f"{cv['n_folds']} folds, {cv['n_station_hours']:,} station-hours")

left, right = st.columns([1, 1.35])
comp = (pd.DataFrame(card["comparison"]).T
          .rename_axis("model").reset_index().sort_values("RMSE"))
base = comp.loc[comp["model"] == "idw", "RMSE"].iloc[0]
comp["vs IDW"] = ((comp["RMSE"] - base) / base * 100).map("{:+.2f}%".format)
left.dataframe(comp, hide_index=True, width="stretch")

right.markdown(f"""
**No method beat simply averaging every monitor by more than 0.5%.** Averaging
scores {card['comparison']['regional_mean']['RMSE']:.2f}; the winner, distance
weighting, scores {base:.2f}. That gap is the entire value this map adds over
quoting one number for the whole city.

Why: once each hour's citywide level is taken out, air quality stops being
predictable from distance beyond about 2 km - and Mumbai's monitors sit 1-40 km
apart. Kriging and a machine-learning model using road, building and industry
data both did worse than plain distance weighting.

**So read the map as the city average, tilted slightly** - not as a precise
street-level picture.
""")

with st.expander("Technical details"):
    total_ms = 1000 * (time.perf_counter() - t_start)
    r = meta["mask_reasons"]
    st.markdown(f"""
| | |
|---|---|
| Source | {"data.gov.in (CPCB), live" if live_ok else ("cached history (browsing)" if history else "cached history (live feed unavailable)")} |
| Timestamp drawn | {ts.tz_convert(TZ):%Y-%m-%d %H:%M} IST ({ts:%H:%M} UTC) |
| Monitors reporting | {meta['n_stations']} of {len(stations)} |
| Live fetch | {"ok" if live_ok else (fallback_reason or "not attempted (browsing history)")} |
| Grid | {resolution} x {resolution}, {meta['cell_km']:.1f} km cells |
| Masked cells | {100 * meta['masked_fraction']:.1f}% ({r['_empty']} empty, {r['_remote']} remote, {r['_other']} other) |
| Colour centre | map median {centre:.1f} ug/m3, +/- {half:.1f} spread |
| Monitor median | {station_median:.1f} ug/m3 |
| Model | IDW, power {card['idw_power']:g}, extrapolation band {meta['band'][0]}-{meta['band'][1]} |
| Uncertainty | sigma = max({card['uncertainty']['floor']:.2f}, {card['uncertainty']['a']:.2f} + {card['uncertainty']['b']:.3f} x prediction) |
| Render | surface {1000 * t_surface:.0f} ms, page {total_ms:.0f} ms |

**On the AQI figures.** CPCB defines the National AQI on a 24-hour average
concentration. This dashboard is hourly, so the numbers here apply the CPCB
PM2.5 breakpoints to a single hour's value. The scale and category names are
CPCB's; the figure is not the official daily AQI and moves around more than a
daily number does.

**On the error bar.** Uncertainty is calibrated against held-out monitors and
scales with the concentration, not with distance from a monitor - error against
distance-to-nearest-monitor measured r = -0.02. Being far from a monitor is not,
by itself, a reason to distrust a square.
""")
