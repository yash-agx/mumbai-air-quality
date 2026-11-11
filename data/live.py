"""Current CPCB readings for the app's live view, from data.gov.in or WAQI.

The historical pipeline (data/fetch.py) pulls from OpenAQ, which republishes
CPCB data on a lag - measured at ~63 hours behind for the Mumbai stations, so it
cannot serve a "latest published" view. data.gov.in publishes the same CPCB
network in near real time, which is what this module reads first.

What it publishes is a rolling 24-hour average, not an hourly reading: CPCB's
AQI is defined on the preceding 24 hours (at least 16 of them), and each record
carries min_value and max_value across that same window. Verified against the
feed on 2026-11-11, where all 25 Mumbai PM2.5 records had min < max with a
median spread of 252 ug/m3, which no single hour can hold. The app says so
rather than calling these values current-hour readings.

data.gov.in blocks requests from Streamlit Cloud's servers, so the deployed app
cannot use it even with a valid key, while it works fine from India. WAQI is
therefore tried as a fallback: it republishes the same CPCB stations, but as US
AQI rather than ug/m3. Each matched station's PM2.5 sub-index (iaqi.pm25) is
read from its own feed and converted back to a concentration (see aqi_to_ugm3),
and the source string says which feed the numbers came from.

fetch_live() never raises. The app has to render something whatever the network
or the upstream API is doing, so every failure comes back as a LiveResult with
ok=False and a message a non-technical reader can act on.
"""

import os
import time
from dataclasses import dataclass, field, replace
from math import asin, cos, radians, sin, sqrt

import pandas as pd
import requests
from dotenv import load_dotenv

# "Real time Air Quality Index from various locations", published by CPCB.
RESOURCE = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
BASE = "https://api.data.gov.in/resource"
USER_AGENT = "mumbai-air-quality-dashboard/0.1"
TIMEOUT = 20
PAGE = 1000

# The feed names stations differently from OpenAQ, so coordinates are the
# reliable join: a live station matches ours within this distance.
MATCH_KM = 2.0

# Older than this and it is not "now" any more, whatever the feed says.
STALE_HOURS = 6

# --------------------------------------------------------------------------
# WAQI FALLBACK
#
# Two steps. map/bounds lists every station WAQI holds inside a lat/lon box,
# which gives each one's uid and coordinates in a single request; feed/@uid
# then gives that station's per-pollutant sub-indices. The second step is what
# we are after: map/bounds reports only the overall AQI, which is whichever
# pollutant dominates there, while feed/@uid carries iaqi.pm25 specifically.
#
# That is one request per matched station, so up to 40 in total. They sit
# behind the app's live-snapshot cache (LIVE_TTL, 15 minutes), so a page view
# does not pay for them; WAQI_BUDGET_S stops the fan-out if the API is slow
# enough that waiting would cost more than the readings are worth.
#
# Same bbox as data/fetch.py and scripts/02_features.py, in WAQI's order:
# south, west, north, east.
# --------------------------------------------------------------------------
WAQI_BASE = "https://api.waqi.info"
WAQI_BBOX = (18.85, 72.70, 19.40, 73.20)
WAQI_FEED_TIMEOUT = 8
WAQI_BUDGET_S = 25

# WAQI's terms require attribution wherever its data is shown.
WAQI_ATTRIBUTION = ("Air quality data from the World Air Quality Index project "
                    "(https://waqi.info/), CC BY-SA 4.0")

SOURCE_DATAGOV = "data.gov.in (CPCB)"
SOURCE_WAQI = "WAQI (CPCB stations, converted from AQI)"

# --------------------------------------------------------------------------
# US EPA PM2.5 AQI BREAKPOINTS, as (AQI_lo, AQI_hi, C_lo, C_hi) in ug/m3.
#
# iaqi.pm25 is an index, not a concentration, so it has to be run back through
# this table. Checked on 2026-11-11 across 17 Mumbai stations: a station's
# overall `aqi` equalled its iaqi.pm25 at 15 of them, which is what a
# sub-index does when PM2.5 dominates, and equalled the AQI computed FROM
# iaqi.pm25-read-as-ug/m3 at none of them.
#
# This is the pre-2024 table (AQI 50 at 12.0 ug/m3), the US EPA scale WAQI
# documents. A live comparison against data.gov.in could NOT separate it from
# EPA's 2024 revision: over the 13 stations reporting on both feeds, MAE came
# to 74.8 ug/m3 with this table and 75.5 with the 2024 one. The tables sit
# 0.75 ug/m3 apart while the feeds disagree by ~70, so the check has no power
# to choose between them and this stays on the documented scale.
#
# The size of that disagreement is the thing to know about the fallback:
# converted values ran a median 68 ug/m3 BELOW data.gov.in for the same
# station, and the feeds are not describing the same window. data.gov.in
# publishes CPCB's rolling 24-hour average; the WAQI stations were stamped
# with a single hour (22:00 IST). So the fallback shows where the city is
# worse, not numbers interchangeable with the primary source.
#
# A converted number is coarse either way: the index arrives as an integer, so
# each step back out lands in a 1-2 ug/m3 bin, wider in the higher bands.
# --------------------------------------------------------------------------
EPA_PM25_BREAKPOINTS = (
    (0, 50, 0.0, 12.0),
    (51, 100, 12.1, 35.4),
    (101, 150, 35.5, 55.4),
    (151, 200, 55.5, 150.4),
    (201, 300, 150.5, 250.4),
    (301, 400, 250.5, 350.4),
    (401, 500, 350.5, 500.4),
)


@dataclass
class LiveResult:
    ok: bool
    readings: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    observed_at: pd.Timestamp = None
    n_reporting: int = 0
    n_total: int = 0
    unmatched: int = 0
    source: str = SOURCE_DATAGOV
    error: str = None
    hint: str = None
    attribution: str = None


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


def api_key():
    load_dotenv()
    return os.getenv("DATAGOV_API_KEY")


def waqi_token():
    load_dotenv()
    return os.getenv("WAQI_TOKEN")


def _rows(key, city="Mumbai"):
    """Every PM2.5 record the feed holds for the city, following pagination."""
    out, offset = [], 0
    while True:
        r = requests.get(
            f"{BASE}/{RESOURCE}",
            params={"api-key": key, "format": "json", "limit": PAGE,
                    "offset": offset, "filters[city]": city},
            headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        if r.status_code == 403:
            raise PermissionError(
                "data.gov.in rejected the key for this dataset "
                "(HTTP 403, 'Key not authorised')")
        r.raise_for_status()
        batch = r.json().get("records", [])
        out.extend(batch)
        if len(batch) < PAGE:
            return out
        offset += PAGE


def _waqi_get(path, params, timeout=TIMEOUT):
    """One WAQI call, returning the `data` member of the response."""
    r = requests.get(f"{WAQI_BASE}{path}", params=params,
                     headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    # WAQI answers HTTP 200 with status "error" for a bad or unauthorised token.
    if payload.get("status") != "ok":
        raise PermissionError(
            f"WAQI rejected the request ({payload.get('data') or 'unknown error'})")
    return payload.get("data")


def _waqi_rows(token):
    """Every station WAQI holds inside the Mumbai bbox, in one request."""
    lat1, lon1, lat2, lon2 = WAQI_BBOX
    return _waqi_get("/map/bounds/",
                     {"token": token,
                      "latlng": f"{lat1},{lon1},{lat2},{lon2}"}) or []


def _waqi_feed(token, uid):
    """One station's feed, which is where the per-pollutant sub-indices are."""
    return _waqi_get(f"/feed/@{uid}/", {"token": token},
                     timeout=WAQI_FEED_TIMEOUT) or {}


def _station_feed(token):
    """A uid -> feed reader that answers None when one station fails.

    One station timing out should not take the whole fallback down. A total
    outage still surfaces, because the map/bounds call ahead of this raises.
    """
    def read(uid):
        try:
            return _waqi_feed(token, uid)
        except (requests.exceptions.RequestException, PermissionError,
                ValueError):
            return None
    return read


def _to_float(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # The feed uses "NA" and occasionally negative sentinels for a dead sensor.
    return f if f >= 0 else None


def aqi_to_ugm3(aqi):
    """US EPA PM2.5 AQI back to a concentration in ug/m3, or None if unusable.

    The AQI formula is piecewise linear within each band, so inverting it is
    the same interpolation run the other way. See EPA_PM25_BREAKPOINTS for
    which table this is and why the result carries extra uncertainty.
    """
    value = _to_float(aqi)
    if value is None:
        return None
    for lo_i, hi_i, lo_c, hi_c in EPA_PM25_BREAKPOINTS:
        if value <= hi_i:
            return round(lo_c + (value - lo_i) * (hi_c - lo_c) / (hi_i - lo_i), 1)
    # The scale stops at 500; anything above it is reported at the top of
    # the table rather than extrapolated past the end of the formula.
    return EPA_PM25_BREAKPOINTS[-1][3]


def nearest_station(lat, lon, stations):
    """Our station id nearest this point, or None if none is within MATCH_KM.

    Both feeds name stations their own way, so this coordinate match is the
    join for each of them.
    """
    lat_c, lon_c = stations["lat"].to_numpy(), stations["lon"].to_numpy()
    ids = stations["station_id"].to_numpy()
    d = [haversine_km(lat, lon, a, o) for a, o in zip(lat_c, lon_c)]
    nearest = int(min(range(len(d)), key=d.__getitem__))
    return int(ids[nearest]) if d[nearest] <= MATCH_KM else None


def parse(records, stations):
    """Map the feed's PM2.5 records onto our station ids by coordinates.

    Field names have shifted across versions of this dataset (the API's own
    field list currently calls them pollutant_min/max/avg with ids
    min_value/max_value/avg_value), so each is looked up through a list of
    aliases rather than assumed.

    The value taken is the average one, which is CPCB's rolling 24-hour mean
    for that station; min_value and max_value bound the same window and are
    not read.
    """
    values, stamps, unmatched = {}, [], 0
    for rec in records:
        pollutant = str(rec.get("pollutant_id") or rec.get("pollutant") or "")
        if pollutant.upper().replace(".", "").replace("_", "") not in ("PM25",):
            continue
        val = next((_to_float(rec.get(k)) for k in
                    ("pollutant_avg", "avg_value", "pollutant_value")
                    if rec.get(k) is not None), None)
        lat = _to_float(rec.get("latitude"))
        lon = _to_float(rec.get("longitude"))
        if val is None or lat is None or lon is None:
            continue

        sid = nearest_station(lat, lon, stations)
        if sid is None:
            unmatched += 1
            continue
        # Two feed rows can land on one of our stations; keep the first.
        values.setdefault(sid, val)

        stamp = rec.get("last_update") or rec.get("last_update_time")
        if stamp:
            stamps.append(stamp)

    observed = None
    if stamps:
        # The feed stamps as "01-01-2026 10:00:00" in IST.
        parsed = pd.to_datetime(pd.Series(stamps), format="%d-%m-%Y %H:%M:%S",
                                errors="coerce").dropna()
        if len(parsed):
            observed = (parsed.max().tz_localize("Asia/Kolkata").tz_convert("UTC"))

    return pd.Series(values, dtype=float), observed, unmatched


def waqi_pm25(feed):
    """The PM2.5 sub-index out of one station's feed, in ug/m3, or None.

    iaqi.pm25 is PM2.5 specifically. The `aqi` field on the same payload is
    whichever pollutant dominates there, which is usually but not always PM2.5,
    so it is not what this reads.
    """
    iaqi = (feed or {}).get("iaqi") or {}
    channel = iaqi.get("pm25") or {}
    return aqi_to_ugm3(channel.get("v"))


def match_waqi_stations(rows, stations):
    """[(our station id, WAQI uid)] for the bbox stations that are ours.

    One entry per station of ours, so the per-station fan-out that follows
    makes one request each and no more.
    """
    matched, unmatched, seen = [], 0, set()
    for rec in rows:
        lat = _to_float(rec.get("lat"))
        lon = _to_float(rec.get("lon"))
        uid = rec.get("uid")
        if lat is None or lon is None or uid is None:
            continue

        sid = nearest_station(lat, lon, stations)
        if sid is None:
            unmatched += 1
            continue
        # Two WAQI stations can land on one of ours; keep the first.
        if sid not in seen:
            seen.add(sid)
            matched.append((sid, uid))
    return matched, unmatched


def parse_waqi(rows, stations, feed, budget_s=WAQI_BUDGET_S):
    """Readings for our stations from WAQI, converted out of the US EPA AQI.

    `feed` is called with a WAQI uid and returns that station's feed payload,
    or None if it could not be read; stations it cannot answer for are skipped
    rather than failing the whole fallback. The fan-out stops once budget_s has
    gone, serving whatever came back by then.
    """
    matched, unmatched = match_waqi_stations(rows, stations)

    values, stamps = {}, []
    started = time.monotonic()
    for sid, uid in matched:
        if time.monotonic() - started > budget_s:
            break
        payload = feed(uid)
        val = waqi_pm25(payload)
        if val is None:
            # A station WAQI holds but with no PM2.5 channel this hour.
            continue
        values[sid] = val

        stamp = ((payload or {}).get("time") or {}).get("iso")
        if stamp:
            stamps.append(stamp)

    observed = None
    if stamps:
        # WAQI stamps ISO 8601 with an offset, e.g. "2026-11-11T14:00:00+05:30".
        parsed = pd.to_datetime(pd.Series(stamps), format="ISO8601", utc=True,
                                errors="coerce").dropna()
        if len(parsed):
            observed = parsed.max()

    return pd.Series(values, dtype=float), observed, unmatched


def _finish(readings, observed, unmatched, stations, source, name,
            attribution=None):
    """The checks both sources share: anything matched, and is it current?"""
    if readings.empty:
        return LiveResult(
            ok=False, n_total=len(stations), unmatched=unmatched, source=source,
            error=f"{name} answered but reported no PM2.5 for any of our "
                  f"monitors right now")

    age_h = None
    if observed is not None:
        age_h = (pd.Timestamp.now(tz="UTC") - observed).total_seconds() / 3600
    if age_h is not None and age_h > STALE_HOURS:
        return LiveResult(
            ok=False, readings=readings, observed_at=observed,
            n_reporting=len(readings), n_total=len(stations), unmatched=unmatched,
            source=source,
            error=f"The {name} feed's newest reading is {age_h:.0f} hours old, "
                  f"which is not current")

    return LiveResult(ok=True, readings=readings, observed_at=observed,
                      n_reporting=len(readings), n_total=len(stations),
                      unmatched=unmatched, source=source,
                      attribution=attribution)


def _fetch_datagov(stations, key=None):
    """The primary source: CPCB's near-real-time feed on data.gov.in."""
    key = key or api_key()
    if not key:
        return LiveResult(ok=False, n_total=len(stations),
                          error="No DATAGOV_API_KEY configured",
                          hint="Locally, add DATAGOV_API_KEY to .env in the "
                               "project root. On Streamlit Cloud, add it under "
                               "Settings > Secrets.")
    try:
        records = _rows(key)
    except PermissionError as e:
        return LiveResult(
            ok=False, n_total=len(stations), error=str(e),
            hint="The key is valid but not enabled for dataset access. On "
                 "data.gov.in, open My Account and generate or activate an API "
                 "key with resource access, then restart the app.")
    except requests.exceptions.RequestException as e:
        return LiveResult(ok=False, n_total=len(stations),
                          error=f"Could not reach data.gov.in ({type(e).__name__})",
                          hint="Check the network connection and try again.")
    except ValueError as e:
        return LiveResult(ok=False, n_total=len(stations),
                          error=f"data.gov.in returned something unreadable ({e})")

    readings, observed, unmatched = parse(records, stations)
    return _finish(readings, observed, unmatched, stations, SOURCE_DATAGOV,
                   "data.gov.in")


def _fetch_waqi(stations, token=None):
    """The fallback: same CPCB stations via WAQI, converted from US AQI."""
    token = token or waqi_token()
    if not token:
        return LiveResult(ok=False, n_total=len(stations), source=SOURCE_WAQI,
                          error="No WAQI_TOKEN configured",
                          hint="Request a free token at https://aqicn.org/data-"
                               "platform/token/ and add WAQI_TOKEN to .env "
                               "locally, or to Settings > Secrets on Streamlit "
                               "Cloud.")
    try:
        rows = _waqi_rows(token)
    except PermissionError as e:
        return LiveResult(ok=False, n_total=len(stations), source=SOURCE_WAQI,
                          error=str(e),
                          hint="Check that WAQI_TOKEN is the token from "
                               "aqicn.org and has not been revoked.")
    except requests.exceptions.RequestException as e:
        return LiveResult(ok=False, n_total=len(stations), source=SOURCE_WAQI,
                          error=f"Could not reach WAQI ({type(e).__name__})",
                          hint="Check the network connection and try again.")
    except ValueError as e:
        return LiveResult(ok=False, n_total=len(stations), source=SOURCE_WAQI,
                          error=f"WAQI returned something unreadable ({e})")

    readings, observed, unmatched = parse_waqi(rows, stations,
                                              _station_feed(token))
    return _finish(readings, observed, unmatched, stations, SOURCE_WAQI, "WAQI",
                   attribution=WAQI_ATTRIBUTION)


def fetch_live(stations, key=None, token=None):
    """Latest published PM2.5 for our stations, as a rolling 24-hour average.

    Never raises; failures come back as ok=False.

    Tries data.gov.in first and falls back to WAQI, whose values are converted
    from US AQI. The returned LiveResult says which source answered.

    `key` and `token` let the caller supply the credentials: the deployed app
    reads them from Streamlit's secrets manager, where there is no .env to
    read. Falling back to the environment keeps local development unchanged.
    """
    primary = _fetch_datagov(stations, key)
    if primary.ok:
        return primary

    fallback = _fetch_waqi(stations, token)
    if fallback.ok:
        return fallback

    # Both are down. Lead with the primary failure, which is the one the app
    # tells the reader to act on, and name the fallback's reason after it.
    # Whatever diagnostics the fallback did collect are kept.
    return replace(
        primary,
        error=f"{primary.error}. The WAQI fallback also failed: {fallback.error}",
        hint=primary.hint or fallback.hint,
        unmatched=primary.unmatched or fallback.unmatched)
