"""Current CPCB readings from data.gov.in, for the app's live view.

The historical pipeline (data/fetch.py) pulls from OpenAQ, which republishes
CPCB data on a lag - measured at ~63 hours behind for the Mumbai stations, so it
cannot serve a "right now" view. data.gov.in publishes the same CPCB network in
near real time, which is what this module reads.

fetch_live() never raises. The app has to render something whatever the network
or the upstream API is doing, so every failure comes back as a LiveResult with
ok=False and a message a non-technical reader can act on.
"""

import os
from dataclasses import dataclass, field
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

# The feed names stations differently from OpenAQ ("Bandra Kurla Complex,
# Mumbai - MPCB" vs "Bandra Kurla Complex"), and Phase 1 already found station
# naming inconsistent across sources. Coordinates are the reliable join, so a
# live station is matched to ours when it lands within this distance.
MATCH_KM = 2.0

# Older than this and it is not "now" any more, whatever the feed says.
STALE_HOURS = 6


@dataclass
class LiveResult:
    ok: bool
    readings: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    observed_at: pd.Timestamp = None
    n_reporting: int = 0
    n_total: int = 0
    unmatched: int = 0
    source: str = "data.gov.in (CPCB)"
    error: str = None
    hint: str = None


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


def api_key():
    load_dotenv()
    return os.getenv("DATAGOV_API_KEY")


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


def _to_float(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # The feed uses "NA" and occasionally negative sentinels for a dead sensor.
    return f if f >= 0 else None


def parse(records, stations):
    """Map the feed's PM2.5 records onto our station ids by coordinates.

    Field names have shifted across versions of this dataset (pollutant_avg was
    once pollutant_value), so each is looked up through a list of aliases rather
    than assumed.
    """
    lat_c, lon_c = stations["lat"].to_numpy(), stations["lon"].to_numpy()
    ids = stations["station_id"].to_numpy()

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

        d = [haversine_km(lat, lon, a, o) for a, o in zip(lat_c, lon_c)]
        nearest = int(min(range(len(d)), key=d.__getitem__))
        if d[nearest] > MATCH_KM:
            unmatched += 1
            continue
        # Two feed rows can land on one of our stations; keep the first.
        values.setdefault(int(ids[nearest]), val)

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


def fetch_live(stations, key=None):
    """Current PM2.5 for our stations. Never raises; failures come back as ok=False.

    `key` lets the caller supply the credential -- the deployed app reads it
    from Streamlit's secrets manager, where there is no .env to read. Falling
    back to the environment keeps local development working unchanged.
    """
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
    if readings.empty:
        return LiveResult(
            ok=False, n_total=len(stations), unmatched=unmatched,
            error="data.gov.in answered but reported no PM2.5 for any of our "
                  "monitors right now")

    age_h = None
    if observed is not None:
        age_h = (pd.Timestamp.now(tz="UTC") - observed).total_seconds() / 3600
    if age_h is not None and age_h > STALE_HOURS:
        return LiveResult(
            ok=False, readings=readings, observed_at=observed,
            n_reporting=len(readings), n_total=len(stations), unmatched=unmatched,
            error=f"The live feed's newest reading is {age_h:.0f} hours old, "
                  f"which is not current")

    return LiveResult(ok=True, readings=readings, observed_at=observed,
                      n_reporting=len(readings), n_total=len(stations),
                      unmatched=unmatched)
