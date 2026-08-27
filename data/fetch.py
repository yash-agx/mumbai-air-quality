"""Fetch hourly PM2.5 for the Mumbai region from OpenAQ v3 into a raw parquet.

Run directly: python data/fetch.py [--refresh]

Three API quirks drive the shape of this module:

1.  A location can own several sensors for the same pollutant, with disjoint
    date ranges. The 2019-era CPCB sensors mostly died in late 2022 and were
    replaced by new sensor ids in Feb 2025, so location-level datetimeFirst
    spans a gap that no single sensor actually covers. We therefore select
    sensors, not locations, and only ones overlapping the target window.
2.  /sensors/{id}/days silently ignores datetime_from/datetime_to. Only
    /sensors/{id}/measurements/hourly honours them, so that is what we page.
3.  Wide windows on busy sensors return 408 server-side; the API's own advice
    is to ask for a shorter span, so fetch_sensor halves and retries.

Each sensor is cached to data/raw/_parts/ as it lands, so an interrupted run
resumes instead of starting the ~10 minute pull over.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API = "https://api.openaq.org/v3"
BBOX = (72.70, 18.85, 73.20, 19.40)  # min_lon, min_lat, max_lon, max_lat
POLLUTANT = "pm25"
WINDOW_MONTHS = 18

# A sensor whose last reading predates this is treated as retired.
STALE_DAYS = 30

PAGE_SIZE = 1000
TIMEOUT = 90
# Free tier is 60 requests/minute; stay just under it.
REQUEST_SPACING = 1.05
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4
# Stop halving the window here and admit the sensor is unfetchable.
MIN_CHUNK_DAYS = 7

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "pm25_raw.parquet"
PARTS_DIR = ROOT / "data" / "raw" / "_parts"

_last_request = 0.0


class ApiUnavailable(Exception):
    """The endpoint kept failing with a retryable status."""


def get_key():
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAQ_API_KEY")
    if not key:
        sys.exit("OPENAQ_API_KEY not found. Add it to .env in the project root.")
    return key


def api_get(path, key, **params):
    """GET with fixed request spacing and backoff on retryable statuses."""
    global _last_request
    for attempt in range(MAX_ATTEMPTS):
        wait = REQUEST_SPACING - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(f"{API}/{path}", headers={"X-API-Key": key},
                             params=params, timeout=TIMEOUT)
        except requests.exceptions.RequestException as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise ApiUnavailable(f"{path}: {e}") from e
            time.sleep(2 ** attempt)
            continue
        _last_request = time.monotonic()

        if r.status_code == 401:
            sys.exit("401 Unauthorized - OPENAQ_API_KEY is missing or invalid.")
        if r.status_code in RETRY_STATUS:
            if r.status_code == 429:
                time.sleep(int(r.headers.get("X-Ratelimit-Reset", 60)) + 1)
            else:
                time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()

    raise ApiUnavailable(f"{path} failed {MAX_ATTEMPTS}x")


def parse_ts(raw):
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def find_sensors(key, start, end):
    """Return one row per PM2.5 sensor whose coverage overlaps [start, end]."""
    locations = []
    page = 1
    while True:
        j = api_get("locations", key, bbox=",".join(map(str, BBOX)),
                    limit=PAGE_SIZE, page=page)
        batch = j.get("results", [])
        locations.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        page += 1

    print(f"{len(locations)} locations in bbox; checking sensor coverage...")
    now = datetime.now(timezone.utc)
    sensors = []

    for loc in locations:
        names = {s.get("parameter", {}).get("name") for s in (loc.get("sensors") or [])}
        if POLLUTANT not in names:
            continue

        coords = loc.get("coordinates") or {}
        if coords.get("latitude") is None:
            continue

        for s in api_get(f"locations/{loc['id']}/sensors", key).get("results", []):
            if s.get("parameter", {}).get("name") != POLLUTANT:
                continue
            first = parse_ts((s.get("datetimeFirst") or {}).get("utc"))
            last = parse_ts((s.get("datetimeLast") or {}).get("utc"))
            if not first or not last:
                continue
            if last < start or first > end or (now - last).days > STALE_DAYS:
                continue
            sensors.append({
                "station_id": loc["id"],
                "station_name": loc.get("name") or f"location-{loc['id']}",
                "lat": coords["latitude"],
                "lon": coords["longitude"],
                "sensor_id": s["id"],
            })

    return pd.DataFrame(sensors)


def fetch_window(key, sensor_id, start, end):
    rows = []
    page = 1
    while True:
        j = api_get(f"sensors/{sensor_id}/measurements/hourly", key,
                    datetime_from=start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    datetime_to=end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    limit=PAGE_SIZE, page=page)
        batch = j.get("results", [])
        for m in batch:
            stamp = ((m.get("period") or {}).get("datetimeFrom") or {}).get("utc")
            if stamp is None:
                continue
            rows.append({
                "timestamp": stamp,
                "value": m.get("value"),
                # OpenAQ marks values its own QA considers suspect.
                "flagged": bool((m.get("flagInfo") or {}).get("hasFlags")),
            })
        if len(batch) < PAGE_SIZE:
            return rows
        page += 1


def fetch_sensor(key, sensor_id, start, end):
    """Page one sensor, halving the window whenever the API gives up on it."""
    try:
        return fetch_window(key, sensor_id, start, end)
    except ApiUnavailable:
        if (end - start).days <= MIN_CHUNK_DAYS:
            raise
        mid = start + (end - start) / 2
        return (fetch_sensor(key, sensor_id, start, mid)
                + fetch_sensor(key, sensor_id, mid, end))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="refetch everything, ignoring cached parts")
    args = ap.parse_args()

    if RAW_PATH.exists() and not args.refresh:
        df = pd.read_parquet(RAW_PATH)
        print(f"Using cached {RAW_PATH.relative_to(ROOT)} "
              f"({len(df):,} rows, {df.station_id.nunique()} stations). "
              f"Pass --refresh to refetch.")
        return

    key = get_key()
    end = datetime.now(timezone.utc)
    start = end - relativedelta(months=WINDOW_MONTHS)
    print(f"Window: {start.date()} -> {end.date()} ({WINDOW_MONTHS} months), "
          f"pollutant={POLLUTANT}")

    sensors = find_sensors(key, start, end)
    if sensors.empty:
        sys.exit("No live PM2.5 sensors found in the bounding box.")
    print(f"{len(sensors)} live sensors across {sensors.station_id.nunique()} stations\n")

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    frames, failed = [], []

    for i, s in enumerate(sensors.itertuples(), 1):
        label = f"  [{i:>2}/{len(sensors)}] {s.station_name[:44]:<44}"
        part_path = PARTS_DIR / f"{s.sensor_id}.parquet"

        if part_path.exists() and not args.refresh:
            part = pd.read_parquet(part_path)
            print(f"{label} {len(part):>6} hours (cached)")
            frames.append(part)
            continue

        try:
            rows = fetch_sensor(key, s.sensor_id, start, end)
        except ApiUnavailable as e:
            print(f"{label}  FAILED: {e}")
            failed.append((s.station_name, s.sensor_id, str(e)))
            continue

        if not rows:
            print(f"{label} {0:>6} hours")
            continue

        part = pd.DataFrame(rows)
        part["timestamp"] = pd.to_datetime(part["timestamp"], utc=True)
        # Halved windows share a boundary hour.
        part = part.drop_duplicates(subset=["timestamp"])
        part["station_id"] = s.station_id
        part["station_name"] = s.station_name
        part["lat"] = s.lat
        part["lon"] = s.lon
        part["sensor_id"] = s.sensor_id
        part.to_parquet(part_path, index=False)

        print(f"{label} {len(part):>6} hours")
        frames.append(part)

    if not frames:
        sys.exit("Every sensor returned zero rows.")

    df = pd.concat(frames, ignore_index=True)
    df["pollutant"] = POLLUTANT
    df = df[["station_id", "station_name", "lat", "lon", "sensor_id",
             "timestamp", "pollutant", "value", "flagged"]]

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_PATH, index=False)
    print(f"\nWrote {len(df):,} rows from {df.station_id.nunique()} stations "
          f"to {RAW_PATH.relative_to(ROOT)}")

    if failed:
        print(f"\n{len(failed)} sensor(s) could not be fetched:")
        for name, sid, err in failed:
            print(f"  {name} [sensor {sid}]: {err}")
        print("Re-run to retry them; successful sensors are cached.")


if __name__ == "__main__":
    main()
