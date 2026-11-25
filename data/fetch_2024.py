"""Fetch 2024 hourly PM2.5 for the 39 panel monitors and reduce it to daily means.

Run directly: python data/fetch_2024.py [--refresh]

The comparison target is the IIT Delhi daily satellite PM2.5 for 2024
(data/interim/satellite_mumbai_2024.nc), so this pulls calendar-year 2024
for the same locations as the main panel (the 39 station ids in
data/processed/pm25_clean.parquet). It reuses data/fetch.py's request and
paging code and data/clean.py's clean() without changes.

One difference from data/fetch.py's sensor selection: that script drops any
sensor whose last reading is older than STALE_DAYS, because it wants live
sensors. For 2024 the sensors that were live then are exactly the ones that
may have since retired, so every PM2.5 sensor at a panel location whose
coverage overlaps 2024 is used.

Days are IST calendar days. A station-day is kept only with >= 18 cleaned
hours.

Writes
  data/raw/pm25_2024_raw.parquet          one row per sensor-hour, as fetched
  data/raw/_parts_2024/                   per-sensor cache
  data/interim/monitors_2024_daily.parquet  station_id, site, agency, cv_group,
                                          lat, lon, date, pm25, n_hours

The printed summary is availability only (sensors, hours, days). It prints no
concentration statistics, so it can be run before the satellite comparison
is pre-registered.
"""

import argparse
import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PM25 = ROOT / "data" / "processed" / "pm25_clean.parquet"
PANEL_RAW = ROOT / "data" / "raw" / "pm25_raw.parquet"
RAW_PATH = ROOT / "data" / "raw" / "pm25_2024_raw.parquet"
PARTS_DIR = ROOT / "data" / "raw" / "_parts_2024"
DAILY_PATH = ROOT / "data" / "interim" / "monitors_2024_daily.parquet"

TZ = "Asia/Kolkata"
START = pd.Timestamp("2024-01-01", tz=TZ).tz_convert("UTC")
END = pd.Timestamp("2025-01-01", tz=TZ).tz_convert("UTC")  # exclusive
MIN_HOURS_PER_DAY = 18

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def panel_stations():
    """The 39 panel monitors, with the name and coordinates clean() expects."""
    ids = pd.read_parquet(CLEAN_PM25, columns=["station_id"])["station_id"].unique()
    meta = (pd.read_parquet(PANEL_RAW, columns=["station_id", "station_name", "lat", "lon"])
              .drop_duplicates("station_id"))
    return meta[meta["station_id"].isin(ids)].sort_values("station_id").reset_index(drop=True)


def find_sensors(f, key, station_ids, start, end):
    rows = []
    for sid in station_ids:
        for s in f.api_get(f"locations/{sid}/sensors", key).get("results", []):
            if (s.get("parameter") or {}).get("name") != f.POLLUTANT:
                continue
            first = f.parse_ts((s.get("datetimeFirst") or {}).get("utc"))
            last = f.parse_ts((s.get("datetimeLast") or {}).get("utc"))
            overlaps = bool(first and last and last >= start and first <= end)
            rows.append({"station_id": sid, "sensor_id": s["id"],
                         "first": first, "last": last, "overlaps": overlaps})
    return pd.DataFrame(rows, columns=["station_id", "sensor_id", "first", "last", "overlaps"])


def fetch(refresh=False):
    f = _load("fetch", ROOT / "data" / "fetch.py")
    key = f.get_key()
    stations = panel_stations()
    # Pad a day each side; the window is applied after cleaning.
    q_start = (START - timedelta(days=1)).to_pydatetime()
    q_end = (END + timedelta(days=1)).to_pydatetime()
    print(f"2024 window (IST) {START} -> {END} UTC; {len(stations)} stations")

    sensors = find_sensors(f, key, stations["station_id"], q_start, q_end)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    sensors.to_parquet(PARTS_DIR / "_sensors.parquet", index=False)
    meta = stations.set_index("station_id")

    frames = []
    for s in sensors[sensors["overlaps"]].itertuples():
        part_path = PARTS_DIR / f"{s.sensor_id}.parquet"
        if part_path.exists() and not refresh:
            frames.append(pd.read_parquet(part_path))
            continue
        try:
            rows = f.fetch_sensor(key, s.sensor_id, q_start, q_end)
        except f.ApiUnavailable as e:
            print(f"  sensor {s.sensor_id} FAILED: {e}")
            continue
        part = pd.DataFrame(rows, columns=["timestamp", "value", "flagged"])
        part["timestamp"] = pd.to_datetime(part["timestamp"], utc=True)
        part = part.drop_duplicates(subset=["timestamp"])
        m = meta.loc[s.station_id]
        part["station_id"] = s.station_id
        part["station_name"] = m["station_name"]
        part["lat"] = m["lat"]
        part["lon"] = m["lon"]
        part["sensor_id"] = s.sensor_id
        part.to_parquet(part_path, index=False)
        print(f"  station {s.station_id} sensor {s.sensor_id}: {len(part)} rows")
        frames.append(part)

    cols = ["station_id", "station_name", "lat", "lon", "sensor_id",
            "timestamp", "pollutant", "value", "flagged"]
    if frames:
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.DataFrame(columns=[c for c in cols if c != "pollutant"])
    df["pollutant"] = f.POLLUTANT
    df = df[cols]
    df["flagged"] = df["flagged"].astype(bool)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_PATH, index=False)
    return df, sensors, stations


def daily_means(clean_df, label_shift=pd.Timedelta(0)):
    """IST calendar-day means, kept only where >= MIN_HOURS_PER_DAY hours survive.

    clean() floors timestamps to the UTC hour. For a source stamped on whole
    IST hours (CPCB CCR) that moves every reading 30 minutes early, which would
    put IST midnight in the previous day; label_shift undoes it for the day
    assignment only.
    """
    stamp = clean_df["timestamp"] + label_shift
    d = clean_df[(stamp >= START) & (stamp < END)].copy()
    d["date"] = (d["timestamp"] + label_shift).dt.tz_convert(TZ).dt.date
    g = (d.groupby(["station_id", "date"])
          .agg(site=("site", "first"), agency=("agency", "first"),
               cv_group=("cv_group", "first"), lat=("lat", "first"),
               lon=("lon", "first"), pm25=("value", "mean"),
               n_hours=("value", "size"))
          .reset_index())
    g = g[g["n_hours"] >= MIN_HOURS_PER_DAY].copy()
    g["date"] = pd.to_datetime(g["date"])
    return g[["station_id", "site", "agency", "cv_group", "lat", "lon",
              "date", "pm25", "n_hours"]].reset_index(drop=True), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    raw, sensors, stations = fetch(args.refresh)
    if raw.empty:
        sys.exit("No 2024 PM2.5 rows for any panel station.")

    c = _load("clean", ROOT / "data" / "clean.py")
    cleaned, rep = c.clean(raw)
    daily, hourly = daily_means(cleaned)
    DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(DAILY_PATH, index=False)

    a = rep["artefact"]
    print(f"\ncleaning (data/clean.py, unchanged): rows in {rep['rows_in']:,}")
    print(f"  QA-flagged -{rep['flagged']:,}  null -{rep['null']:,}  "
          f"out of range -{rep['out_of_range']:,}  dup hours -{rep['duplicate_hours']:,}")
    print(f"  saturation -{a['saturation']:,}  lone spikes -{a['lone_spike']:,}  "
          f"stuck runs -{rep['stuck_masked']:,}  rows out {rep['rows_out']:,}")

    n_sensors = sensors[sensors["overlaps"]].groupby("station_id").size()
    hours = hourly.groupby("station_id").size()
    days = daily.groupby("station_id").agg(days=("date", "size"),
                                           first=("date", "min"), last=("date", "max"))
    tab = stations[["station_id", "station_name"]].copy()
    tab["sensors_2024"] = tab["station_id"].map(n_sensors).fillna(0).astype(int)
    tab["clean_hours"] = tab["station_id"].map(hours).fillna(0).astype(int)
    tab = tab.join(days, on="station_id")
    tab["days"] = tab["days"].fillna(0).astype(int)
    for col in ("first", "last"):
        tab[col] = tab[col].dt.date
    tab = tab.sort_values(["days", "clean_hours"], ascending=False)

    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(f"\nper station (days = IST days with >= {MIN_HOURS_PER_DAY} cleaned hours, of 366)")
        print(tab.to_string(index=False))
    usable = tab[tab["days"] > 0]
    print(f"\n{len(usable)} of {len(tab)} stations have >= 1 usable 2024 day; "
          f"{len(daily):,} station-days in total")
    print(f"wrote {DAILY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
