"""Fetch hourly NO2 and NOx for the 39 PM2.5 monitors from OpenAQ v3.

Run directly: python data/fetch_no2.py [--refresh]

Same source, API and window handling as data/fetch.py (whose helpers this
reuses), restricted to the locations already in the PM2.5 panel and to the
PM2.5 record's span, so the two pollutants come from the same instruments
over the same hours.

Writes data/raw/no2_raw.parquet (one row per sensor-hour, both parameters),
caching each sensor in data/raw/_parts_no2/ so an interrupted run resumes.

The printed summary is availability only: sensors, units and counts of
reported hours. It deliberately prints no concentration statistics, so it
can be run before the NO2 analysis is pre-registered.
"""

import argparse
import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PM25 = ROOT / "data" / "processed" / "pm25_clean.parquet"
RAW_PATH = ROOT / "data" / "raw" / "no2_raw.parquet"
PARTS_DIR = ROOT / "data" / "raw" / "_parts_no2"
PARAMETERS = ("no2", "nox")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _fetch_module():
    spec = importlib.util.spec_from_file_location("fetch", ROOT / "data" / "fetch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pm25_window():
    """Station ids and the [first, last] hour of the cleaned PM2.5 panel."""
    pm = pd.read_parquet(CLEAN_PM25, columns=["station_id", "timestamp", "site",
                                              "agency", "cv_group"])
    return pm, pm["timestamp"].min(), pm["timestamp"].max()


def find_sensors(f, key, station_ids, start, end):
    rows = []
    for sid in station_ids:
        for s in f.api_get(f"locations/{sid}/sensors", key).get("results", []):
            p = s.get("parameter") or {}
            if p.get("name") not in PARAMETERS:
                continue
            first = f.parse_ts((s.get("datetimeFirst") or {}).get("utc"))
            last = f.parse_ts((s.get("datetimeLast") or {}).get("utc"))
            overlaps = bool(first and last and last >= start and first <= end)
            rows.append({"station_id": sid, "sensor_id": s["id"],
                         "parameter": p.get("name"), "units": p.get("units"),
                         "first": first, "last": last, "overlaps": overlaps})
    return pd.DataFrame(rows)


def fetch(refresh=False):
    f = _fetch_module()
    key = f.get_key()
    pm, start, end = pm25_window()
    station_ids = sorted(pm["station_id"].unique())
    # Pad the request by a day each side; the window is applied when counting.
    q_start, q_end = start - timedelta(days=1), end + timedelta(days=1)
    print(f"PM2.5 window {start} -> {end}; {len(station_ids)} stations")

    sensors = find_sensors(f, key, station_ids, q_start, q_end)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for s in sensors[sensors["overlaps"]].itertuples():
        part_path = PARTS_DIR / f"{s.sensor_id}.parquet"
        if part_path.exists() and not refresh:
            frames.append(pd.read_parquet(part_path))
            continue
        try:
            rows = f.fetch_sensor(key, s.sensor_id, q_start, q_end)
        except f.ApiUnavailable as e:
            print(f"  sensor {s.sensor_id} ({s.parameter}) FAILED: {e}")
            continue
        part = pd.DataFrame(rows, columns=["timestamp", "value", "flagged"])
        part["timestamp"] = pd.to_datetime(part["timestamp"], utc=True)
        part = part.drop_duplicates(subset=["timestamp"])
        part["station_id"] = s.station_id
        part["sensor_id"] = s.sensor_id
        part["parameter"] = s.parameter
        part["units"] = s.units
        part.to_parquet(part_path, index=False)
        print(f"  station {s.station_id} sensor {s.sensor_id} {s.parameter}: "
              f"{len(part)} rows")
        frames.append(part)

    df = (pd.concat(frames, ignore_index=True) if frames else
          pd.DataFrame(columns=["timestamp", "value", "flagged", "station_id",
                                "sensor_id", "parameter", "units"]))
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_PATH, index=False)
    sensors.to_parquet(PARTS_DIR / "_sensors.parquet", index=False)
    return df, sensors, pm, start, end


def availability(df, sensors, pm, start, end):
    """Per station and parameter: reported hours inside the PM2.5 window.

    A reported hour is a non-null, unflagged reading, floored to the hour and
    de-duplicated across a station's sensors. No value is inspected beyond
    "is it present".
    """
    d = df[df["value"].notna() & ~df["flagged"].astype(bool)].copy()
    d["timestamp"] = d["timestamp"].dt.floor("h")
    d = d[(d["timestamp"] >= start) & (d["timestamp"] <= end)]
    d = d.drop_duplicates(["station_id", "parameter", "timestamp"])
    pm_hours = pm[["station_id", "timestamp"]].assign(pm25=True)
    window_h = int((end - start).total_seconds() // 3600) + 1

    meta = (pm.drop_duplicates("station_id")
              .set_index("station_id")[["site", "agency", "cv_group"]])
    out = []
    for sid, m in meta.iterrows():
        row = {"station_id": sid, "site": m["site"], "agency": m["agency"],
               "cv_group": m["cv_group"]}
        for p in PARAMETERS:
            s = sensors[(sensors["station_id"] == sid) & (sensors["parameter"] == p)]
            g = d[(d["station_id"] == sid) & (d["parameter"] == p)]
            both = g.merge(pm_hours, on=["station_id", "timestamp"], how="inner")
            row[f"{p}_sensors"] = len(s)
            row[f"{p}_units"] = ",".join(sorted(set(s["units"].dropna())))
            row[f"{p}_hours"] = len(g)
            row[f"{p}_pct"] = round(100 * len(g) / window_h, 1)
            row[f"{p}_with_pm25"] = len(both)
        out.append(row)
    return pd.DataFrame(out), window_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    df, sensors, pm, start, end = fetch(args.refresh)
    tab, window_h = availability(df, sensors, pm, start, end)
    with pd.option_context("display.width", 250, "display.max_rows", 100,
                           "display.max_columns", 30):
        print(f"\nwindow {window_h:,} hours")
        print(tab.sort_values("no2_hours", ascending=False).to_string(index=False))
    print(f"\nwrote {RAW_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
