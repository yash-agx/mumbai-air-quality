"""Load the CPCB CCR 2024 hourly PM2.5 exports, clean them, and build daily means.

Run directly: python data/load_cpcb_2024.py

OpenAQ has no 2024 PM2.5 for any panel location (see data/fetch_2024.py), so
2024 comes from the CPCB CCR portal instead: one manual export per station in
data/raw/cpcb_2024/, named

    Raw_Data_2024_site_<ccr id>_<station slug>_1Hr.csv

with a "Timestamp" column (IST, start of the hour) and "PM2.5 (µg/m³)".

Each file is matched to a panel station by its name slug against the station
names in data/raw/pm25_raw.parquet. Panel coordinates and ids are used, since
the exports carry neither. The hourly rows then go through data/clean.py's
clean() unchanged and data/fetch_2024.py's daily_means() (IST days, >= 18
cleaned hours), and are written to data/interim/monitors_2024_daily.parquet.

Files whose PM2.5 series are identical are dropped, all of them. Two
stations cannot share an hourly record, so such a pair is a mis-download and
neither can be attributed.

The printed summary is availability only (hours, days, groups). It prints no
concentration statistics.
"""

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "cpcb_2024"
SAT = ROOT / "data" / "interim" / "satellite_mumbai_2024.nc"
PM_COL = "PM2.5 (µg/m³)"
FILE_RE = re.compile(r"Raw_Data_2024_site_(\d+)_(.+)_1Hr\.csv$")
CCR_TZ = "Asia/Kolkata"
# clean() floors to the UTC hour; IST hours sit on :30 UTC.
LABEL_SHIFT = pd.Timedelta(minutes=30)
MIN_GROUPS_PER_DAY = 10

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _key(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def read_files(stations):
    """One frame of hourly rows per file, mapped to panel ids."""
    by_key = {_key(n): r for n, r in zip(stations["station_name"], stations.itertuples())}
    frames, series_hash, mapping = [], {}, []
    for path in sorted(SRC.glob("*.csv")):
        m = FILE_RE.search(path.name)
        if not m:
            sys.exit(f"unexpected file name: {path.name}")
        ccr_id, slug = int(m.group(1)), m.group(2)
        st = by_key.get(_key(slug))
        if st is None:
            sys.exit(f"{path.name}: no panel station matches '{slug}'")

        d = pd.read_csv(path, usecols=["Timestamp", PM_COL])
        ts = pd.to_datetime(d["Timestamp"], format="%Y-%m-%d %H:%M:%S")
        part = pd.DataFrame({
            "station_id": st.station_id,
            "station_name": st.station_name,
            "lat": st.lat,
            "lon": st.lon,
            "sensor_id": ccr_id,
            "timestamp": ts.dt.tz_localize(CCR_TZ).dt.tz_convert("UTC"),
            "pollutant": "pm25",
            "value": pd.to_numeric(d[PM_COL], errors="coerce"),
            # CCR exports carry no QA flag.
            "flagged": False,
        })
        h = hashlib.md5(part["value"].to_numpy(dtype="float64").tobytes()).hexdigest()
        series_hash.setdefault(h, []).append(path.name)
        mapping.append({"file": path.name, "ccr_id": ccr_id, "station_id": st.station_id,
                        "station_name": st.station_name, "hash": h,
                        "reported_h": int(part["value"].notna().sum())})
        frames.append(part)

    mapping = pd.DataFrame(mapping)
    if mapping["station_id"].duplicated().any():
        sys.exit("two files map to one panel station:\n"
                 + mapping[mapping["station_id"].duplicated(keep=False)].to_string())
    dup = {h for h, names in series_hash.items() if len(names) > 1}
    mapping["duplicate_series"] = mapping["hash"].isin(dup)
    keep = set(mapping.loc[~mapping["duplicate_series"], "station_id"])
    raw = pd.concat(frames, ignore_index=True)
    return raw[raw["station_id"].isin(keep)].reset_index(drop=True), mapping


def match_distances(stations):
    """Nearest land cell for each in-box station (haversine km)."""
    import netCDF4
    c = _load("clean", ROOT / "data" / "clean.py")
    with netCDF4.Dataset(SAT) as ds:
        lat, lon = ds["lat"][:].data, ds["lon"][:].data
        land = np.isfinite(ds["pm25"][:].filled(np.nan)).all(axis=0)
    ii, jj = np.nonzero(land)
    out = {}
    for r in stations.itertuples():
        if not (lat.min() - 0.005 <= r.lat <= lat.max() + 0.005
                and lon.min() - 0.005 <= r.lon <= lon.max() + 0.005):
            continue
        d = [c.haversine_km(r.lat, r.lon, lat[i], lon[j]) for i, j in zip(ii, jj)]
        out[r.station_id] = round(1000 * min(d))
    return out


def main():
    f24 = _load("fetch_2024", ROOT / "data" / "fetch_2024.py")
    c = _load("clean", ROOT / "data" / "clean.py")
    stations = f24.panel_stations()

    raw, mapping = read_files(stations)
    cleaned, rep = c.clean(raw)
    daily, hourly = f24.daily_means(cleaned, label_shift=LABEL_SHIFT)
    f24.DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(f24.DAILY_PATH, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)

    print("files -> panel stations")
    print(mapping.drop(columns=["hash"]).to_string(index=False))
    if mapping["duplicate_series"].any():
        print("\nDROPPED: files with identical PM2.5 series (mis-download, unattributable):")
        for n in mapping.loc[mapping["duplicate_series"], "file"]:
            print(f"  {n}")

    a = rep["artefact"]
    print(f"\ncleaning (data/clean.py, unchanged): rows in {rep['rows_in']:,}")
    print(f"  QA-flagged -{rep['flagged']:,}  null -{rep['null']:,}  "
          f"out of range -{rep['out_of_range']:,}  dup hours -{rep['duplicate_hours']:,}")
    print(f"  saturation -{a['saturation']:,}  lone spikes -{a['lone_spike']:,}  "
          f"stuck runs -{rep['stuck_masked']:,}  rows out {rep['rows_out']:,}")

    dist = match_distances(stations)
    hours = hourly.groupby("station_id").size()
    per = (daily.groupby(["station_id", "site", "agency", "cv_group"])
                .agg(days=("date", "size"), first=("date", "min"), last=("date", "max"))
                .reset_index())
    per["clean_hours"] = per["station_id"].map(hours)
    per["match_m"] = per["station_id"].map(dist)
    for col in ("first", "last"):
        per[col] = per[col].dt.date
    print(f"\nper station (days = IST days with >= {f24.MIN_HOURS_PER_DAY} cleaned hours, of 366)")
    print(per.sort_values("days", ascending=False).to_string(index=False))

    grp_days = daily.groupby(["cv_group", "date"]).size().reset_index()
    per_grp = grp_days.groupby("cv_group").size().rename("days")
    members = daily.groupby("cv_group")["station_id"].nunique().rename("monitors")
    print("\nper cv_group (a group-day exists if any member has a valid day)")
    print(pd.concat([members, per_grp], axis=1).sort_values("days", ascending=False)
            .to_string())

    groups_per_day = grp_days.groupby("date").size()
    kept = groups_per_day[groups_per_day >= MIN_GROUPS_PER_DAY].index
    kept_gd = grp_days[grp_days["date"].isin(kept)]
    gk = kept_gd.groupby("cv_group").size()
    print(f"\ngroups per day: median {int(groups_per_day.median())}, "
          f"range {groups_per_day.min()}-{groups_per_day.max()}")
    print(f"kept days (>= {MIN_GROUPS_PER_DAY} groups): {len(kept)} of 366; "
          f"{len(kept_gd):,} group-days on them")
    print(f"groups with >= 60 kept days: {(gk >= 60).sum()}  "
          f"(confirmatory minimum: >= 15 groups and >= 120 kept days -> "
          f"{'MET' if (gk >= 60).sum() >= 15 and len(kept) >= 120 else 'NOT MET'})")
    by_month = pd.Series(kept).dt.strftime("%Y-%m").value_counts().sort_index()
    print("kept days by month: " + ", ".join(f"{m[5:]}:{n}" for m, n in by_month.items()))
    print(f"\n{daily['station_id'].nunique()} stations, {daily['cv_group'].nunique()} groups, "
          f"{len(daily):,} station-days -> {f24.DAILY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
