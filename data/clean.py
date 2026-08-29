"""Clean the raw OpenAQ PM2.5 pull and print a data quality report.

Run directly: python data/clean.py

Reads  data/raw/pm25_raw.parquet   (written by data/fetch.py)
Writes data/processed/pm25_clean.parquet
"""

import sys
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "pm25_raw.parquet"
CLEAN_PATH = ROOT / "data" / "processed" / "pm25_clean.parquet"

# A valid hourly PM2.5 mean. Anything outside is a sensor fault, not weather:
# OpenAQ passes through -999 sentinels, and Mumbai has never plausibly held a
# 1000 ug/m3 hourly average.
VALID_RANGE = (0.0, 1000.0)

# --------------------------------------------------------------------------
# SPATIAL CV GROUPS -- do not drop this without reading the note.
#
# These station ids sit close enough that holding one out while the other
# trains is not a real spatial holdout: the model can read the answer off its
# neighbour and leave-one-station-out CV reports a score it has not earned.
# Phase 2 must hold out whole cv_group values, never individual station_ids.
#
#   bandra_east    3409328 Bandra Kurla Complex - IITM   19.0627,  72.84614
#                  3409486 Kherwadi_Bandra East - MPCB   19.06321, 72.84563  ~70 m
#                  7850    Bandra - MPCB                 19.0627,  72.84614  same coords (retired 2021)
#
#   borivali_east  6965    Borivali East - MPCB          19.22747, 72.86439
#                  11606   Borivali East - IITM          19.23241, 72.86895  ~640 m
#
#   worli          3409323 Worli - MPCB                  18.99362, 72.81281
#                  6959    Siddharth Nagar-Worli - IITM  19.00008, 72.81399  ~730 m
#
#   ulhasnagar     3409484 Sidhi Vinayak Nagar - MPCB    19.23558, 73.15912
#                  6258871 Vithalwadi - MPCB             19.23076, 73.15519  ~680 m
# --------------------------------------------------------------------------
CV_STATION_GROUPS = {
    3409328: "bandra_east",
    3409486: "bandra_east",
    7850: "bandra_east",
    6965: "borivali_east",
    11606: "borivali_east",
    3409323: "worli",
    6959: "worli",
    3409484: "ulhasnagar",
    6258871: "ulhasnagar",
}

# Warn if any ungrouped pair is closer than this.
PROXIMITY_WARN_KM = 1.0

# --------------------------------------------------------------------------
# EXCLUDED STATIONS
#
#   8039  "Mumbai"  19.07283, 72.88261
#     294 readings over an 18-month window (2.2% coverage) around a single
#     12,764 h hole, and a bare city name where every other station carries a
#     site and an agency -- so it cannot be matched to a real monitoring site
#     or grouped against its neighbours. It can neither inform an
#     interpolation nor serve as a held-out station, so it is dropped outright
#     instead of being carried as a mostly-empty row.
# --------------------------------------------------------------------------
EXCLUDE_STATIONS = {8039}

# --------------------------------------------------------------------------
# ARTEFACT SCREEN
#
# VALID_RANGE only catches the physically absurd. Two narrower rules catch
# faults that sit inside it, each justified by something in the data rather
# than by a round number:
#
#   saturation   985.0 is the exact maximum at five unrelated stations (Sion,
#                Borivali East, Airport T2, Mahape, Khadakpada). Independent
#                sensors do not agree on a maximum to one decimal place unless
#                it is a ceiling they are all pinned against.
#
#   lone spike   A reading above SPIKE_MIN while the median of every other
#                reporting station that hour is below SPIKE_REGIONAL_MAX.
#                PM2.5 episodes are regional -- a stagnant winter night lifts
#                the whole basin. One station at 690 with the rest of the city
#                at 23 is a sensor, not weather. The corroboration is what does
#                the work here: a high reading that the city agrees with is
#                kept no matter how high it goes.
# --------------------------------------------------------------------------
SATURATION_VALUE = 985.0
SPIKE_MIN = 150.0
SPIKE_REGIONAL_MAX = 50.0

# A sensor stuck on one reading repeats it hour after hour. Runs are measured
# in unbroken hours: an outage ends a run, because a station reading 41.0
# before a two-week silence and 41.0 after it was not stuck for two weeks.
STUCK_RUN_HOURS = 24


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


def split_name(name):
    """Split "Kurla, Mumbai - MPCB" into site and agency.

    The same physical site appears under different reporting agencies, so the
    agency suffix has to come off before names can be compared at all.
    """
    site, sep, agency = name.rpartition(" - ")
    return (site.strip(), agency.strip()) if sep else (name.strip(), "")


def run_lengths(df):
    """Length, in unbroken hours, of the identical-value run each row sits in.

    df must be sorted by station then timestamp. A run continues only while the
    next row is the same station, the very next hour, and the same value, so a
    missing hour splits a run rather than bridging it.
    """
    same_station = df["station_id"].eq(df["station_id"].shift())
    next_hour = df["timestamp"].diff().dt.total_seconds().eq(3600)
    same_value = df["value"].eq(df["value"].shift())
    run_id = (~(same_station & next_hour & same_value)).cumsum()
    return df.groupby(run_id)["value"].transform("size")


def stuck_cost(df, lengths, stuck):
    """Per-station tally of what the stuck mask removes."""
    if not stuck.any():
        return pd.DataFrame(columns=["station_id", "site", "masked_h",
                                     "longest_run_h", "of_obs", "pct"])
    obs = df.groupby("station_id").size()
    hit = df.loc[stuck, ["station_id", "site"]].assign(run=lengths[stuck])
    cost = (hit.groupby(["station_id", "site"])
               .agg(masked_h=("run", "size"), longest_run_h=("run", "max"))
               .reset_index())
    cost["of_obs"] = cost["station_id"].map(obs)
    cost["pct"] = (100 * cost["masked_h"] / cost["of_obs"]).round(1)
    return cost.sort_values("masked_h", ascending=False)


def other_station_median(df, candidates):
    """Median across every station but this one, for each candidate reading.

    Computed leave-one-out and only for candidates, which is both exact and
    cheap: a few thousand rows rather than the whole panel.
    """
    hours = df[df["timestamp"].isin(set(candidates["timestamp"]))]
    by_hour = {t: g.to_numpy() for t, g in hours.groupby("timestamp")["value"]}
    out = np.empty(len(candidates))
    for i, (t, v) in enumerate(zip(candidates["timestamp"], candidates["value"])):
        vals = by_hour[t]
        rest = np.delete(vals, np.argmax(vals == v))
        # Nothing to corroborate against: keep the reading rather than delete
        # what we cannot show is wrong. NaN fails the < comparison downstream.
        out[i] = np.median(rest) if len(rest) else np.nan
    return out


def artefact_screen(df):
    """Drop saturated readings, then uncorroborated spikes. Returns (df, report)."""
    rep = {}
    sat = df["value"] == SATURATION_VALUE
    rep["saturation"] = int(sat.sum())
    rep["saturation_stations"] = int(df.loc[sat, "station_id"].nunique())
    df = df[~sat]

    cand = df[df["value"] > SPIKE_MIN]
    rep["above_spike_min"] = len(cand)
    if len(cand):
        med = other_station_median(df, cand)
        lone = cand.index[(med < SPIKE_REGIONAL_MAX)]
        rep["uncorroborated"] = int(np.isnan(med).sum())
        rep["lone_spike"] = len(lone)
        rep["lone_spike_stations"] = int(df.loc[lone, "station_id"].nunique())
        rep["survivors"] = df.loc[cand.index.difference(lone)].copy()
        df = df.drop(index=lone)
    else:
        rep.update(uncorroborated=0, lone_spike=0, lone_spike_stations=0,
                   survivors=df.iloc[:0].copy())
    return df, rep


def clean(raw):
    n0 = len(raw)
    report = {}

    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    excluded = df["station_id"].isin(EXCLUDE_STATIONS)
    report["excluded"] = int(excluded.sum())
    report["excluded_ids"] = sorted(set(df.loc[excluded, "station_id"]))
    df = df[~excluded]

    report["flagged"] = int(df["flagged"].sum())
    df = df[~df["flagged"]].drop(columns=["flagged"])

    null_mask = df["value"].isna()
    report["null"] = int(null_mask.sum())
    df = df[~null_mask]

    lo, hi = VALID_RANGE
    bad = (df["value"] <= lo) | (df["value"] > hi)
    report["out_of_range"] = int(bad.sum())
    df = df[~bad]

    # Snap to the exact hour: OpenAQ stamps some series at :15 or :30 past.
    df["timestamp"] = df["timestamp"].dt.floor("h")

    # A station with two overlapping sensors yields two rows for one hour.
    report["duplicate_hours"] = int(
        df.duplicated(subset=["station_id", "timestamp"]).sum())
    df = (df.groupby(["station_id", "timestamp"], as_index=False)
            .agg(station_name=("station_name", "first"),
                 lat=("lat", "first"),
                 lon=("lon", "first"),
                 pollutant=("pollutant", "first"),
                 value=("value", "mean")))

    df[["site", "agency"]] = df["station_name"].apply(
        lambda n: pd.Series(split_name(n)))
    df["cv_group"] = df["station_id"].map(CV_STATION_GROUPS).fillna(
        df["station_id"].astype(str))

    # Screen artefacts before the stuck mask, so a sensor pinned at 985.0 is
    # reported as the saturation it is rather than as a stuck run.
    df, art = artefact_screen(df)
    report["artefact"] = art

    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    # Mask stuck runs rather than dropping the station: the rest of the series
    # is usable, and a masked hour is just a missing hour, which the pipeline
    # already handles everywhere else.
    lengths = run_lengths(df)
    stuck = lengths >= STUCK_RUN_HOURS
    report["stuck_masked"] = int(stuck.sum())
    report["stuck"] = stuck_cost(df, lengths, stuck)
    df = df[~stuck]

    report["rows_in"] = n0
    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def per_station(df):
    window_hours = int(
        (df["timestamp"].max() - df["timestamp"].min()).total_seconds() // 3600) + 1

    rows = []
    for sid, g in df.groupby("station_id"):
        g = g.sort_values("timestamp")
        # Hours absent between consecutive readings, so an unbroken series is 0.
        gaps = g["timestamp"].diff().dt.total_seconds().div(3600).sub(1)
        rows.append({
            "station_id": sid,
            "site": g["site"].iloc[0],
            "agency": g["agency"].iloc[0],
            "cv_group": g["cv_group"].iloc[0],
            "lat": round(g["lat"].iloc[0], 5),
            "lon": round(g["lon"].iloc[0], 5),
            "n_obs": len(g),
            "coverage_pct": round(100 * len(g) / window_hours, 1),
            "first": g["timestamp"].min().date(),
            "last": g["timestamp"].max().date(),
            "max_gap_h": int(gaps.max()) if len(g) > 1 else 0,
            # Post-mask, so this is always < STUCK_RUN_HOURS.
            "stuck_run_h": int(run_lengths(g).max()),
        })
    return pd.DataFrame(rows).sort_values("coverage_pct", ascending=False), window_hours


def proximity_check(stations):
    """Flag close pairs that are not already in the same cv_group."""
    warn = []
    recs = stations.to_dict("records")
    for i, a in enumerate(recs):
        for b in recs[i + 1:]:
            if a["cv_group"] == b["cv_group"]:
                continue
            d = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
            if d < PROXIMITY_WARN_KM:
                warn.append((d, a["site"], a["station_id"], b["site"], b["station_id"]))
    return sorted(warn)


def main():
    if not RAW_PATH.exists():
        sys.exit(f"{RAW_PATH.relative_to(ROOT)} not found - run data/fetch.py first.")

    raw = pd.read_parquet(RAW_PATH)
    df, rep = clean(raw)
    stations, window_hours = per_station(df)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[["station_id", "lat", "lon", "timestamp", "pollutant", "value",
        "site", "agency", "cv_group"]].to_parquet(CLEAN_PATH, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 100)

    print("=" * 92)
    print("PM2.5 DATA QUALITY REPORT")
    print("=" * 92)
    print(f"window          {df['timestamp'].min()}  ->  {df['timestamp'].max()}")
    print(f"                {window_hours:,} hours, {len(stations)} stations, "
          f"{df['cv_group'].nunique()} CV groups")
    print(f"\nrows in         {rep['rows_in']:,}")
    print(f"  excluded stn  -{rep['excluded']:,}   "
          f"(station {', '.join(map(str, rep['excluded_ids']))}, see EXCLUDE_STATIONS)")
    print(f"  QA-flagged    -{rep['flagged']:,}")
    print(f"  null value    -{rep['null']:,}")
    print(f"  out of range  -{rep['out_of_range']:,}   "
          f"(outside {VALID_RANGE[0]}-{VALID_RANGE[1]} ug/m3)")
    print(f"  dup hours     -{rep['duplicate_hours']:,}   "
          f"(overlapping sensors, averaged)")
    a = rep["artefact"]
    print(f"  saturation    -{a['saturation']:,}   "
          f"(exactly {SATURATION_VALUE:g} ug/m3 at {a['saturation_stations']} stations)")
    print(f"  lone spikes   -{a['lone_spike']:,}   "
          f"(>{SPIKE_MIN:g} while other stations' median <{SPIKE_REGIONAL_MAX:g})")
    print(f"  stuck runs    -{rep['stuck_masked']:,}   "
          f"(one value repeated {STUCK_RUN_HOURS}h+ unbroken, masked)")
    print(f"rows out        {rep['rows_out']:,}")

    possible = window_hours * len(stations)
    dense = 100 * rep["rows_out"] / possible
    print(f"\nstation-hours   {rep['rows_out']:,} of {possible:,} possible "
          f"= {dense:.1f}% dense ({100 - dense:.1f}% missing)")

    print("\n" + "-" * 92)
    print("PER STATION  (coverage_pct = share of full window; max_gap_h = longest outage)")
    print("-" * 92)
    print(stations.to_string(index=False))

    print("\n" + "-" * 92)
    print("NOTES")
    print("-" * 92)

    grouped = stations[stations["cv_group"].isin(set(CV_STATION_GROUPS.values()))]
    if not grouped.empty:
        print("Near-coincident stations grouped for spatial CV "
              "(hold out the whole group, not the station):")
        for gname, g in grouped.groupby("cv_group"):
            ids = ", ".join(f"{r.site} [{r.station_id}]" for r in g.itertuples())
            print(f"  {gname:<14} {ids}")
    else:
        print("No grouped stations present in this pull.")

    warn = proximity_check(stations)
    if warn:
        print(f"\nUngrouped pairs closer than {PROXIMITY_WARN_KM} km "
              f"- consider adding to CV_STATION_GROUPS:")
        for d, sa, ia, sb, ib in warn:
            print(f"  {d * 1000:>5.0f} m   {sa} [{ia}]  <->  {sb} [{ib}]")
    else:
        print(f"\nNo ungrouped station pairs closer than {PROXIMITY_WARN_KM} km.")

    offline = stations[stations["max_gap_h"] >= 48]
    if not offline.empty:
        print(f"\nStations with an outage of 2+ days ({len(offline)} of {len(stations)}):")
        for r in offline.itertuples():
            print(f"  {r.max_gap_h:>5} h   {r.site} [{r.station_id}]")

    cost = rep["stuck"]
    if not cost.empty:
        print(f"\nStuck-run mask: {rep['stuck_masked']:,} readings removed from "
              f"{len(cost)} of {len(stations)} stations. The station is kept; the")
        print(f"masked hours become missing hours. "
              f"(longest = longest unbroken stuck run found)")
        print(f"  {'masked':>6} {'of obs':>7} {'':>6}  {'longest':>7}   station")
        for r in cost.itertuples():
            print(f"  {r.masked_h:>6,} {r.of_obs:>7,} {r.pct:>5.1f}%  "
                  f"{r.longest_run_h:>5} h   {r.site} [{r.station_id}]")
        print(f"  {cost['masked_h'].sum():>6,}   total")
    else:
        print(f"\nNo station repeats a value for {STUCK_RUN_HOURS}h+ of unbroken hours.")

    print("\n" + "-" * 92)
    surv = a["survivors"]
    print("REGIONAL EPISODES KEPT  (the check that the screen removed faults, not weather)")
    print("-" * 92)
    print(f"readings above {SPIKE_MIN:g} ug/m3: {a['above_spike_min']:,} considered, "
          f"{a['lone_spike']:,} dropped as lone spikes, {len(surv):,} kept as corroborated")
    if a["uncorroborated"]:
        print(f"  {a['uncorroborated']:,} could not be checked (no other station "
              f"reporting that hour) and were kept")
    if not surv.empty:
        hourly = df.groupby("timestamp")["value"].median()
        print(f"  kept readings span {surv['value'].min():.0f}-{surv['value'].max():.0f} "
              f"ug/m3 across {surv['station_id'].nunique()} stations "
              f"and {surv['timestamp'].nunique():,} distinct hours")
        print(f"  city-wide median across all hours peaks at {hourly.max():.0f} ug/m3; "
              f"{int((hourly >= 75).sum()):,} hours sit at or above 75")
        by_month = (surv.assign(month=surv["timestamp"].dt.tz_convert("Asia/Kolkata")
                                .dt.strftime("%Y-%m"))
                        .groupby("month").size())
        print("  kept high readings by month (Mumbai's season runs Nov-Feb):")
        for month, n in by_month.items():
            print(f"    {month}  {n:>5,}  {'#' * min(60, int(n / max(1, by_month.max() / 60)))}")
    else:
        print("  WARNING: no reading above the threshold survived - the screen is "
              "removing real episodes, not just faults.")

    print(f"\nWrote {len(df):,} rows to {CLEAN_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
