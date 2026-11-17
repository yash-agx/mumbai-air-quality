"""Test 1 of reports/preregistration_2026-11.md: the sea/land breeze idea.

Hypothesis as posted: a daytime sea breeze pushes pollution inland and a
night-time land breeze pushes it back over the coast, so coastal monitors peak
at night and inland monitors in the afternoon.

Three parts, all fixed in the pre-registration before anything was run:

  1a  peak hour against distance to the coast, 35 cv_group units, circular-
      linear correlation with a permutation null.
  1b  wind-conditional concentration: are coastal monitors relatively dirtier
      when the wind blows offshore, and inland monitors when it blows onshore?
  1c  Powai, described rather than tested.

Wind comes from the Open-Meteo Historical Weather API (ERA5 reanalysis),
https://open-meteo.com/en/docs/historical-weather-api, free for non-commercial
use, hourly 10 m wind speed and direction at 19.08 N, 72.88 E. Cached under
data/raw/ after the first fetch; that directory is not in the repository, so a
clean checkout refetches.

Two constraints on what this can show, both from the pre-registration:

  - The hypothesis has already been round the block on this data. One
    pre-specified test of it (dist_coast_m x trough_hour) was null, and one
    exploratory variant looked strong and was never promoted. There are only
    39 monitors, so this re-uses every one of them. 1a is a re-test on used
    data; only 1b brings new information.
  - Hours are autocorrelated and every monitor shares one city-wide wind
    series, so the weather is n = 1 forcing, not thousands of independent
    hours. No hour-level p-value is computed anywhere in this file. Inference
    is at the monitor level, across 35 units.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
WIND_PATH = ROOT / "data" / "raw" / "openmeteo_wind.parquet"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WIND_LAT, WIND_LON = 19.08, 72.88
TIMEOUT = 60

# Mumbai's coast runs roughly north-south with the sea to the west, and
# meteorological direction is the direction the wind comes FROM.
ONSHORE = (225, 315)        # off the sea
OFFSHORE = (45, 135)        # off the land
CALM_MS = 1.0               # below this the direction means nothing

COASTAL_M = 2000            # the 6 monitors nearest the sea
INLAND_M = 8000             # the 10 furthest from it

# Below this many hours in a class, a window has no land breeze to test.
MIN_CLASS_HOURS = 200
# Per monitor, in each class, before it can contribute a difference.
MIN_MONITOR_HOURS = 50


def load_diurnal_module():
    """scripts/03_diurnal_landuse.py, for its statistics.

    Imported by path for the same reason interpolate.py imports the feature
    script that way: the name starts with a digit. Reused rather than copied so
    the circular statistics and the unit collapsing are literally the same code
    the earlier land-use tests ran on.
    """
    spec = importlib.util.spec_from_file_location(
        "diurnal_landuse", ROOT / "scripts" / "03_diurnal_landuse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch_wind(start, end):
    """Hourly 10 m wind for the window, cached on disk after the first call."""
    if WIND_PATH.exists():
        w = pd.read_parquet(WIND_PATH)
        if w["time"].min() <= start and w["time"].max() >= end:
            return w
    params = {"latitude": WIND_LAT, "longitude": WIND_LON,
              "start_date": start.strftime("%Y-%m-%d"),
              "end_date": end.strftime("%Y-%m-%d"),
              "hourly": "wind_speed_10m,wind_direction_10m", "timezone": "UTC"}
    r = requests.get(ARCHIVE_URL, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    h = r.json()["hourly"]
    w = pd.DataFrame({
        "time": pd.to_datetime(h["time"], utc=True),
        "wind_speed_ms": np.asarray(h["wind_speed_10m"], float) / 3.6,
        "wind_dir_deg": np.asarray(h["wind_direction_10m"], float)})
    WIND_PATH.parent.mkdir(parents=True, exist_ok=True)
    w.to_parquet(WIND_PATH)
    return w


def classify(w):
    """onshore / offshore / unclassified per hour."""
    d, s = w["wind_dir_deg"].to_numpy(), w["wind_speed_ms"].to_numpy()
    out = np.full(len(w), "unclassified", dtype=object)
    out[(d >= ONSHORE[0]) & (d <= ONSHORE[1])] = "onshore"
    out[(d >= OFFSHORE[0]) & (d <= OFFSHORE[1])] = "offshore"
    out[~np.isfinite(d) | ~np.isfinite(s) | (s < CALM_MS)] = "unclassified"
    return out


def residualise(values, hours, blocks=None):
    """Subtract each column's own mean within each hour of day (and block).

    This is what stops the test re-measuring the diurnal cycle. Offshore hours
    are mostly night and onshore hours mostly afternoon, so without it a
    difference between the two classes would mostly be the day's normal shape
    wearing a wind label.

    `blocks` is the EXPLORATORY extension, not in the pre-registration: on the
    full record the offshore hours pile into October-December, which is also
    Mumbai's dirty season, so a city-wide offshore excess there is as much
    winter as wind. Residualising within month-and-hour removes the season the
    same way the hour removes the daily cycle.
    """
    out = np.full_like(values, np.nan, dtype=float)
    keys = (hours if blocks is None
            else np.char.add(np.asarray(blocks, str), hours.astype(str)))
    for k in np.unique(keys):
        m = keys == k
        if not m.any():
            continue
        out[m] = values[m] - np.nanmean(values[m], axis=0)
    return out


def per_unit_difference(values, hours, klass, feat, dm, residual=True,
                        blocks=None):
    """Offshore-minus-onshore mean, per cv_group, in that unit's own SD."""
    v = (residualise(values, hours, blocks) if residual
         else values.astype(float))
    off, on = klass == "offshore", klass == "onshore"
    rows = []
    for j, sid in enumerate(dm.columns):
        col = v[:, j]
        sd = np.nanstd(values[:, j].astype(float))
        if not np.isfinite(sd) or sd == 0:
            continue
        a, b = col[off], col[on]
        if (np.isfinite(a).sum() < MIN_MONITOR_HOURS
                or np.isfinite(b).sum() < MIN_MONITOR_HOURS):
            continue
        rows.append({"station_id": sid,
                     "offshore_minus_onshore_sd":
                         (np.nanmean(a) - np.nanmean(b)) / sd,
                     "n_offshore": int(np.isfinite(a).sum()),
                     "n_onshore": int(np.isfinite(b).sum())})
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    d = pd.DataFrame(rows).set_index("station_id")
    d = d.join(feat[["cv_group", "dist_coast_m", "site"]])
    g = d.groupby("cv_group").agg(
        diff_sd=("offshore_minus_onshore_sd", "mean"),
        dist_coast_m=("dist_coast_m", "mean"),
        n_offshore=("n_offshore", "median"), n_onshore=("n_onshore", "median"))
    return d, g


def report_spearman(label, x, y, mod, predicted):
    rho, p, n = mod.perm_test(x, y, circular=False)
    verdict = "supports" if (p < 0.05 and np.sign(rho) == predicted) else (
        "SIGNIFICANT, WRONG DIRECTION" if p < 0.05 else "null")
    print(f"  {label}")
    print(f"    rho = {rho:+.3f}   p = {p:.4f}   n = {n} units   -> {verdict}")
    return rho, p, n


def run_1a(mod, wide, feat, hours, mask, label):
    """Peak hour against distance to the coast, on one window."""
    print("-" * 72)
    print(f"1a. peak hour against distance to the coast  [{label}]")
    print("-" * 72)
    prof = mod.profiles_for(wide, hours, mask)
    units = mod.to_groups(mod.summary_table(prof), feat)

    stat, p, n = mod.perm_test(units["dist_coast_m"], units["peak_hour"],
                               circular=True)
    print(f"  circular-linear R = {stat:.3f}   p = {p:.4f}   n = {n} units")
    print(f"  -> {'supports' if p < 0.05 else 'null'}")
    print("  (R has no sign; the direction is read off the groups below)")

    near = units[units["dist_coast_m"] < COASTAL_M]
    far = units[units["dist_coast_m"] > INLAND_M]
    for name, sub in (("coastal", near), ("inland ", far)):
        th = np.asarray(sub["peak_hour"], float) * 2 * np.pi / 24
        rbar = float(np.hypot(np.cos(th).mean(), np.sin(th).mean()))
        print(f"  {name} (n={len(sub)} units): mean peak hour "
              f"{mod.circ_mean(sub['peak_hour']):.1f}h, concentration "
              f"R-bar = {rbar:.2f} "
              f"({'tight' if rbar > 0.7 else 'scattered, so the mean says little'})")
        print("      peak hours: "
              + " ".join(f"{h:.1f}" for h in sorted(sub["peak_hour"])))
    if len(near) and len(far):
        u = stats.mannwhitneyu(near["peak_hour"], far["peak_hour"])
        print(f"  Mann-Whitney on peak hour (secondary; peak hour is circular "
              f"so this is a weak test): p = {u.pvalue:.4f}")
    print("  The breeze story predicts inland LATER (afternoon) than coastal.")
    print()
    return prof, units


def run_1b(mod, wide, feat, ist, hours, mask, label):
    """Wind-conditional concentration, on one window."""
    print("-" * 72)
    print(f"1b. wind-conditional concentration  [{label}]")
    print("-" * 72)
    idx = wide.index[mask]
    try:
        w = fetch_wind(idx.min(), idx.max())
    except Exception as exc:                       # offline, or API down
        sys.exit(f"could not fetch wind from Open-Meteo: {exc}")

    klass = classify(w.set_index("time").reindex(idx))
    counts = pd.Series(klass).value_counts()
    print(f"  Open-Meteo ERA5, {WIND_LAT} N {WIND_LON} E, hourly 10 m wind")
    for k in ("onshore", "offshore", "unclassified"):
        print(f"    {k:14s} {int(counts.get(k, 0)):6,d} hours "
              f"({100 * counts.get(k, 0) / len(klass):4.1f}%)")

    frame = pd.DataFrame({"klass": klass, "hour": hours[mask],
                          "month": ist[mask].strftime("%Y-%m")})
    off_by_hour = frame[frame["klass"] == "offshore"]["hour"].value_counts()
    on_by_hour = frame[frame["klass"] == "onshore"]["hour"].value_counts()
    print("\n  when each class happens (the confound the residualising is for):")
    print("    hour  offshore  onshore")
    for h in range(0, 24, 3):
        print(f"    {h:02d}:00 {int(off_by_hour.get(h, 0)):9,d} "
              f"{int(on_by_hour.get(h, 0)):8,d}")
    print("    offshore share by month:")
    for m, sub in frame.groupby("month"):
        print(f"      {m}  {100 * (sub['klass'] == 'offshore').mean():5.1f}%  "
              f"({int((sub['klass'] == 'offshore').sum()):,} h)")

    if int(counts.get("offshore", 0)) < MIN_CLASS_HOURS:
        print(f"\n  Fewer than {MIN_CLASS_HOURS} offshore hours. Per the "
              f"pre-registration the count IS the result for this window: "
              f"there is almost no land breeze in it to test against.")

    values = wide[mask].to_numpy(dtype=float)
    hrs = hours[mask]
    months = ist[mask].strftime("%Y-%m").to_numpy()
    print()
    out = {}
    for residual, blocks, name in (
            (True, None, "PRIMARY, hour-of-day residualised"),
            (False, None, "raw, CONFOUNDED with the diurnal cycle"),
            (True, months, "EXPLORATORY, month-and-hour residualised "
                           "(not pre-registered; removes the dirty season "
                           "the offshore hours sit in)")):
        per_st, per_unit = per_unit_difference(values, hrs, klass, feat,
                                               wide[mask], residual=residual,
                                               blocks=blocks)
        if len(per_unit) < 10:
            print(f"  {name}: only {len(per_unit)} units cleared the "
                  f"{MIN_MONITOR_HOURS}-hour minimum in each class; "
                  f"not testable")
            continue
        print(f"  {name}  ({len(per_unit)} units, median "
              f"{int(per_unit['n_offshore'].median())} offshore hours each)")
        report_spearman(
            "dist_coast_m vs (offshore - onshore), in each monitor's own SD",
            per_unit["dist_coast_m"], per_unit["diff_sd"], mod, predicted=-1)
        near_u = per_unit[per_unit["dist_coast_m"] < COASTAL_M]["diff_sd"]
        far_u = per_unit[per_unit["dist_coast_m"] > INLAND_M]["diff_sd"]
        print(f"    coastal units mean {near_u.mean():+.3f} SD "
              f"(n={len(near_u)}), inland units mean {far_u.mean():+.3f} SD "
              f"(n={len(far_u)})")
        if len(near_u) and len(far_u):
            print(f"    Mann-Whitney coastal vs inland: "
                  f"p = {stats.mannwhitneyu(near_u, far_u).pvalue:.4f}")
        print()
        if residual and blocks is None:
            out = {"per_station": per_st, "per_unit": per_unit}
    return out


def main():
    mod = load_diurnal_module()
    interp = mod.load_engine()
    wide, feat, _ = interp.load()
    ist = wide.index.tz_convert(mod.TZ)
    hours = ist.hour.to_numpy()
    common = np.asarray(ist >= pd.Timestamp(mod.COMMON_START, tz=mod.TZ))
    everything = np.ones(len(ist), dtype=bool)

    print("=" * 72)
    print("TEST 1 -- SEA / LAND BREEZE")
    print("=" * 72)
    print(f"PRIMARY window {ist[common].min():%Y-%m-%d} to "
          f"{ist[common].max():%Y-%m-%d} ({int(common.sum()):,} hours)")
    print(f"SENSITIVITY    {ist.min():%Y-%m-%d} to {ist.max():%Y-%m-%d} "
          f"({len(ist):,} hours, the full record)")
    print(f"units: {feat['cv_group'].nunique()} cv groups from {len(feat)} monitors")
    d = feat["dist_coast_m"]
    print(f"coast distance: {int((d < COASTAL_M).sum())} monitors within "
          f"{COASTAL_M/1000:.0f} km, {int((d > INLAND_M).sum())} beyond "
          f"{INLAND_M/1000:.0f} km, median {d.median()/1000:.1f} km")
    print()

    prof, units = run_1a(mod, wide, feat, hours, common, "PRIMARY window")
    primary = run_1b(mod, wide, feat, ist, hours, common, "PRIMARY window")

    # The pre-registration names the full record as the sensitivity for both.
    # It is where the dry season lives, and with it any land breeze at all.
    print("=" * 72)
    print("SENSITIVITY -- the same two tests on the full record")
    print("Pre-registered as sensitivity, never as primary: monitors that came")
    print("online later average a different slice of the year.")
    print("=" * 72)
    run_1a(mod, wide, feat, hours, everything, "full record")
    full = run_1b(mod, wide, feat, ist, hours, everything, "full record")

    # ------------------------------------------------ 1c Powai
    print("-" * 72)
    print("1c. Powai -- the case the breeze idea does not obviously explain")
    print("-" * 72)
    powai = feat[feat["site"].str.startswith("Powai")]
    if len(powai):
        row = powai.iloc[0]
        sid = powai.index[0]
        prof_p = prof.loc[sid].to_numpy()
        pct = 100 * prof_p / prof_p.mean()
        print(f"  {row['site']}: {row['dist_coast_m']/1000:.1f} km from the "
              f"coast, {row['dist_major_road_m']:.0f} m from a major road")
        print(f"  peak hour (fitted) {units.loc[row['cv_group'], 'peak_hour']:.1f}h, "
              f"raw argmax {int(np.argmax(prof_p))}h, "
              f"raw argmin {int(np.argmin(prof_p))}h")
        print("  hour-of-day, % of its own daily mean (common window):")
        print("    " + " ".join(f"{h:>4d}" for h in range(0, 24, 3)))
        print("    " + " ".join(f"{pct[h]:4.0f}" for h in range(0, 24, 3)))
        for name, res in (("common window", primary), ("full record", full)):
            per_st = res.get("per_station")
            if per_st is not None and len(per_st) and sid in per_st.index:
                print(f"  offshore minus onshore, residualised, {name}: "
                      f"{per_st.loc[sid, 'offshore_minus_onshore_sd']:+.3f} SD "
                      f"({int(per_st.loc[sid, 'n_offshore'])} offshore hours)")
    print()
    print("  One monitor, n = 1. Nothing in 1c is a test.")


if __name__ == "__main__":
    main()
