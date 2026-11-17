"""Do the models beat the city-average baseline during extreme episodes?

Pre-registered in reports/preregistration_2026-11b_episodes.md before running.

Across all hours the spatial models barely clear the baseline - IDW is about
0.16 ug/m3 of RMSE ahead of a city-wide average, under 1% of it. The question
here is whether that changes when the air is actually bad, which is when an
estimate would be worth having.

Predictions come from the SAME grouped spatial hold-out as the shipped CV:
35 leave-one-cv_group-out folds, each model scored on monitors it never saw,
with the IDW power and the variogram fitted inside the fold. The episode
subsets are taken afterwards, so nothing about them touches the training.

The unit of analysis is the EPISODE, not the hour. 123 top-1% hours are not 123
independent facts - they arrive in 46 contiguous runs, and one synoptic state
covers the whole city for a day at a time. Every interval below is a cluster
bootstrap over whole episodes.

Three families, two of which are underpowered by construction and say so:

  (a) top 1% and top 5% most polluted hours citywide   46 and 91 episodes
  (b) Diwali 2025 nights                               1 episode, no p-value
  (c) winter inversion spells, Dec-Jan                 6 spells, one winter
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PRED_PATH = ROOT / "data" / "processed" / "cv_predictions.parquet"
TZ = "Asia/Kolkata"

MODELS = ("regional_mean", "idw", "kriging", "trees", "trees+idw")
BASELINE = "regional_mean"

MIN_REPORTING = 10          # monitors needed before an hour is rankable
EPISODE_BRIDGE = pd.Timedelta("6h")
N_BOOT = 2000
SEED = 0

# (b) Lakshmi Puja 2025 fell on 20 October; this covers it and the regional
# one-day ambiguity on both sides, 18:00-02:00 each night.
DIWALI_DAYS = (pd.Timestamp("2025-10-18", tz=TZ), pd.Timestamp("2025-10-22", tz=TZ))
NIGHT_FROM, NIGHT_TO = 18, 2

# (c) primary: 24h rolling citywide mean >= 50 ug/m3 held for >= 48 hours in
# Dec-Jan. Sensitivity at 60, which yields four spells and so cannot reach
# significance at all (see the pre-registration).
WINTER_MONTHS = (12, 1)
WINTER_PRIMARY = (50.0, 48)
WINTER_SENSITIVITY = (60.0, 48)


def load_engine():
    spec = importlib.util.spec_from_file_location(
        "interpolate", ROOT / "model" / "interpolate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cv_predictions(mod, wide, feat):
    """Per station-hour held-out predictions from the grouped spatial CV.

    The shipped cross_validate() aggregates to one score per station, which
    cannot be re-sliced by episode, so the same fold structure is run here and
    the predictions themselves are kept.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    values = wide.to_numpy(dtype=float)
    avail = ~np.isnan(values)
    times = wide.index.tz_convert(mod.TZ)
    dist = mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    groups = feat["cv_group"].to_numpy()
    tree_cols = mod.OSM_FEATURES + ["dist_nearest_src_km", "hour", "dayofweek",
                                    "month", "regional_mean"]

    rows = []
    t0 = time.time()
    for fold, g in enumerate(sorted(set(groups)), 1):
        targets = np.flatnonzero(groups == g)
        sources = np.flatnonzero(groups != g)

        power = mod.select_idw_power(values, avail, dist, sources, groups)
        mean, sd, _ = mod.hourly_level(values, avail, sources)
        z = (values - mean[:, None]) / sd[:, None]
        params = mod.fit_variogram(z, avail, dist, sources)
        z_krige = mod.krige(z, avail, dist, sources, targets, params)

        train_x, train_y = [], []
        for s in sources:
            x, y, _ = mod.design(values, avail, dist, feat, times, s,
                                 np.setdiff1d(sources, s), power)
            train_x.append(x)
            train_y.append(y)
        train_x = pd.concat(train_x, ignore_index=True)
        train_y = np.concatenate(train_y)
        plain = HistGradientBoostingRegressor(**mod.TREE_KWARGS).fit(
            train_x[tree_cols], train_y)
        hybrid = HistGradientBoostingRegressor(**mod.TREE_KWARGS).fit(
            train_x[tree_cols + ["idw_est"]], train_y)

        for k, t in enumerate(targets):
            x, y, ok = mod.design(values, avail, dist, feat, times, t, sources,
                                  power)
            frame = pd.DataFrame({
                "time": times[ok], "station_id": feat.index[t], "cv_group": g,
                "observed": y,
                "regional_mean": mean[ok],
                "idw": x["idw_est"].to_numpy(),
                "kriging": mean[ok] + sd[ok] * z_krige[ok, k],
                "trees": plain.predict(x[tree_cols]),
                "trees+idw": hybrid.predict(x[tree_cols + ["idw_est"]]),
            })
            rows.append(frame)
        print(f"  fold {fold:2d}/{len(set(groups))}  {str(g):22s} "
              f"{time.time() - t0:6.0f}s", flush=True)
    return pd.concat(rows, ignore_index=True)


def citywide(wide):
    """Mean observed PM2.5 across reporting monitors, hours with enough of them."""
    V = wide.to_numpy(dtype=float)
    n = np.isfinite(V).sum(axis=1)
    city = pd.Series(np.nanmean(V, axis=1), index=wide.index.tz_convert(TZ))
    return city[n >= MIN_REPORTING]


def episodes_of(index):
    """Label contiguous runs, merging anything separated by <= the bridge."""
    s = pd.Series(index=pd.DatetimeIndex(sorted(index)), dtype=float)
    gaps = s.index.to_series().diff() > EPISODE_BRIDGE
    return pd.Series(gaps.cumsum().to_numpy(), index=s.index, name="episode")


def rmse(err):
    return float(np.sqrt(np.mean(np.square(err))))


def gap_table(pred, label, episode_of, n_boot=N_BOOT):
    """RMSE per model, the gap to the baseline, and a bootstrap over episodes."""
    d = pred.copy()
    d["episode"] = d["time"].map(episode_of)
    d = d[d["episode"].notna()]
    if d.empty:
        print(f"  {label}: no station-hours; nothing to report")
        return None

    eps = np.sort(d["episode"].unique())
    rng = np.random.default_rng(SEED)
    # Resample whole episodes: within one, a city-wide weather state makes the
    # station-hours anything but independent.
    picks = [rng.choice(eps, size=len(eps), replace=True) for _ in range(n_boot)]
    by_ep = {e: sub for e, sub in d.groupby("episode")}

    out = []
    base_err = d[BASELINE].to_numpy() - d["observed"].to_numpy()
    for m in MODELS:
        err = d[m].to_numpy() - d["observed"].to_numpy()
        gap = rmse(err) - rmse(base_err)
        boot = np.empty(n_boot)
        for i, pick in enumerate(picks):
            sub = pd.concat([by_ep[e] for e in pick], ignore_index=True)
            o = sub["observed"].to_numpy()
            boot[i] = (rmse(sub[m].to_numpy() - o)
                       - rmse(sub[BASELINE].to_numpy() - o))
        lo, hi = np.percentile(boot, [2.5, 97.5])

        per_ep = []
        for e in eps:
            sub = by_ep[e]
            o = sub["observed"].to_numpy()
            per_ep.append(rmse(sub[m].to_numpy() - o)
                          - rmse(sub[BASELINE].to_numpy() - o))
        per_ep = np.asarray(per_ep)
        if m == BASELINE or len(eps) < 6 or np.allclose(per_ep, 0):
            p = np.nan
        else:
            p = stats.wilcoxon(per_ep).pvalue
        out.append({"model": m, "rmse": rmse(err),
                    "mae": float(np.mean(np.abs(err))), "gap": gap,
                    "lo": lo, "hi": hi, "p": p,
                    "n_episodes": len(eps), "n_hours": d["time"].nunique(),
                    "n_station_hours": len(d)})
    t = pd.DataFrame(out)

    first = t.iloc[0]
    print(f"\n  {label}")
    print(f"    {int(first['n_episodes'])} episodes, "
          f"{int(first['n_hours']):,} hours, "
          f"{int(first['n_station_hours']):,} station-hours")
    print(f"    {'model':14s} {'RMSE':>8s} {'MAE':>8s} {'gap':>8s} "
          f"{'95% CI over episodes':>24s} {'Wilcoxon':>9s}")
    for _, r in t.iterrows():
        ci = f"[{r['lo']:+.3f}, {r['hi']:+.3f}]"
        p = "    -" if not np.isfinite(r["p"]) else f"{r['p']:9.4f}"
        flag = "" if r["model"] == BASELINE else (
            "  beats baseline" if r["hi"] < 0 else
            "  WORSE than baseline" if r["lo"] > 0 else "  null")
        print(f"    {r['model']:14s} {r['rmse']:8.3f} {r['mae']:8.3f} "
              f"{r['gap']:+8.3f} {ci:>24s} {p}{flag}")
    return t


def main():
    mod = load_engine()
    wide, feat, _ = mod.load()

    if PRED_PATH.exists():
        pred = pd.read_parquet(PRED_PATH)
        print(f"using cached held-out predictions ({len(pred):,} rows)")
    else:
        print("running the grouped spatial CV to save held-out predictions")
        pred = cv_predictions(mod, wide, feat)
        PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(PRED_PATH)
        print(f"wrote {PRED_PATH.relative_to(ROOT)} ({len(pred):,} rows)")
    pred["time"] = pd.to_datetime(pred["time"], utc=True).dt.tz_convert(TZ)

    city = citywide(wide)
    print("=" * 78)
    print("EPISODES -- do the models beat the city average when it matters?")
    print("=" * 78)
    print(f"citywide hourly mean over {len(city):,} usable hours: "
          f"median {city.median():.0f}, p95 {city.quantile(.95):.0f}, "
          f"max {city.max():.0f} ug/m3")

    all_hours = pd.Series(0, index=city.index)       # one "episode" = everything
    print("\n" + "-" * 78)
    print("REFERENCE -- all hours, the number the rest is compared with")
    print("-" * 78)
    ref = gap_table(pred, "all hours (one block, so the CI is not an "
                          "episode bootstrap)", all_hours, n_boot=200)

    # ------------------------------------------------------- (a) top percentiles
    print("\n" + "-" * 78)
    print("(a) THE MOST POLLUTED HOURS CITYWIDE")
    print("-" * 78)
    tops = {}
    for pct in (1, 5):
        thr = float(np.percentile(city, 100 - pct))
        hours = city[city >= thr]
        ep = episodes_of(hours.index)
        tops[pct] = gap_table(
            pred, f"top {pct}% of hours (>= {thr:.0f} ug/m3)", ep)
        months = pd.Series(hours.index.strftime("%Y-%m")).nunique()
        print(f"    those hours fall in {months} calendar months")

        # Pre-specified sensitivity: month as the cluster, not the run.
        month_cluster = pd.Series(
            pd.factorize(hours.index.strftime("%Y-%m"))[0], index=hours.index)
        gap_table(pred, f"top {pct}%, clustered by MONTH (sensitivity)",
                  month_cluster)

    # ------------------------------------------------------------- (b) Diwali
    print("\n" + "-" * 78)
    print("(b) DIWALI 2025 NIGHTS -- one episode, described, never tested")
    print("-" * 78)
    lo, hi = DIWALI_DAYS
    in_days = (city.index >= lo) & (city.index < hi + pd.Timedelta("1D"))
    hour = city.index.hour
    at_night = (hour >= NIGHT_FROM) | (hour <= NIGHT_TO)
    diwali = city[in_days & at_night]
    print(f"    window {lo:%d %b %Y} to {hi:%d %b %Y}, "
          f"{NIGHT_FROM:02d}:00-{NIGHT_TO:02d}:00 IST")
    print(f"    {len(diwali)} hours, citywide mean {diwali.mean():.0f} "
          f"ug/m3 against {city.mean():.0f} for the record, "
          f"max {diwali.max():.0f}")
    pct_rank = 100 * (city < diwali.mean()).mean()
    print(f"    the window's average hour sits at the {pct_rank:.0f}th "
          f"percentile of all hours")
    one = pd.Series(0, index=diwali.index)
    gap_table(pred, "Diwali nights (n = 1 episode: no p-value by design)",
              one, n_boot=200)
    print("    per night, citywide mean ug/m3:")
    for day, sub in diwali.groupby(diwali.index.normalize()):
        print(f"      {day:%d %b}: {sub.mean():6.0f}  (max {sub.max():.0f}, "
              f"{len(sub)} hours)")

    # ------------------------------------------------------------- (c) winter
    print("\n" + "-" * 78)
    print("(c) WINTER INVERSION SPELLS -- Dec-Jan, one winter in this record")
    print("-" * 78)
    roll = city.rolling("24h").mean()
    for name, (thr, min_hours) in (("PRIMARY", WINTER_PRIMARY),
                                   ("SENSITIVITY", WINTER_SENSITIVITY)):
        winter = roll[roll.index.month.isin(WINTER_MONTHS) & (roll >= thr)]
        if winter.empty:
            print(f"  {name} (>= {thr:.0f} ug/m3 for >= {min_hours}h): "
                  f"no qualifying hours")
            continue
        ep = episodes_of(winter.index)
        sizes = ep.groupby(ep).size()
        keep = sizes[sizes >= min_hours].index
        sel = ep[ep.isin(keep)]
        print(f"\n  {name}: 24h rolling citywide >= {thr:.0f} ug/m3 "
              f"sustained >= {min_hours}h -> {len(keep)} spells")
        if len(keep) == 0:
            continue
        for e, sub in sel.groupby(sel):
            print(f"      spell {int(e)}: {sub.index.min():%d %b %H:%M} to "
                  f"{sub.index.max():%d %b %H:%M}  ({len(sub)} hours)")
        t = gap_table(pred, f"winter spells ({name.lower()})", sel)
        if t is not None and len(keep) < 6:
            print(f"    NOTE: at {len(keep)} episodes a signed-rank test "
                  f"cannot return p < 0.05 at all, so no p is shown.")
    print("\n    All of these spells come from ONE winter (Dec 2025-Jan 2026).")
    print("    They are correlated samples of one season, not independent "
          "winters.")

    print("\n" + "=" * 78)
    print("The comparison that answers the question: the gap during episodes")
    print("against the all-hours gap in the REFERENCE block above.")
    print("=" * 78)
    if ref is not None:
        base = ref.set_index("model")["gap"]
        print("  The all-hours gap is treated as fixed: it rests on 375,872")
        print("  station-hours against a few thousand in an episode, so the")
        print("  uncertainty in the difference is the episode interval shifted.")
        for pct in (1, 5):
            t = tops.get(pct)
            if t is None:
                continue
            print(f"\n  top {pct}%:")
            for _, r in t.iterrows():
                if r["model"] == BASELINE:
                    continue
                b = base[r["model"]]
                verdict = ("bigger advantage in episodes"
                           if r["hi"] - b < 0 else
                           "smaller advantage in episodes"
                           if r["lo"] - b > 0 else "null")
                print(f"    {r['model']:14s} all hours {b:+7.3f}"
                      f"   episodes {r['gap']:+7.3f}   "
                      f"difference {r['gap'] - b:+7.3f}  "
                      f"[{r['lo'] - b:+.3f}, {r['hi'] - b:+.3f}]  {verdict}")


if __name__ == "__main__":
    main()
