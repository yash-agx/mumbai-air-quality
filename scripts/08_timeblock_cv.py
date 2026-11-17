"""Test 4 of reports/preregistration_2026-11.md: time-block cross-validation.

The shipped CV holds out whole cv_groups, so a model is scored on a monitor it
never saw. It is still scored on hours it saw plenty of, through every other
monitor. This asks whether the gap between the models and the city-average
baseline changes when whole months are held out as well.

Design, fixed in the pre-registration before running:

  - cells are (cv_group, month). 4 months drawn with seed 0 from the record,
    crossed with all 35 groups.
  - the comparison is PAIRED within cell: both schemes are scored on exactly
    the same held-out rows, and differ only in whether the training data also
    excludes that month.
      spatial only   train on other groups, all months  (what ships)
      spatial + time train on other groups, minus that month
  - metric is RMSE per cell; the gap is model RMSE minus regional_mean RMSE.
  - the test is Wilcoxon signed-rank on the change in gap, with the 35 groups
    as units. Months inside a group are not independent, so a group's cells
    are averaged to one number before testing.

Stated in the pre-registration so it is not mistaken for a finding: idw,
kriging and regional_mean predict from CONCURRENT readings at other stations,
so holding out a month cannot touch them except through fitted hyperparameters
(the IDW power, the variogram). The tree models are the ones with something to
leak - and `month` is one of their input features, so under this scheme they
are asked about a month value they never saw in training.
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "processed" / "timeblock_cv.parquet"

N_MONTHS = 4          # pre-registered; fallback was 3 if too slow
SEED = 0
MODELS = ("regional_mean", "idw", "kriging", "trees", "trees+idw")


def load_engine():
    spec = importlib.util.spec_from_file_location(
        "interpolate", ROOT / "model" / "interpolate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_scheme(mod, values, avail, dist, feat, times, sources, groups,
               tree_cols, train_rows):
    """Everything a fold needs that depends on the training data.

    `train_rows` is the hour mask the scheme is allowed to learn from. Hiding
    hours is done by marking them unavailable, which is what both the power
    search and the variogram read, so neither can see the held-out month.
    """
    v_tr = values.copy()
    a_tr = avail.copy()
    v_tr[~train_rows] = np.nan
    a_tr[~train_rows] = False

    power = mod.select_idw_power(v_tr, a_tr, dist, sources, groups)
    mean_tr, sd_tr, _ = mod.hourly_level(v_tr, a_tr, sources)
    z_tr = (v_tr - mean_tr[:, None]) / sd_tr[:, None]
    params = mod.fit_variogram(z_tr, a_tr, dist, sources)

    train_x, train_y = [], []
    for s in sources:
        x, y, ok = mod.design(values, avail, dist, feat, times, s,
                              np.setdiff1d(sources, s), power)
        keep = train_rows[ok]
        if keep.any():
            train_x.append(x[keep])
            train_y.append(y[keep])
    train_x = pd.concat(train_x, ignore_index=True)
    train_y = np.concatenate(train_y)

    from sklearn.ensemble import HistGradientBoostingRegressor
    plain = HistGradientBoostingRegressor(**mod.TREE_KWARGS).fit(
        train_x[tree_cols], train_y)
    hybrid = HistGradientBoostingRegressor(**mod.TREE_KWARGS).fit(
        train_x[tree_cols + ["idw_est"]], train_y)
    return {"power": power, "params": params, "plain": plain,
            "hybrid": hybrid, "n_train_rows": int(len(train_y))}


def score_cell(mod, fit, values, avail, dist, feat, times, targets, sources,
               tree_cols, test_rows):
    """RMSE per model on one (group, month) cell, for one fitted scheme."""
    # Prediction-time inputs are concurrent readings, never masked: at the hour
    # being predicted the other monitors are reporting in real life too.
    mean, sd, _ = mod.hourly_level(values, avail, sources)
    z = (values - mean[:, None]) / sd[:, None]
    z_krige = mod.krige(z, avail, dist, sources, targets, fit["params"])

    out = []
    for k, t in enumerate(targets):
        x, y, ok = mod.design(values, avail, dist, feat, times, t, sources,
                              fit["power"])
        sel = test_rows[ok]
        if sel.sum() < 24:          # less than a day of held-out readings
            continue
        preds = {
            "regional_mean": mean[ok],
            "idw": x["idw_est"].to_numpy(),
            "kriging": mean[ok] + sd[ok] * z_krige[ok, k],
            "trees": fit["plain"].predict(x[tree_cols]),
            "trees+idw": fit["hybrid"].predict(x[tree_cols + ["idw_est"]]),
        }
        for name, p in preds.items():
            _, rmse, n = mod.metrics(p[sel], y[sel])
            out.append({"station_id": feat.index[t], "model": name,
                        "rmse": rmse, "n": n})
    return out


def main():
    mod = load_engine()
    wide, feat, _ = mod.load()
    values = wide.to_numpy(dtype=float)
    avail = ~np.isnan(values)
    times = wide.index.tz_convert(mod.TZ)
    dist = mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    groups = feat["cv_group"].to_numpy()
    tree_cols = mod.OSM_FEATURES + ["dist_nearest_src_km", "hour", "dayofweek",
                                    "month", "regional_mean"]

    month_key = times.strftime("%Y-%m").to_numpy()
    all_months = np.array(sorted(set(month_key)))
    rng = np.random.default_rng(SEED)
    test_months = np.sort(rng.choice(all_months, size=N_MONTHS, replace=False))

    print("=" * 72)
    print("TEST 4 -- TIME-BLOCK CROSS-VALIDATION")
    print("=" * 72)
    print(f"record: {len(all_months)} months, {wide.shape[0]:,} hours, "
          f"{wide.shape[1]} monitors in {len(set(groups))} cv groups")
    print(f"held-out months (seed {SEED}): {', '.join(test_months)}")
    print(f"cells: {len(set(groups))} groups x {len(test_months)} months")
    print()

    rows = []
    t0 = time.time()
    for fold, g in enumerate(sorted(set(groups)), 1):
        targets = np.flatnonzero(groups == g)
        sources = np.flatnonzero(groups != g)

        # Spatial-only is fitted once per group: it does not depend on which
        # month is being scored.
        spatial = fit_scheme(mod, values, avail, dist, feat, times, sources,
                             groups, tree_cols, np.ones(len(times), bool))

        for m in test_months:
            test_rows = month_key == m
            if not (avail[np.ix_(test_rows, targets)]).any():
                continue
            both = fit_scheme(mod, values, avail, dist, feat, times, sources,
                              groups, tree_cols, ~test_rows)
            for scheme, fit in (("spatial_only", spatial),
                                ("spatial_time", both)):
                for r in score_cell(mod, fit, values, avail, dist, feat, times,
                                    targets, sources, tree_cols, test_rows):
                    r.update({"cv_group": g, "month": m, "scheme": scheme,
                              "idw_power": fit["power"],
                              "n_train_rows": fit["n_train_rows"]})
                    rows.append(r)
        print(f"  fold {fold:2d}/{len(set(groups))}  {str(g):22s} "
              f"{time.time() - t0:6.0f}s elapsed", flush=True)

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)
    print(f"\nwrote {OUT_PATH.relative_to(ROOT)} ({len(df):,} rows)")
    report(df)


def report(df):
    # One RMSE per (group, month, scheme, model): average the stations inside
    # a group, which is the unit the spatial CV holds out.
    cell = (df.groupby(["cv_group", "month", "scheme", "model"])["rmse"]
              .mean().reset_index())
    wide_m = cell.pivot_table(index=["cv_group", "month", "scheme"],
                              columns="model", values="rmse").reset_index()

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print("RMSE (ug/m3), mean over cells, and the gap to the city-average "
          "baseline.")
    print("A negative gap means the model beats the baseline.\n")
    print(f"  {'model':14s} {'spatial only':>14s} {'spatial+time':>14s} "
          f"{'change':>9s}")
    for m in MODELS:
        a = wide_m[wide_m["scheme"] == "spatial_only"][m].mean()
        b = wide_m[wide_m["scheme"] == "spatial_time"][m].mean()
        print(f"  {m:14s} {a:14.3f} {b:14.3f} {b - a:+9.3f}")

    print("\n  gap to regional_mean (model RMSE - baseline RMSE):")
    print(f"  {'model':14s} {'spatial only':>14s} {'spatial+time':>14s} "
          f"{'change':>9s} {'Wilcoxon p':>12s}")
    for m in MODELS:
        if m == "regional_mean":
            continue
        per_group = []
        for g, sub in wide_m.groupby("cv_group"):
            a = sub[sub["scheme"] == "spatial_only"]
            b = sub[sub["scheme"] == "spatial_time"]
            if a.empty or b.empty:
                continue
            gap_a = (a[m] - a["regional_mean"]).mean()
            gap_b = (b[m] - b["regional_mean"]).mean()
            per_group.append((gap_a, gap_b))
        arr = np.array(per_group)
        diff = arr[:, 1] - arr[:, 0]
        if np.allclose(diff, 0):
            p = 1.0
        else:
            p = stats.wilcoxon(diff).pvalue
        print(f"  {m:14s} {arr[:, 0].mean():14.3f} {arr[:, 1].mean():14.3f} "
              f"{diff.mean():+9.3f} {p:12.4f}   (n={len(arr)} groups)")

    print("\n  A positive change means the model lost ground against the "
          "baseline\n  once it could no longer train on the month it was "
          "scored in.")


if __name__ == "__main__":
    main()
