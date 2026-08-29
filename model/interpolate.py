"""PM2.5 interpolation models and their spatial cross-validation.

Run directly: python model/interpolate.py

Reads  data/processed/pm25_clean.parquet
       data/processed/station_features.parquet
Writes data/processed/cv_results.parquet   (per fold, per model, per station)
       data/processed/feature_bounds.json  (training range, for masking later)

Three models, each given exactly the same information: the readings from every
other station at the hour being predicted.

  idw      inverse distance weighting, the baseline to beat
  kriging  ordinary kriging on hourly-standardised values, one fitted variogram
  trees    gradient-boosted trees on the OSM land-use features plus time

A fourth row, trees+idw, is the tree model handed the IDW estimate as an extra
input. It is not a fourth approach, it is the test of whether land use adds
anything on top of plain distance weighting.

Validation holds out whole cv_group values, never single stations. Two stations
70 m apart are not an independent test of a spatial model: with one in the
training set the other is trivially predictable, and the score would be
measuring that rather than interpolation.
"""

import argparse
import json
import sys
import time
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingRegressor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "processed" / "pm25_clean.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "station_features.parquet"
CV_PATH = ROOT / "data" / "processed" / "cv_results.parquet"
BOUNDS_PATH = ROOT / "data" / "processed" / "feature_bounds.json"

OSM_FEATURES = ["dist_major_road_m", "road_density_500m", "road_density_1km",
                "dist_coast_m", "dist_industrial_m", "building_density_500m"]

# The conventional 1/d^2 is not the best setting here -- it leans too hard on
# the single nearest station, which at these separations is mostly local noise.
# Each fold picks its own power on its training stations, so the baseline is as
# strong as it can be without ever consulting the held-out group.
IDW_POWERS = (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)

# Two stations in different cv_groups are >1 km apart (data/clean.py checks it),
# so this floor never binds in CV; it only guards a grid cell that lands on top
# of a station later.
MIN_DIST_KM = 0.05

VARIOGRAM_BINS = 15
# Readings are UTC; rush hour and the nocturnal inversion are local phenomena.
TZ = "Asia/Kolkata"

TREE_KWARGS = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                   min_samples_leaf=40, l2_regularization=1.0,
                   early_stopping=False, random_state=0)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


def distance_matrix(lat, lon):
    n = len(lat)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = haversine_km(lat[i], lon[i], lat[j], lon[j])
    return d


def load(screen=None):
    """Load the panel. `screen` drops readings above a ceiling, in ug/m3.

    data/clean.py passes anything up to 1000 ug/m3, and a residue of sensor
    artefacts clears that bar: 985.0 turns up as the maximum at five unrelated
    stations, three quarters of the readings over 500 are lone hours, and at
    those hours the rest of the city sits at its usual ~23. They are 0.2% of the
    rows and they dominate every squared error, so the flag exists to show
    whether a result survives them. It is off by default -- screening is a
    cleaning decision, and it belongs in data/clean.py once it is settled.
    """
    if not CLEAN_PATH.exists():
        sys.exit(f"{CLEAN_PATH.relative_to(ROOT)} not found - run data/clean.py first.")
    if not FEATURES_PATH.exists():
        sys.exit(f"{FEATURES_PATH.relative_to(ROOT)} not found - "
                 f"run scripts/02_features.py first.")

    obs = pd.read_parquet(CLEAN_PATH)
    n_screened = 0
    if screen is not None:
        n_screened = int((obs["value"] > screen).sum())
        obs = obs[obs["value"] <= screen]
    wide = obs.pivot_table(index="timestamp", columns="station_id", values="value")
    feat = pd.read_parquet(FEATURES_PATH).set_index("station_id").loc[wide.columns]
    return wide, feat, n_screened


def idw_estimate(values, avail, dist_row, sources, power):
    """IDW at one target for every hour, from the given source stations."""
    w = 1.0 / np.maximum(dist_row[sources], MIN_DIST_KM) ** power
    v = np.nan_to_num(values[:, sources])
    a = avail[:, sources]
    den = a @ w
    num = (v * a) @ w
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def select_idw_power(values, avail, dist, sources, groups):
    """Pick the IDW power on the training stations, predicting each from the rest.

    Nested inside the fold: the held-out group is never consulted, so the tuned
    baseline stays an honest one. The inner split drops each training station's
    whole cv_group too -- tuning against a partner 70 m away would pick the power
    that best exploits a neighbour the real task will not have.
    """
    best, best_rmse = IDW_POWERS[0], np.inf
    inner = [(s, sources[groups[sources] != groups[s]]) for s in sources]
    for power in IDW_POWERS:
        err = []
        for s, src in inner:
            if len(src) == 0:
                continue
            est = idw_estimate(values, avail, dist[s], src, power)
            obs = values[:, s]
            ok = ~np.isnan(est) & ~np.isnan(obs)
            err.append(est[ok] - obs[ok])
        rmse = np.sqrt(np.mean(np.concatenate(err) ** 2))
        if rmse < best_rmse:
            best, best_rmse = power, rmse
    return best


def hourly_level(values, avail, sources):
    """Mean and spread across the source stations, hour by hour.

    Kriging the raw field would spend its variogram on the fact that every
    station rises and falls together; standardising each hour by this strips
    that shared signal out and leaves the spatial pattern to model.
    """
    a = avail[:, sources]
    n = a.sum(1)
    v = np.nan_to_num(values[:, sources])
    mean = np.where(n > 0, v.sum(1) / np.maximum(n, 1), np.nan)
    var = np.where(n > 0, (v ** 2 * a).sum(1) / np.maximum(n, 1) - np.nan_to_num(mean) ** 2, 0.0)
    sd = np.sqrt(np.maximum(var, 0.0))
    # A flat hour carries no spatial information; leaving sd at 1 makes the
    # standardised field exactly 0, so kriging returns the hourly mean.
    return mean, np.where(sd > 1e-6, sd, 1.0), n


def exponential_variogram(h, nugget, sill, rng):
    return nugget + sill * (1.0 - np.exp(-3.0 * h / rng))


def fit_variogram(z, avail, dist, sources):
    """Fit an exponential variogram to the standardised field.

    The panel makes this easy: every station pair has thousands of shared hours,
    so each pair gives one well-determined semivariance instead of the usual
    scatter of single differences.
    """
    pair_d, pair_g = [], []
    for ii, i in enumerate(sources):
        for j in sources[ii + 1:]:
            both = avail[:, i] & avail[:, j]
            if both.sum() < 100:
                continue
            diff = z[both, i] - z[both, j]
            pair_d.append(dist[i, j])
            pair_g.append(0.5 * np.mean(diff ** 2))
    pair_d, pair_g = np.array(pair_d), np.array(pair_g)

    edges = np.linspace(0, pair_d.max() * 1.001, VARIOGRAM_BINS + 1)
    idx = np.digitize(pair_d, edges) - 1
    centres, gammas, counts = [], [], []
    for b in range(VARIOGRAM_BINS):
        m = idx == b
        if m.sum() == 0:
            continue
        centres.append(pair_d[m].mean())
        gammas.append(pair_g[m].mean())
        counts.append(m.sum())
    centres, gammas, counts = map(np.array, (centres, gammas, counts))

    try:
        popt, _ = curve_fit(exponential_variogram, centres, gammas,
                            p0=[0.2, 0.8, 15.0], sigma=1.0 / np.sqrt(counts),
                            bounds=([0.0, 1e-3, 1.0], [3.0, 10.0, 300.0]), maxfev=20000)
    except RuntimeError:
        popt = [0.2, 0.8, 15.0]
    return tuple(popt)


def krige(z, avail, dist, sources, targets, params):
    """Ordinary kriging of the standardised field at each target station.

    Kriging weights depend only on which stations are reporting, not on what
    they report, so the system is solved once per distinct availability pattern
    and reused across every hour that shares it -- ~5k solves a fold instead of
    one per station-hour.
    """
    n_hours = z.shape[0]
    out = np.full((n_hours, len(targets)), np.nan)
    a = avail[:, sources]
    patterns, inverse = np.unique(a, axis=0, return_inverse=True)

    g_targets = np.column_stack(
        [exponential_variogram(dist[sources, t], *params) for t in targets])

    for p, pattern in enumerate(patterns):
        hours = np.flatnonzero(inverse == p)
        k = int(pattern.sum())
        if k == 0:
            continue
        if k < 2:
            out[hours, :] = 0.0
            continue
        sel = np.flatnonzero(pattern)
        src = np.asarray(sources)[sel]

        # Ordinary kriging: unbiasedness is imposed with a Lagrange multiplier,
        # which is the extra row and column of ones.
        gamma = exponential_variogram(dist[np.ix_(src, src)], *params)
        np.fill_diagonal(gamma, 0.0)
        lhs = np.ones((k + 1, k + 1))
        lhs[:k, :k] = gamma
        lhs[k, k] = 0.0
        rhs = np.ones((k + 1, len(targets)))
        rhs[:k, :] = g_targets[sel, :]

        try:
            w = np.linalg.solve(lhs, rhs)[:k, :]
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(lhs, rhs, rcond=None)[0][:k, :]
        out[hours, :] = z[np.ix_(hours, src)] @ w
    return out


def design(values, avail, dist, feat, times, target, sources, power):
    """One design matrix: every observed hour at `target`, seen from `sources`.

    Called with sources excluding the target itself for training rows, so a
    station never helps predict itself and the training rows carry the same
    information a held-out station will.
    """
    est = idw_estimate(values, avail, dist[target], sources, power)
    mean, _, n_src = hourly_level(values, avail, sources)
    obs = values[:, target]

    ok = ~np.isnan(obs) & ~np.isnan(est) & (n_src > 0)
    osm = feat.iloc[target][OSM_FEATURES].to_numpy(dtype=float)
    nearest = np.min(dist[target, sources])

    x = pd.DataFrame(np.repeat(osm[None, :], ok.sum(), axis=0), columns=OSM_FEATURES)
    x["dist_nearest_src_km"] = nearest
    x["hour"] = times.hour[ok]
    x["dayofweek"] = times.dayofweek[ok]
    x["month"] = times.month[ok]
    x["regional_mean"] = mean[ok]
    x["idw_est"] = est[ok]
    # ok travels with the design matrix so the kriging column, which is built
    # separately over all hours, is sliced to exactly the same rows.
    return x, obs[ok], ok


def metrics(pred, truth):
    m = ~np.isnan(pred) & ~np.isnan(truth)
    if m.sum() == 0:
        return np.nan, np.nan, 0
    err = pred[m] - truth[m]
    return (float(np.mean(np.abs(err))),
            float(np.sqrt(np.mean(err ** 2))),
            int(m.sum()))


def cross_validate(wide, feat):
    values = wide.to_numpy(dtype=float)
    avail = ~np.isnan(values)
    times = wide.index.tz_convert(TZ)
    lat, lon = feat["lat"].to_numpy(), feat["lon"].to_numpy()
    dist = distance_matrix(lat, lon)
    groups = feat["cv_group"].to_numpy()
    sites = feat["site"].to_numpy()
    ids = feat.index.to_numpy()

    tree_cols = OSM_FEATURES + ["dist_nearest_src_km", "hour", "dayofweek",
                                "month", "regional_mean"]
    rows = []

    for fold, g in enumerate(sorted(set(groups)), 1):
        targets = np.flatnonzero(groups == g)
        sources = np.flatnonzero(groups != g)
        t0 = time.time()

        power = select_idw_power(values, avail, dist, sources, groups)
        mean, sd, n_src = hourly_level(values, avail, sources)
        z = (values - mean[:, None]) / sd[:, None]
        params = fit_variogram(z, avail, dist, sources)
        z_krige = krige(z, avail, dist, sources, targets, params)

        # Trees train on the source stations only, each one seeing the others.
        train_x, train_y = [], []
        for s in sources:
            x, y, _ = design(values, avail, dist, feat, times, s,
                             np.setdiff1d(sources, s), power)
            train_x.append(x)
            train_y.append(y)
        train_x = pd.concat(train_x, ignore_index=True)
        train_y = np.concatenate(train_y)

        plain = HistGradientBoostingRegressor(**TREE_KWARGS).fit(
            train_x[tree_cols], train_y)
        hybrid = HistGradientBoostingRegressor(**TREE_KWARGS).fit(
            train_x[tree_cols + ["idw_est"]], train_y)

        for k, t in enumerate(targets):
            x, y, ok = design(values, avail, dist, feat, times, t, sources, power)
            preds = {
                # The city-wide average of the reporting stations. Not a model:
                # it is the floor any spatial method has to clear to justify
                # itself, and the variogram says it will be a hard floor.
                "regional_mean": mean[ok],
                "idw": x["idw_est"].to_numpy(),
                "kriging": mean[ok] + sd[ok] * z_krige[ok, k],
                "trees": plain.predict(x[tree_cols]),
                "trees+idw": hybrid.predict(x[tree_cols + ["idw_est"]]),
            }
            for name, p in preds.items():
                mae, rmse, n = metrics(p, y)
                rows.append({"fold": fold, "cv_group": g, "station_id": ids[t],
                             "site": sites[t], "model": name, "n": n,
                             "idw_power": power,
                             "mae": mae, "rmse": rmse,
                             "mean_obs": float(np.mean(y))})

        print(f"  [{fold:>2}/{len(set(groups))}] {g:<14} "
              f"{len(targets)} station(s), idw_p={power:g}, nugget={params[0]:.2f} "
              f"sill={params[1]:.2f} range={params[2]:.1f} km  "
              f"({time.time() - t0:.1f}s)")

    return pd.DataFrame(rows)


def summarise(cv):
    """Pool errors across stations by weighting each station by its hours."""
    out = []
    for name, g in cv.groupby("model"):
        w = g["n"].to_numpy()
        out.append({
            "model": name,
            "MAE": np.average(g["mae"], weights=w),
            "RMSE": np.sqrt(np.average(g["rmse"] ** 2, weights=w)),
            "worst_station_RMSE": g["rmse"].max(),
            "best_station_RMSE": g["rmse"].min(),
            "n_station_hours": int(w.sum()),
        })
    return pd.DataFrame(out).sort_values("RMSE").reset_index(drop=True)


def feature_bounds(feat):
    """The range the models were actually trained on, for masking a surface later.

    Recorded, not enforced: a grid cell in the sea or inside the national park
    sits far outside every one of these, and nothing in the CV score speaks to
    what the models do there.
    """
    b = {}
    for c in OSM_FEATURES:
        v = feat[c].to_numpy(dtype=float)
        b[c] = {"min": float(v.min()), "max": float(v.max()),
                "p01": float(np.percentile(v, 1)), "p99": float(np.percentile(v, 99)),
                "median": float(np.median(v))}
    return {"n_stations": int(len(feat)), "features": b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=float, default=None, metavar="UGM3",
                    help="drop readings above this before fitting (see load())")
    args = ap.parse_args()

    wide, feat, n_screened = load(args.screen)
    print("=" * 96)
    print("SPATIAL CROSS-VALIDATION  (leave one cv_group out)")
    print("=" * 96)
    print(f"{wide.shape[0]:,} hours x {wide.shape[1]} stations, "
          f"{int(wide.notna().sum().sum()):,} readings, "
          f"{feat['cv_group'].nunique()} groups")
    if args.screen is not None:
        print(f"screened out {n_screened:,} readings above {args.screen:g} ug/m3")
    print()

    cv = cross_validate(wide, feat)
    out_path = (CV_PATH if args.screen is None
                else CV_PATH.with_name(f"cv_results_screen{args.screen:g}.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv.to_parquet(out_path, index=False)

    bounds = feature_bounds(feat)
    BOUNDS_PATH.write_text(json.dumps(bounds, indent=2), encoding="utf-8")

    summary = summarise(cv)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)

    print("\n" + "-" * 96)
    print("MODEL COMPARISON  (ug/m3, pooled over every held-out station-hour)")
    print("-" * 96)
    base = summary.loc[summary["model"] == "idw"].iloc[0]
    show = summary.copy()
    show["vs_idw_RMSE"] = (100 * (show["RMSE"] - base["RMSE"]) / base["RMSE"])
    show["vs_idw_MAE"] = (100 * (show["MAE"] - base["MAE"]) / base["MAE"])
    print(show.round(3).to_string(index=False))
    print("\nnegative vs_idw = better than the baseline")

    print("\n" + "-" * 96)
    print("PER HELD-OUT STATION  (RMSE)")
    print("-" * 96)
    key = ["cv_group", "site", "station_id"]
    piv = cv.pivot_table(index=key, columns="model", values="rmse")
    piv["mean_obs"] = cv.groupby(key)["mean_obs"].first()
    print(piv.round(2).sort_values("idw", ascending=False).to_string())

    print("\n" + "-" * 96)
    print("TRAINING RANGE PER FEATURE  (recorded for masking, not yet applied)")
    print("-" * 96)
    print(pd.DataFrame(bounds["features"]).T.round(1).to_string())

    print(f"\nWrote {len(cv):,} rows to {out_path.relative_to(ROOT)}")
    print(f"Wrote {BOUNDS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
