"""PM2.5 interpolation models and their spatial cross-validation.

Run directly: python model/interpolate.py

Reads  data/processed/pm25_clean.parquet
       data/processed/station_features.parquet
Writes data/processed/cv_results.parquet   (per fold, per model, per station)
       data/processed/feature_bounds.json  (training range, drives the mask)
       data/processed/model_card.json      (production model + calibrations)

Three models, each given exactly the same information: the readings from every
other station at the hour being predicted.

  idw      inverse distance weighting, the baseline to beat
  kriging  ordinary kriging on hourly-standardised values, one fitted variogram
  trees    gradient-boosted trees on the OSM land-use features plus time

A fourth row, trees+idw, is the tree model handed the IDW estimate as an extra
input. It is not a fourth approach, it is the test of whether land use adds
anything on top of plain distance weighting.

predict_surface() ships IDW alone. Kriging and the trees both lost to it here,
and a model that scores worse is not worth the extra failure surface in an app.
It returns a mask marking cells whose land use falls outside anything in
training, which is most of the bounding box: the box is a rectangle over a
coastal city and much of it is sea, forest and open land.

Validation holds out whole cv_group values, never single stations. Two stations
70 m apart are not an independent test of a spatial model: with one in the
training set the other is trivially predictable, and the score would be
measuring that rather than interpolation.
"""

import argparse
import importlib.util
import json
import sys
import time
from functools import lru_cache
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# scipy.optimize and sklearn are imported inside the two functions that need
# them. Both are cross-validation-only -- the shipped surface is IDW, which
# needs neither -- and importing sklearn eagerly cost the app 2.6s of start-up
# for a model it never runs.

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "processed" / "pm25_clean.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "station_features.parquet"
CV_PATH = ROOT / "data" / "processed" / "cv_results.parquet"
BOUNDS_PATH = ROOT / "data" / "processed" / "feature_bounds.json"
CARD_PATH = ROOT / "data" / "processed" / "model_card.json"
GRID_MASK_PATH = ROOT / "data" / "processed" / "grid_masks.npz"

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

    from scipy.optimize import curve_fit

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
    from sklearn.ensemble import HistGradientBoostingRegressor

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
    held_out = []

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
            held_out.append((preds["idw"], y))
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

    pred = np.concatenate([p for p, _ in held_out])
    obs = np.concatenate([o for _, o in held_out])
    return pd.DataFrame(rows), (pred, obs)


def calibrate_uncertainty(pred, obs, n_bins=20):
    """Per-cell error bars, calibrated on the held-out predictions.

    Error does not grow with distance from the nearest station -- correlation
    -0.02, which is the pure-nugget variogram showing up again: neighbours were
    not helping much, so being far from them costs little. What error does track
    is the level itself, from RMSE ~10 ug/m3 where the city is clean to ~45
    where it is not. So sigma is fitted against the prediction, not the
    geometry, which is the opposite of what a kriging variance would give.
    """
    ok = ~np.isnan(pred) & ~np.isnan(obs)
    pred, obs = pred[ok], obs[ok]
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, len(edges) - 2)
    centres, rmses, counts = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 50:
            continue
        centres.append(pred[m].mean())
        rmses.append(np.sqrt(np.mean((pred[m] - obs[m]) ** 2)))
        counts.append(m.sum())
    centres, rmses, w = np.array(centres), np.array(rmses), np.sqrt(np.array(counts))
    a, b = np.linalg.lstsq(np.column_stack([w, w * centres]), w * rmses, rcond=None)[0]
    return {"form": "max(floor, a + b * prediction)", "a": float(a), "b": float(b),
            # The fitted intercept can land slightly negative, which would hand
            # back a nonsensical sigma on a very clean hour. Floor it at the
            # smallest error any calibration bin actually showed.
            "floor": float(rmses.min()),
            "rmse_overall": float(np.sqrt(np.mean((pred - obs) ** 2))),
            "note": "fitted on held-out predictions; error tracks level, not "
                    "distance to the nearest station (r = -0.02)"}


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



# --------------------------------------------------------------------------
# PRODUCTION SURFACE
#
# IDW only. Kriging and the trees both lost to it under leave-one-group-out CV
# and neither is shipped: a model that scores worse is not worth the extra
# failure surface in an app.
# --------------------------------------------------------------------------

# Cells per axis. 25 over the ~53 km box puts a cell at ~2.1 km, which is about
# where the variogram goes flat. A finer grid would draw structure the data
# cannot support, so `cell_km` comes back in the metadata for the app to show.
GRID_CELLS = 25

# The resolutions the app's "Map detail" control offers. The extrapolation mask
# for each is baked ahead of time by --masks, so the deployed app never loads
# the 41 MB of OSM point clouds -- it only ever needed them to answer a question
# whose answer does not change: which cells look nothing like a monitor site.
GRID_RESOLUTIONS = (15, 25, 40, 60)

# Why a cell is masked, which matters more than that it is. Emptiness is the
# uninteresting half -- sea and forest, correctly refused. Remoteness is the
# real caveat: inhabited ground that is simply farther from a road or an
# industrial estate than any monitor sits, where the model is extrapolating
# into quieter land than it was ever trained on.
MASK_NONE, MASK_EMPTY, MASK_REMOTE, MASK_OTHER = 0, 1, 2, 3

# Falling below the band on these means "less built than anywhere we measured".
DENSITY_FEATURES = ("road_density_500m", "road_density_1km", "building_density_500m")
# Rising above the band on these means "farther from one than anywhere we measured".
REMOTENESS_FEATURES = ("dist_major_road_m", "dist_industrial_m")


class Surface(NamedTuple):
    lats: np.ndarray          # cell-centre latitudes, north-positive (n,)
    lons: np.ndarray          # cell-centre longitudes (n,)
    values: np.ndarray        # predicted ug/m3, (n_lat, n_lon)
    sigma: np.ndarray         # 1-sigma uncertainty, same shape
    mask: np.ndarray          # True where the cell is outside training range
    mask_kind: np.ndarray     # MASK_* per cell, saying which kind
    meta: dict


def grid_axes(bbox, n):
    """Cell-centre latitudes and longitudes for an n x n grid over the bbox.

    Used by both the mask baking and the prediction path; if these two ever
    built the grid differently the mask would silently describe other cells.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    dlon, dlat = (lon_max - lon_min) / n, (lat_max - lat_min) / n
    return (np.linspace(lat_min + dlat / 2, lat_max - dlat / 2, n),
            np.linspace(lon_min + dlon / 2, lon_max - dlon / 2, n))


@lru_cache(maxsize=1)
def grid_masks():
    """The baked masks, or None when they have not been generated yet."""
    if not GRID_MASK_PATH.exists():
        return None
    z = np.load(GRID_MASK_PATH, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return {"bbox": tuple(float(v) for v in z["bbox"]),
            "band": tuple(meta["band"]),
            "reasons": meta["reasons"],
            "kind": {int(k.split("_")[1]): z[k] for k in z.files
                     if k.startswith("kind_")}}


def station_cells(bbox, n):
    """Flat indices of the grid cells that contain a monitor.

    Cells tile the bbox and their centres sit at half-cell offsets, so the
    nearest centre is the containing cell.
    """
    feat = pd.read_parquet(FEATURES_PATH)
    lats, lons = grid_axes(bbox, n)
    i = np.abs(feat["lat"].to_numpy()[:, None] - lats[None, :]).argmin(axis=1)
    j = np.abs(feat["lon"].to_numpy()[:, None] - lons[None, :]).argmin(axis=1)
    return np.unique(i * n + j)


def clear_station_cells(kind, bbox, n):
    """Force cells that contain a monitor in-range. Returns (kind, n_cleared).

    The mask exists to flag places with no comparable training data. A cell with
    a monitor standing in it has ground truth by definition, so refusing to
    estimate there is incoherent whatever the land-use features say.

    It is not a hypothetical. At 2.3 km cells the mask is evaluated at the cell
    centre, which can sit most of a kilometre from the monitor, and a min-max
    band has no tolerance at its own edge: the Bandra Kurla Complex cell was
    excluded for road density of 28.99 against a training maximum of 28.98,
    a margin of 0.035%. The fix is this rule rather than a percentage
    tolerance, which would be an arbitrary number chosen to make one case pass.
    """
    idx = station_cells(bbox, n)
    cleared = int((kind[idx] != MASK_NONE).sum())
    kind[idx] = MASK_NONE
    return kind, cleared


def write_grid_masks(resolutions=GRID_RESOLUTIONS, band=("min", "max")):
    """Bake the extrapolation mask for each resolution the app offers.

    Needs the OSM layers, so it runs here in the pipeline rather than in the
    app. The output is a few kilobytes against 41 MB of source data, because
    the mask is a yes/no per cell and the cells are the same every hour.
    """
    bounds = json.loads(BOUNDS_PATH.read_text(encoding="utf-8"))["features"]
    bbox = osm_features().BBOX
    payload = {"bbox": np.array(bbox, dtype=float)}
    meta = {"band": list(band), "resolutions": list(resolutions), "reasons": {}}
    for n in resolutions:
        lats, lons = grid_axes(bbox, n)
        glon, glat = np.meshgrid(lons, lats)
        _, kind, reasons = extrapolation_mask(glat.ravel(), glon.ravel(), bounds, band)
        kind, cleared = clear_station_cells(kind, bbox, n)
        payload[f"kind_{n}"] = kind.reshape(n, n).astype(np.int8)
        # The per-feature counts still describe why cells fell outside the band,
        # before the monitor override; the aggregates the app displays are
        # recounted from the mask that actually ships.
        counts = {k: int(v) for k, v in reasons.items()}
        counts["_empty"] = int((kind == MASK_EMPTY).sum())
        counts["_remote"] = int((kind == MASK_REMOTE).sum())
        counts["_other"] = int((kind == MASK_OTHER).sum())
        counts["_station_cells_cleared"] = cleared
        meta["reasons"][str(n)] = counts
    payload["meta"] = np.array(json.dumps(meta))
    GRID_MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GRID_MASK_PATH, **payload)
    grid_masks.cache_clear()
    return meta


@lru_cache(maxsize=1)
def osm_features():
    """Load scripts/02_features.py by path.

    Its name starts with a digit, so it cannot be imported normally, and
    renaming it would break the numbered-script convention the repo already
    uses. This is the one place that awkwardness shows up.
    """
    spec = importlib.util.spec_from_file_location(
        "osm_features", ROOT / "scripts" / "02_features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def production():
    """Station geometry, the tuned power, and the calibrations, loaded once."""
    if not CARD_PATH.exists():
        sys.exit(f"{CARD_PATH.relative_to(ROOT)} not found - "
                 f"run model/interpolate.py first.")
    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    bounds = json.loads(BOUNDS_PATH.read_text(encoding="utf-8"))["features"]
    wide, feat, _ = load()
    return {"wide": wide, "feat": feat, "card": card, "bounds": bounds,
            "lat": feat["lat"].to_numpy(), "lon": feat["lon"].to_numpy()}


def haversine_grid_km(lat, lon, slat, slon):
    """Distances from every point to every station, (n_points, n_stations)."""
    lat, lon = np.radians(lat)[:, None], np.radians(lon)[:, None]
    slat, slon = np.radians(slat)[None, :], np.radians(slon)[None, :]
    a = (np.sin((slat - lat) / 2) ** 2
         + np.cos(lat) * np.cos(slat) * np.sin((slon - lon) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def extrapolation_mask(lats, lons, bounds, band=("min", "max")):
    """True where any OSM feature falls outside the training range.

    The station network only ever saw built-up Mumbai. A cell in the Arabian Sea
    or inside the national park has no road, no buildings and no analogue in
    training, and nothing in the CV score says what the model does there.

    The band is min-max, so it is guaranteed to contain every station the model
    trained on. The tighter p01-p99 is available via band=("p01", "p99"), but at
    39 stations p01 sits just above the minimum and 9 stations fall outside
    their own band -- the mask would shade locations where we hold ground truth.
    It costs 1.4 points of coverage to avoid that, which is worth it here;
    p01-p99 would earn its keep with a few hundred stations, not thirty-nine.
    """
    lo_key, hi_key = band
    feats = osm_features().features_for(lats, lons)
    n = len(feats)
    mask = np.zeros(n, dtype=bool)
    below, above = np.zeros(n, bool), np.zeros(n, bool)
    empty, remote, reasons = np.zeros(n, bool), np.zeros(n, bool), {}
    for name, b in bounds.items():
        v = feats[name].to_numpy()
        lo, hi = v < b[lo_key], v > b[hi_key]
        reasons[name] = int((lo | hi).sum())
        below |= lo
        above |= hi
        mask |= lo | hi
        if name in DENSITY_FEATURES:
            empty |= lo
        if name in REMOTENESS_FEATURES:
            remote |= hi

    # Emptiness wins where a cell is both: open water far from a road is
    # refused because there is nothing there, not because of the distance.
    kind = np.where(empty, MASK_EMPTY,
                    np.where(remote, MASK_REMOTE,
                             np.where(mask, MASK_OTHER, MASK_NONE))).astype(np.int8)
    reasons["_below_band"] = int(below.sum())
    reasons["_above_band"] = int(above.sum())
    reasons["_empty"] = int((kind == MASK_EMPTY).sum())
    reasons["_remote"] = int((kind == MASK_REMOTE).sum())
    reasons["_other"] = int((kind == MASK_OTHER).sum())
    return mask, kind, reasons


def predict_surface(timestamp, pollutant="pm25", grid_resolution=GRID_CELLS,
                    band=("min", "max"), readings=None):
    """Interpolated PM2.5 over the Mumbai bbox for one hour.

    Returns a Surface: predictions, per-cell sigma, and a mask marking cells
    whose land use falls outside anything the model was trained on. The mask is
    advisory -- values are still returned for masked cells so the app can shade
    rather than hole-punch them.

    `readings` overrides the stored panel with a station_id -> ug/m3 Series, so
    the same interpolation can run on a live feed instead of on history. The
    timestamp is then only a label for what is being drawn.
    """
    if pollutant != "pm25":
        raise ValueError(
            f"only pm25 was fetched in Phase 1; got {pollutant!r}. "
            f"data/fetch.py would need to pull the other pollutants first.")

    p = production()
    ts = pd.Timestamp(timestamp)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    ts = ts.floor("h")

    n = int(grid_resolution)
    baked = grid_masks()
    use_baked = (baked is not None and tuple(band) == baked["band"]
                 and n in baked["kind"])

    # The bbox comes from the baked file when there is one, so the deployed app
    # does not import the OSM module at all -- not even for a constant.
    bbox = baked["bbox"] if baked is not None else osm_features().BBOX
    lon_min, lat_min, lon_max, lat_max = bbox
    dlon, dlat = (lon_max - lon_min) / n, (lat_max - lat_min) / n
    lats, lons = grid_axes(bbox, n)
    glon, glat = np.meshgrid(lons, lats)
    flat_lat, flat_lon = glat.ravel(), glon.ravel()

    if use_baked:
        kind = baked["kind"][n].ravel()
        mask = kind != MASK_NONE
        reasons = dict(baked["reasons"][str(n)])
    else:
        # Falls through to the OSM layers: a resolution or band nobody baked.
        mask, kind, reasons = extrapolation_mask(flat_lat, flat_lon,
                                                 p["bounds"], band)
        kind, _ = clear_station_cells(kind, bbox, n)
        mask = kind != MASK_NONE

    if readings is not None:
        row = (pd.Series(readings, dtype=float)
                 .reindex(p["wide"].columns).to_numpy(dtype=float))
    elif ts in p["wide"].index:
        row = p["wide"].loc[ts].to_numpy(dtype=float)
    else:
        row = None

    if row is None or not np.isfinite(row).any():
        blank = np.full((n, n), np.nan)
        return Surface(lats, lons, blank, blank, mask.reshape(n, n),
                       kind.reshape(n, n),
                       {"timestamp": ts, "n_stations": 0, "cell_km": None,
                        "masked_fraction": float(mask.mean()),
                        "mask_reasons": reasons,
                        "note": "no station reported this hour"})

    live = ~np.isnan(row)
    power = p["card"]["idw_power"]
    d = haversine_grid_km(flat_lat, flat_lon, p["lat"][live], p["lon"][live])
    w = 1.0 / np.maximum(d, MIN_DIST_KM) ** power
    values = (w @ row[live]) / w.sum(axis=1)

    u = p["card"]["uncertainty"]
    sigma = np.maximum(u["a"] + u["b"] * values, u["floor"])

    cell_km = float(np.mean([dlat * 111.0,
                             dlon * 111.0 * np.cos(np.radians((lat_min + lat_max) / 2))]))
    return Surface(
        lats, lons, values.reshape(n, n), sigma.reshape(n, n), mask.reshape(n, n),
        kind.reshape(n, n),
        {"timestamp": ts, "n_stations": int(live.sum()), "idw_power": power,
         "cell_km": round(cell_km, 2), "masked_fraction": float(mask.mean()),
         "mask_reasons": reasons, "band": list(band),
         "cv_rmse": p["card"]["cv"]["rmse"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=float, default=None, metavar="UGM3",
                    help="drop readings above this before fitting (see load())")
    ap.add_argument("--masks", action="store_true",
                    help="only bake the grid masks, skipping cross-validation")
    args = ap.parse_args()

    if args.masks:
        meta = write_grid_masks()
        size = GRID_MASK_PATH.stat().st_size
        print(f"Baked masks for {meta['resolutions']} at band "
              f"{'-'.join(meta['band'])} -> {GRID_MASK_PATH.relative_to(ROOT)} "
              f"({size / 1024:.1f} KB)")
        for n in meta["resolutions"]:
            r = meta["reasons"][str(n)]
            masked = r["_empty"] + r["_remote"] + r["_other"]
            pct = 100 * masked / (n * n)
            print(f"  {n:>3}x{n:<3} {pct:>5.1f}% masked  "
                  f"(empty {r['_empty']}, remote {r['_remote']}, "
                  f"other {r['_other']})")
        return

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

    cv, (idw_pred, idw_obs) = cross_validate(wide, feat)
    out_path = (CV_PATH if args.screen is None
                else CV_PATH.with_name(f"cv_results_screen{args.screen:g}.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv.to_parquet(out_path, index=False)

    bounds = feature_bounds(feat)
    BOUNDS_PATH.write_text(json.dumps(bounds, indent=2), encoding="utf-8")

    summary = summarise(cv)

    # The production model refits the power over every station, using the same
    # leave-one-group-out rule the folds used, so no group ever tunes on itself.
    values, avail = wide.to_numpy(dtype=float), wide.notna().to_numpy()
    dist = distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    prod_power = select_idw_power(values, avail, dist,
                                  np.arange(len(feat)), feat["cv_group"].to_numpy())
    card = {
        "model": "idw",
        "why": "beat kriging and gradient-boosted trees under leave-one-group-out CV",
        "idw_power": float(prod_power),
        "cv": {"mae": float(summary.loc[summary.model == "idw", "MAE"].iloc[0]),
               "rmse": float(summary.loc[summary.model == "idw", "RMSE"].iloc[0]),
               "n_folds": int(cv["fold"].nunique()),
               "n_stations": int(cv["station_id"].nunique()),
               "n_station_hours": int(summary.loc[summary.model == "idw",
                                                  "n_station_hours"].iloc[0])},
        "comparison": summary.set_index("model")[["MAE", "RMSE"]].round(3).to_dict("index"),
        "uncertainty": calibrate_uncertainty(idw_pred, idw_obs),
    }
    CARD_PATH.write_text(json.dumps(card, indent=2), encoding="utf-8")
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

    u = card["uncertainty"]
    print("\n" + "-" * 96)
    print("PRODUCTION SURFACE  (IDW only - kriging and trees lost, so they are not shipped)")
    print("-" * 96)
    print(f"idw power       {prod_power:g}  (refitted over all {len(feat)} stations)")
    print(f"uncertainty     sigma = {u['a']:.2f} + {u['b']:.3f} x prediction  "
          f"(pooled RMSE {u['rmse_overall']:.2f})")
    print(f"                error tracks the level, not the distance to a station")

    production.cache_clear()
    ts = wide.notna().sum(axis=1).idxmax()
    for res in (15, 25, 40, 60):
        surf = predict_surface(ts, grid_resolution=res)
        m = surf.meta
        print(f"  grid {res:>3}x{res:<3} = {res * res:>5,} cells at ~{m['cell_km']:.1f} km   "
              f"masked {100 * m['masked_fraction']:>5.1f}%   "
              f"values {np.nanmin(surf.values):.1f}-{np.nanmax(surf.values):.1f} ug/m3")

    surf = predict_surface(ts)
    print(f"\nAt {surf.meta['timestamp']} ({surf.meta['n_stations']} stations reporting), "
          f"{100 * surf.meta['masked_fraction']:.1f}% of the {GRID_CELLS}x{GRID_CELLS} grid "
          f"is outside the training range.")
    r = surf.meta["mask_reasons"]
    cells = surf.values.size
    print(f"  {100 * r['_below_band'] / cells:>5.1f}%  below the band "
          f"(sea, forest, open land the network never sampled)")
    print(f"  {100 * r['_above_band'] / cells:>5.1f}%  above the band "
          f"(denser than any station is placed in)")
    print("Cells masked by each feature (a cell can fail several):")
    for name, cnt in sorted(r.items(), key=lambda kv: -kv[1]):
        if cnt and not name.startswith("_"):
            print(f"  {100 * cnt / cells:>5.1f}%  {name}")

    # A band that excludes the training stations would hide ground truth.
    print("\nSelf-consistency of the band (stations outside their own bounds):")
    for lo, hi in (("p01", "p99"), ("min", "max")):
        own = np.zeros(len(feat), bool)
        for name, b in bounds["features"].items():
            v = feat[name].to_numpy()
            own |= (v < b[lo]) | (v > b[hi])
        alt = predict_surface(ts, band=(lo, hi))
        print(f"  {lo}-{hi:<4} masks {own.sum():>2}/{len(feat)} training stations, "
              f"{100 * alt.meta['masked_fraction']:>5.1f}% of the grid")

    print(f"\nWrote {len(cv):,} rows to {out_path.relative_to(ROOT)}")
    print(f"Wrote {CARD_PATH.relative_to(ROOT)}")
    print(f"Wrote {BOUNDS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
