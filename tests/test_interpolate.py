"""Tests for model/interpolate.py -- the interpolation models and their CV.

Everything runs on the in-memory panel from the `panel` fixture. The two paths
that would otherwise read from data/processed -- production() and the baked grid
masks -- are replaced per test, and the OSM module behind extrapolation_mask()
is stubbed, so no parquet, npz or point cloud on disk is involved.
"""

import json

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def test_haversine_km_zero_for_same_point(interp_mod):
    assert interp_mod.haversine_km(19.0, 72.8, 19.0, 72.8) == 0.0


def test_haversine_km_matches_a_degree_of_latitude(interp_mod):
    assert interp_mod.haversine_km(19.0, 72.8, 20.0, 72.8) == pytest.approx(111.2, rel=1e-3)


def test_distance_matrix_is_symmetric_with_a_zero_diagonal(interp_mod):
    lat = np.array([19.00, 19.05, 19.20])
    lon = np.array([72.82, 72.88, 72.90])
    d = interp_mod.distance_matrix(lat, lon)
    assert d.shape == (3, 3)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)
    assert d[0, 2] > d[0, 1]


def test_haversine_grid_km_shape_and_agreement(interp_mod):
    """The grid-wide vectorised distance has to agree with the scalar one."""
    lats = np.array([19.0, 19.1])
    lons = np.array([72.8, 72.9])
    slat = np.array([19.05, 19.20])
    slon = np.array([72.85, 72.90])
    d = interp_mod.haversine_grid_km(lats, lons, slat, slon)
    assert d.shape == (2, 2)
    assert d[0, 0] == pytest.approx(
        interp_mod.haversine_km(19.0, 72.8, 19.05, 72.85), rel=1e-9)


# --------------------------------------------------------------------------
# IDW
# --------------------------------------------------------------------------

def test_idw_estimate_returns_the_only_source(interp_mod):
    values = np.array([[10.0, 50.0]])
    avail = ~np.isnan(values)
    dist_row = np.array([0.0, 3.0])
    est = interp_mod.idw_estimate(values, avail, dist_row, np.array([1]), 2.0)
    assert est == pytest.approx([50.0])


def test_idw_estimate_averages_two_equidistant_sources(interp_mod):
    values = np.array([[np.nan, 20.0, 40.0]])
    avail = ~np.isnan(values)
    dist_row = np.array([0.0, 5.0, 5.0])
    est = interp_mod.idw_estimate(values, avail, dist_row, np.array([1, 2]), 2.0)
    assert est == pytest.approx([30.0])


def test_idw_estimate_leans_on_the_nearer_source_as_power_rises(interp_mod):
    values = np.array([[np.nan, 20.0, 40.0]])
    avail = ~np.isnan(values)
    dist_row = np.array([0.0, 1.0, 4.0])
    weak = interp_mod.idw_estimate(values, avail, dist_row, np.array([1, 2]), 0.1)[0]
    strong = interp_mod.idw_estimate(values, avail, dist_row, np.array([1, 2]), 3.0)[0]
    assert 20.0 < strong < weak < 40.0


def test_idw_estimate_with_power_zero_is_the_plain_mean(interp_mod):
    values = np.array([[np.nan, 20.0, 40.0, 60.0]])
    avail = ~np.isnan(values)
    dist_row = np.array([0.0, 1.0, 10.0, 100.0])
    est = interp_mod.idw_estimate(values, avail, dist_row, np.array([1, 2, 3]), 0.0)
    assert est == pytest.approx([40.0])


def test_idw_estimate_skips_hours_with_nothing_reporting(interp_mod):
    values = np.array([[np.nan, 20.0], [np.nan, np.nan]])
    avail = ~np.isnan(values)
    dist_row = np.array([0.0, 2.0])
    est = interp_mod.idw_estimate(values, avail, dist_row, np.array([1]), 2.0)
    assert est[0] == pytest.approx(20.0)
    assert np.isnan(est[1])


def test_idw_estimate_floors_the_distance(interp_mod):
    """A grid cell on top of a station must not divide by zero."""
    values = np.array([[np.nan, 20.0, 40.0]])
    avail = ~np.isnan(values)
    dist_row = np.array([0.0, 0.0, 1.0])
    est = interp_mod.idw_estimate(values, avail, dist_row, np.array([1, 2]), 2.0)
    assert np.isfinite(est[0])
    # At MIN_DIST_KM against 1 km the coincident station dominates but does not
    # take the whole weight.
    assert 20.0 < est[0] < 20.1


def test_idw_estimate_stays_within_the_range_of_its_sources(interp_mod, panel):
    """IDW is a convex combination, so it can never leave the data's range."""
    wide, feat = panel
    values = wide.to_numpy(dtype=float)
    avail = ~np.isnan(values)
    dist = interp_mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    sources = np.array([1, 2, 3])
    est = interp_mod.idw_estimate(values, avail, dist[0], sources, 1.0)
    lo = np.nanmin(values[:, sources], axis=1)
    hi = np.nanmax(values[:, sources], axis=1)
    assert np.all(est >= lo - 1e-9)
    assert np.all(est <= hi + 1e-9)


def test_select_idw_power_returns_a_candidate_power(interp_mod, panel):
    wide, feat = panel
    values = wide.to_numpy(dtype=float)
    avail = ~np.isnan(values)
    dist = interp_mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    groups = feat["cv_group"].to_numpy()
    power = interp_mod.select_idw_power(values, avail, dist,
                                       np.array([0, 1, 2]), groups)
    assert power in interp_mod.IDW_POWERS


def test_select_idw_power_is_indifferent_on_a_flat_field(interp_mod):
    """Every station reading the same value makes every power exact, so the
    first candidate stands -- nothing is being chosen on noise."""
    values = np.full((50, 4), 30.0)
    avail = np.ones_like(values, dtype=bool)
    dist = interp_mod.distance_matrix(np.array([19.0, 19.1, 19.2, 19.3]),
                                      np.array([72.8, 72.85, 72.9, 72.95]))
    groups = np.array(["a", "b", "c", "d"])
    power = interp_mod.select_idw_power(values, avail, dist,
                                       np.array([0, 1, 2, 3]), groups)
    assert power == interp_mod.IDW_POWERS[0]


def test_select_idw_power_never_sees_the_held_out_group(interp_mod, panel):
    """Corrupting the held-out station must not change the chosen power."""
    wide, feat = panel
    values = wide.to_numpy(dtype=float)
    avail = ~np.isnan(values)
    dist = interp_mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    groups = feat["cv_group"].to_numpy()
    sources = np.array([0, 1, 2])

    before = interp_mod.select_idw_power(values, avail, dist, sources, groups)
    corrupted = values.copy()
    corrupted[:, 3] = 900.0
    after = interp_mod.select_idw_power(corrupted, avail, dist, sources, groups)
    assert before == after


# --------------------------------------------------------------------------
# Hourly standardisation
# --------------------------------------------------------------------------

def test_hourly_level_mean_sd_and_count(interp_mod):
    values = np.array([[10.0, 20.0, 30.0]])
    avail = ~np.isnan(values)
    mean, sd, n = interp_mod.hourly_level(values, avail, np.array([0, 1, 2]))
    assert mean == pytest.approx([20.0])
    assert sd == pytest.approx([np.sqrt(200.0 / 3)])
    assert list(n) == [3]


def test_hourly_level_ignores_missing_stations(interp_mod):
    values = np.array([[10.0, np.nan, 30.0]])
    avail = ~np.isnan(values)
    mean, _, n = interp_mod.hourly_level(values, avail, np.array([0, 1, 2]))
    assert mean == pytest.approx([20.0])
    assert list(n) == [2]


def test_hourly_level_holds_sd_at_one_on_a_flat_hour(interp_mod):
    """A flat hour carries no spatial information, so the standardised field is
    exactly 0 and kriging falls back to the hourly mean."""
    values = np.array([[25.0, 25.0, 25.0]])
    avail = ~np.isnan(values)
    mean, sd, _ = interp_mod.hourly_level(values, avail, np.array([0, 1, 2]))
    assert sd == pytest.approx([1.0])
    assert ((values - mean[:, None]) / sd[:, None]) == pytest.approx(np.zeros((1, 3)))


def test_hourly_level_is_nan_when_nothing_reports(interp_mod):
    values = np.array([[np.nan, np.nan]])
    avail = ~np.isnan(values)
    mean, sd, n = interp_mod.hourly_level(values, avail, np.array([0, 1]))
    assert np.isnan(mean[0])
    assert sd[0] == 1.0
    assert n[0] == 0


# --------------------------------------------------------------------------
# Variogram and kriging
# --------------------------------------------------------------------------

def test_exponential_variogram_starts_at_the_nugget(interp_mod):
    assert interp_mod.exponential_variogram(0.0, 0.2, 0.8, 15.0) == pytest.approx(0.2)


def test_exponential_variogram_approaches_nugget_plus_sill(interp_mod):
    far = interp_mod.exponential_variogram(1e6, 0.2, 0.8, 15.0)
    assert far == pytest.approx(1.0, rel=1e-6)


def test_exponential_variogram_is_monotone_in_distance(interp_mod):
    h = np.linspace(0, 100, 50)
    g = interp_mod.exponential_variogram(h, 0.2, 0.8, 15.0)
    assert np.all(np.diff(g) > 0)


def test_exponential_variogram_reaches_95_percent_at_the_range(interp_mod):
    """The 3h/range convention: one range is where the curve is ~95% of sill."""
    g = interp_mod.exponential_variogram(15.0, 0.0, 1.0, 15.0)
    assert g == pytest.approx(1.0 - np.exp(-3.0), rel=1e-9)


def test_fit_variogram_returns_parameters_inside_the_fit_bounds(interp_mod, panel):
    wide, feat = panel
    values = wide.to_numpy(dtype=float)
    avail = ~np.isnan(values)
    dist = interp_mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    mean, sd, _ = interp_mod.hourly_level(values, avail, np.array([0, 1, 2, 3]))
    z = (values - mean[:, None]) / sd[:, None]

    nugget, sill, rng = interp_mod.fit_variogram(z, avail, dist, np.array([0, 1, 2, 3]))
    assert 0.0 <= nugget <= 3.0
    assert 1e-3 <= sill <= 10.0
    assert 1.0 <= rng <= 300.0


def test_fit_variogram_skips_pairs_without_enough_shared_hours(interp_mod, panel):
    """Fewer than 100 shared hours is not a usable semivariance, so the pair is
    dropped rather than fitted on noise."""
    wide, feat = panel
    values = wide.to_numpy(dtype=float)
    avail = ~np.isnan(values)
    # Station 3 reports only 50 hours, so every pair involving it is excluded.
    avail[50:, 3] = False
    dist = interp_mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    z = np.nan_to_num(values)
    params = interp_mod.fit_variogram(z, avail, dist, np.array([0, 1, 2, 3]))
    assert len(params) == 3
    assert all(np.isfinite(params))


def test_krige_weights_sum_to_one(interp_mod, panel):
    """Ordinary kriging imposes unbiasedness, so a constant field comes back
    unchanged whatever the geometry."""
    _, feat = panel
    dist = interp_mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    z = np.full((10, 4), 1.5)
    avail = np.ones((10, 4), dtype=bool)
    out = interp_mod.krige(z, avail, dist, np.array([0, 1, 2]), [3],
                           (0.2, 0.8, 15.0))
    assert out.shape == (10, 1)
    assert out == pytest.approx(np.full((10, 1), 1.5))


def test_krige_returns_the_hourly_mean_with_one_source(interp_mod, panel):
    """With a single reporting station there is nothing to interpolate, so the
    standardised field is 0 -- which is the hourly mean once unstandardised."""
    _, feat = panel
    dist = interp_mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    z = np.full((5, 4), 2.0)
    avail = np.zeros((5, 4), dtype=bool)
    avail[:, 0] = True
    out = interp_mod.krige(z, avail, dist, np.array([0, 1, 2]), [3],
                           (0.2, 0.8, 15.0))
    assert out == pytest.approx(np.zeros((5, 1)))


def test_krige_leaves_an_empty_hour_as_nan(interp_mod, panel):
    _, feat = panel
    dist = interp_mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    z = np.zeros((3, 4))
    avail = np.zeros((3, 4), dtype=bool)
    out = interp_mod.krige(z, avail, dist, np.array([0, 1, 2]), [3],
                           (0.2, 0.8, 15.0))
    assert np.isnan(out).all()


def test_krige_reuses_one_solve_per_availability_pattern(interp_mod, panel):
    """Hours sharing an availability pattern must get identical weights, which
    is what makes the pattern cache correct and not just fast."""
    _, feat = panel
    dist = interp_mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    rng = np.random.default_rng(1)
    z = rng.normal(size=(6, 4))
    avail = np.ones((6, 4), dtype=bool)
    avail[3:, 1] = False
    out = interp_mod.krige(z, avail, dist, np.array([0, 1, 2]), [3],
                           (0.2, 0.8, 15.0))
    # Same pattern, same z: hours 0 and 1 differ only through z, so feed them
    # the same row and the output must match exactly.
    z[1] = z[0]
    out2 = interp_mod.krige(z, avail, dist, np.array([0, 1, 2]), [3],
                            (0.2, 0.8, 15.0))
    assert out2[0, 0] == pytest.approx(out2[1, 0])
    assert out.shape == (6, 1)


# --------------------------------------------------------------------------
# Design matrix
# --------------------------------------------------------------------------

def test_design_builds_one_row_per_observed_hour(interp_mod, panel):
    wide, feat = panel
    values = wide.to_numpy(dtype=float)
    avail = ~np.isnan(values)
    dist = interp_mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    times = wide.index.tz_convert(interp_mod.TZ)

    x, y, ok = interp_mod.design(values, avail, dist, feat, times, 0,
                                 np.array([1, 2, 3]), 1.0)
    assert len(x) == len(y) == ok.sum() == 200
    assert list(x.columns) == interp_mod.OSM_FEATURES + [
        "dist_nearest_src_km", "hour", "dayofweek", "month",
        "regional_mean", "idw_est"]
    assert y == pytest.approx(values[ok, 0])


def test_design_holds_the_station_features_constant(interp_mod, panel):
    """Land use does not change hour to hour, so each column is one value."""
    wide, feat = panel
    values = wide.to_numpy(dtype=float)
    avail = ~np.isnan(values)
    dist = interp_mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    times = wide.index.tz_convert(interp_mod.TZ)

    x, _, _ = interp_mod.design(values, avail, dist, feat, times, 2,
                                np.array([0, 1, 3]), 1.0)
    for col in interp_mod.OSM_FEATURES:
        assert x[col].nunique() == 1
        assert x[col].iloc[0] == feat.iloc[2][col]
    assert x["dist_nearest_src_km"].nunique() == 1
    assert x["dist_nearest_src_km"].iloc[0] == pytest.approx(
        np.min(dist[2, [0, 1, 3]]))


def test_design_uses_local_time_for_the_hour_column(interp_mod, panel):
    """Rush hour and the nocturnal inversion are local, the readings are UTC."""
    wide, feat = panel
    values = wide.to_numpy(dtype=float)
    avail = ~np.isnan(values)
    dist = interp_mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    times = wide.index.tz_convert(interp_mod.TZ)

    x, _, _ = interp_mod.design(values, avail, dist, feat, times, 0,
                                np.array([1, 2, 3]), 1.0)
    # 2024-01-01T00:00Z is 05:30 IST.
    assert x["hour"].iloc[0] == 5
    assert wide.index[0].hour == 0


def test_design_drops_hours_the_target_did_not_report(interp_mod, panel):
    wide, feat = panel
    values = wide.to_numpy(dtype=float).copy()
    values[:40, 0] = np.nan
    avail = ~np.isnan(values)
    dist = interp_mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    times = wide.index.tz_convert(interp_mod.TZ)

    x, y, ok = interp_mod.design(values, avail, dist, feat, times, 0,
                                 np.array([1, 2, 3]), 1.0)
    assert len(x) == len(y) == 160
    assert not ok[:40].any()


def test_design_excludes_the_target_from_its_own_sources(interp_mod, panel):
    """A station must never help predict itself, so the IDW column built from
    the other stations cannot simply echo the observation."""
    wide, feat = panel
    values = wide.to_numpy(dtype=float)
    avail = ~np.isnan(values)
    dist = interp_mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    times = wide.index.tz_convert(interp_mod.TZ)

    x, y, _ = interp_mod.design(values, avail, dist, feat, times, 0,
                                np.array([1, 2, 3]), 1.0)
    assert not np.allclose(x["idw_est"].to_numpy(), y)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def test_metrics_mae_rmse_and_count(interp_mod):
    pred = np.array([1.0, 2.0, 3.0])
    truth = np.array([2.0, 2.0, 5.0])
    mae, rmse, n = interp_mod.metrics(pred, truth)
    assert mae == pytest.approx(1.0)
    assert rmse == pytest.approx(np.sqrt(5 / 3))
    assert n == 3


def test_metrics_scores_only_paired_values(interp_mod):
    pred = np.array([1.0, np.nan, 3.0])
    truth = np.array([2.0, 2.0, np.nan])
    mae, rmse, n = interp_mod.metrics(pred, truth)
    assert n == 1
    assert mae == pytest.approx(1.0)
    assert rmse == pytest.approx(1.0)


def test_metrics_with_nothing_to_score(interp_mod):
    mae, rmse, n = interp_mod.metrics(np.array([np.nan]), np.array([1.0]))
    assert np.isnan(mae)
    assert np.isnan(rmse)
    assert n == 0


def test_metrics_is_exact_on_a_perfect_prediction(interp_mod):
    v = np.array([10.0, 20.0, 30.0])
    assert interp_mod.metrics(v, v) == (0.0, 0.0, 3)


def test_summarise_weights_each_station_by_its_hours(interp_mod):
    """A station with ten times the hours carries ten times the weight."""
    cv = pd.DataFrame({
        "model": ["idw", "idw", "kriging", "kriging"],
        "n": [100, 900, 100, 900],
        "mae": [10.0, 20.0, 30.0, 30.0],
        "rmse": [10.0, 20.0, 30.0, 30.0],
    })
    out = interp_mod.summarise(cv).set_index("model")
    assert out.loc["idw", "MAE"] == pytest.approx(19.0)
    assert out.loc["idw", "RMSE"] == pytest.approx(np.sqrt(0.1 * 100 + 0.9 * 400))
    assert out.loc["idw", "n_station_hours"] == 1000
    assert out.loc["idw", "worst_station_RMSE"] == 20.0
    assert out.loc["idw", "best_station_RMSE"] == 10.0


def test_summarise_is_sorted_best_rmse_first(interp_mod):
    cv = pd.DataFrame({
        "model": ["bad", "good", "middling"],
        "n": [100, 100, 100],
        "mae": [30.0, 10.0, 20.0],
        "rmse": [30.0, 10.0, 20.0],
    })
    assert list(interp_mod.summarise(cv)["model"]) == ["good", "middling", "bad"]


def test_calibrate_uncertainty_tracks_the_level_not_the_geometry(interp_mod):
    """Sigma is fitted against the prediction: error grows with the level, so
    the slope has to come back positive."""
    rng = np.random.default_rng(2)
    pred = rng.uniform(10, 100, size=4000)
    obs = pred + rng.normal(0, 1, size=4000) * (0.3 * pred)

    cal = interp_mod.calibrate_uncertainty(pred, obs)
    assert set(cal) == {"form", "a", "b", "floor", "rmse_overall", "note"}
    assert cal["b"] > 0.1
    assert cal["floor"] > 0
    assert cal["rmse_overall"] == pytest.approx(
        float(np.sqrt(np.mean((pred - obs) ** 2))))


def test_calibrate_uncertainty_ignores_unpaired_rows(interp_mod):
    rng = np.random.default_rng(3)
    pred = rng.uniform(10, 100, size=4000)
    obs = pred + rng.normal(0, 0.2 * pred)
    with_holes_pred = np.concatenate([pred, [np.nan, 50.0]])
    with_holes_obs = np.concatenate([obs, [50.0, np.nan]])

    a = interp_mod.calibrate_uncertainty(pred, obs)
    b = interp_mod.calibrate_uncertainty(with_holes_pred, with_holes_obs)
    assert a == b


def test_calibrate_uncertainty_floor_is_the_smallest_bin_error(interp_mod):
    """The fitted intercept can land negative, which would be a nonsensical
    sigma on a clean hour; the floor is a number the data actually showed."""
    rng = np.random.default_rng(4)
    pred = rng.uniform(10, 100, size=4000)
    obs = pred + rng.normal(0, 1, size=4000) * (0.3 * pred)
    cal = interp_mod.calibrate_uncertainty(pred, obs)
    assert cal["floor"] <= cal["rmse_overall"]
    assert cal["form"] == "max(floor, a + b * prediction)"


# --------------------------------------------------------------------------
# Feature bounds
# --------------------------------------------------------------------------

def test_feature_bounds_records_the_training_range(interp_mod, panel):
    _, feat = panel
    bounds = interp_mod.feature_bounds(feat)
    assert bounds["n_stations"] == 4
    assert set(bounds["features"]) == set(interp_mod.OSM_FEATURES)

    road = bounds["features"]["dist_major_road_m"]
    assert set(road) == {"min", "max", "p01", "p99", "median"}
    assert road["min"] == 55.0
    assert road["max"] == 900.0
    assert road["min"] <= road["p01"] <= road["median"] <= road["p99"] <= road["max"]


def test_feature_bounds_is_json_serialisable(interp_mod, panel):
    """It is written to feature_bounds.json and read back by the app."""
    _, feat = panel
    assert json.loads(json.dumps(interp_mod.feature_bounds(feat)))


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------

BBOX = (72.70, 18.85, 73.20, 19.40)


def test_grid_axes_puts_centres_at_half_cell_offsets(interp_mod):
    lats, lons = interp_mod.grid_axes(BBOX, 5)
    assert len(lats) == len(lons) == 5
    dlat = (BBOX[3] - BBOX[1]) / 5
    assert lats[0] == pytest.approx(BBOX[1] + dlat / 2)
    assert lats[-1] == pytest.approx(BBOX[3] - dlat / 2)
    # Every centre is strictly inside the box, and evenly spaced.
    assert lats.min() > BBOX[1]
    assert lats.max() < BBOX[3]
    assert np.allclose(np.diff(lats), dlat)


def test_grid_axes_is_north_positive(interp_mod):
    lats, lons = interp_mod.grid_axes(BBOX, 10)
    assert np.all(np.diff(lats) > 0)
    assert np.all(np.diff(lons) > 0)


def test_grid_axes_scales_with_resolution(interp_mod):
    for n in interp_mod.GRID_RESOLUTIONS:
        lats, lons = interp_mod.grid_axes(BBOX, n)
        assert len(lats) == len(lons) == n


@pytest.fixture
def station_features_on_disk(interp_mod, panel, tmp_path, monkeypatch):
    """Point FEATURES_PATH at a fixture parquet instead of data/processed."""
    _, feat = panel
    path = tmp_path / "station_features.parquet"
    feat.reset_index().to_parquet(path, index=False)
    monkeypatch.setattr(interp_mod, "FEATURES_PATH", path)
    return feat


def test_station_cells_finds_the_cell_containing_each_monitor(interp_mod, panel,
                                                              station_features_on_disk):
    n = 25
    idx = interp_mod.station_cells(BBOX, n)
    assert len(idx) == 4
    assert idx.dtype.kind in "iu"
    assert np.all((idx >= 0) & (idx < n * n))

    # Recover the cell centre and check the monitor really is nearest to it.
    lats, lons = interp_mod.grid_axes(BBOX, n)
    feat = station_features_on_disk
    for lat, lon in zip(feat["lat"], feat["lon"]):
        i = int(np.abs(lats - lat).argmin())
        j = int(np.abs(lons - lon).argmin())
        assert i * n + j in set(idx.tolist())


def test_station_cells_collapses_monitors_sharing_a_cell(interp_mod,
                                                         station_features_on_disk):
    """At a coarse resolution several monitors land in one cell, and the flat
    indices are returned unique."""
    idx = interp_mod.station_cells(BBOX, 2)
    assert len(idx) == len(set(idx.tolist()))
    assert len(idx) <= 4


def test_clear_station_cells_forces_monitor_cells_in_range(interp_mod,
                                                           station_features_on_disk):
    """A cell with a monitor standing in it has ground truth by definition, so
    masking it is incoherent whatever the land-use features say."""
    n = 25
    kind = np.full(n * n, interp_mod.MASK_EMPTY, dtype=np.int8)
    cleared_kind, n_cleared = interp_mod.clear_station_cells(kind.copy(), BBOX, n)
    idx = interp_mod.station_cells(BBOX, n)

    assert n_cleared == len(idx)
    assert np.all(cleared_kind[idx] == interp_mod.MASK_NONE)
    # Nothing else is touched.
    others = np.setdiff1d(np.arange(n * n), idx)
    assert np.all(cleared_kind[others] == interp_mod.MASK_EMPTY)


def test_clear_station_cells_counts_only_what_it_changed(interp_mod,
                                                         station_features_on_disk):
    n = 25
    kind = np.full(n * n, interp_mod.MASK_NONE, dtype=np.int8)
    _, n_cleared = interp_mod.clear_station_cells(kind, BBOX, n)
    assert n_cleared == 0


# --------------------------------------------------------------------------
# Extrapolation mask
# --------------------------------------------------------------------------

@pytest.fixture
def bounds():
    """A training band wide enough to classify the stub points below."""
    return {
        "dist_major_road_m": {"min": 50.0, "max": 1000.0, "p01": 60.0,
                              "p99": 900.0, "median": 300.0},
        "road_density_500m": {"min": 5.0, "max": 30.0, "p01": 6.0,
                              "p99": 28.0, "median": 15.0},
        "road_density_1km": {"min": 4.0, "max": 25.0, "p01": 5.0,
                             "p99": 23.0, "median": 12.0},
        "dist_coast_m": {"min": 100.0, "max": 20000.0, "p01": 200.0,
                         "p99": 18000.0, "median": 4000.0},
        "dist_industrial_m": {"min": 0.0, "max": 6000.0, "p01": 0.0,
                              "p99": 5500.0, "median": 1500.0},
        "building_density_500m": {"min": 100.0, "max": 1500.0, "p01": 150.0,
                                  "p99": 1400.0, "median": 700.0},
    }


def _stub_osm(interp_mod, monkeypatch, frame):
    """Stand in for scripts/02_features.py with a fixed feature table."""
    class Stub:
        BBOX = BBOX

        @staticmethod
        def features_for(lats, lons):
            return frame.iloc[:len(np.atleast_1d(lats))].reset_index(drop=True)

    monkeypatch.setattr(interp_mod, "osm_features", lambda: Stub)
    return Stub


def test_extrapolation_mask_passes_a_point_inside_the_band(interp_mod, bounds,
                                                           monkeypatch):
    frame = pd.DataFrame([{
        "dist_major_road_m": 300.0, "road_density_500m": 15.0,
        "road_density_1km": 12.0, "dist_coast_m": 4000.0,
        "dist_industrial_m": 1500.0, "building_density_500m": 700.0,
    }])
    _stub_osm(interp_mod, monkeypatch, frame)
    mask, kind, reasons = interp_mod.extrapolation_mask([19.0], [72.9], bounds)
    assert not mask[0]
    assert kind[0] == interp_mod.MASK_NONE
    assert reasons["_below_band"] == 0
    assert reasons["_above_band"] == 0


def test_extrapolation_mask_calls_open_water_empty(interp_mod, bounds, monkeypatch):
    """No road, no buildings: the uninteresting half of the mask, correctly
    refused because there is nothing there."""
    frame = pd.DataFrame([{
        "dist_major_road_m": 300.0, "road_density_500m": 0.0,
        "road_density_1km": 0.0, "dist_coast_m": 4000.0,
        "dist_industrial_m": 1500.0, "building_density_500m": 0.0,
    }])
    _stub_osm(interp_mod, monkeypatch, frame)
    mask, kind, reasons = interp_mod.extrapolation_mask([19.0], [72.9], bounds)
    assert mask[0]
    assert kind[0] == interp_mod.MASK_EMPTY
    assert reasons["road_density_500m"] == 1
    assert reasons["_below_band"] == 1


def test_extrapolation_mask_calls_quiet_inhabited_land_remote(interp_mod, bounds,
                                                              monkeypatch):
    """The caveat that matters: built-up ground farther from a road than any
    monitor sits, where the model extrapolates into quieter land."""
    frame = pd.DataFrame([{
        "dist_major_road_m": 5000.0, "road_density_500m": 15.0,
        "road_density_1km": 12.0, "dist_coast_m": 4000.0,
        "dist_industrial_m": 20000.0, "building_density_500m": 700.0,
    }])
    _stub_osm(interp_mod, monkeypatch, frame)
    mask, kind, _ = interp_mod.extrapolation_mask([19.0], [72.9], bounds)
    assert mask[0]
    assert kind[0] == interp_mod.MASK_REMOTE


def test_extrapolation_mask_lets_emptiness_win_over_remoteness(interp_mod, bounds,
                                                               monkeypatch):
    """Open water far from a road is refused because there is nothing there,
    not because of the distance."""
    frame = pd.DataFrame([{
        "dist_major_road_m": 50000.0, "road_density_500m": 0.0,
        "road_density_1km": 0.0, "dist_coast_m": 4000.0,
        "dist_industrial_m": 50000.0, "building_density_500m": 0.0,
    }])
    _stub_osm(interp_mod, monkeypatch, frame)
    _, kind, _ = interp_mod.extrapolation_mask([19.0], [72.9], bounds)
    assert kind[0] == interp_mod.MASK_EMPTY


def test_extrapolation_mask_falls_back_to_other(interp_mod, bounds, monkeypatch):
    """Out of band on a feature that is neither a density nor a remoteness
    measure -- here, closer to the sea than any monitor."""
    frame = pd.DataFrame([{
        "dist_major_road_m": 300.0, "road_density_500m": 15.0,
        "road_density_1km": 12.0, "dist_coast_m": 10.0,
        "dist_industrial_m": 1500.0, "building_density_500m": 700.0,
    }])
    _stub_osm(interp_mod, monkeypatch, frame)
    mask, kind, _ = interp_mod.extrapolation_mask([19.0], [72.9], bounds)
    assert mask[0]
    assert kind[0] == interp_mod.MASK_OTHER


def test_extrapolation_mask_band_choice_changes_the_verdict(interp_mod, bounds,
                                                            monkeypatch):
    """A point between the minimum and p01 is in-range on min-max and out of
    range on the tighter band -- which is why min-max is what ships."""
    frame = pd.DataFrame([{
        "dist_major_road_m": 55.0, "road_density_500m": 15.0,
        "road_density_1km": 12.0, "dist_coast_m": 4000.0,
        "dist_industrial_m": 1500.0, "building_density_500m": 700.0,
    }])
    _stub_osm(interp_mod, monkeypatch, frame)
    wide_band, _, _ = interp_mod.extrapolation_mask([19.0], [72.9], bounds)
    tight, _, _ = interp_mod.extrapolation_mask([19.0], [72.9], bounds,
                                                band=("p01", "p99"))
    assert not wide_band[0]
    assert tight[0]


def test_extrapolation_mask_counts_reasons_per_feature(interp_mod, bounds,
                                                       monkeypatch):
    frame = pd.DataFrame([
        {"dist_major_road_m": 300.0, "road_density_500m": 15.0,
         "road_density_1km": 12.0, "dist_coast_m": 4000.0,
         "dist_industrial_m": 1500.0, "building_density_500m": 700.0},
        {"dist_major_road_m": 300.0, "road_density_500m": 0.0,
         "road_density_1km": 0.0, "dist_coast_m": 4000.0,
         "dist_industrial_m": 1500.0, "building_density_500m": 0.0},
        {"dist_major_road_m": 9000.0, "road_density_500m": 15.0,
         "road_density_1km": 12.0, "dist_coast_m": 4000.0,
         "dist_industrial_m": 1500.0, "building_density_500m": 700.0},
    ])
    _stub_osm(interp_mod, monkeypatch, frame)
    mask, kind, reasons = interp_mod.extrapolation_mask([19.0] * 3, [72.9] * 3,
                                                        bounds)
    assert list(mask) == [False, True, True]
    assert reasons["_empty"] == 1
    assert reasons["_remote"] == 1
    assert reasons["_other"] == 0
    assert set(reasons) >= set(interp_mod.OSM_FEATURES)
    assert sum(1 for k in kind if k != interp_mod.MASK_NONE) == 2


def test_mask_kind_codes_are_distinct(interp_mod):
    codes = (interp_mod.MASK_NONE, interp_mod.MASK_EMPTY,
             interp_mod.MASK_REMOTE, interp_mod.MASK_OTHER)
    assert len(set(codes)) == 4
    assert interp_mod.MASK_NONE == 0


# --------------------------------------------------------------------------
# predict_surface()
# --------------------------------------------------------------------------

CARD = {
    "model": "idw",
    "idw_power": 0.5,
    "uncertainty": {"form": "max(floor, a + b * prediction)",
                    "a": -0.6, "b": 0.6, "floor": 9.26},
    "cv": {"rmse": 21.1, "mae": 11.2},
}


@pytest.fixture
def prod(interp_mod, panel, bounds, monkeypatch):
    """Stand in for production() and for the baked masks.

    grid_masks() returns None, so predict_surface takes the live path and
    computes the mask itself -- which is also what the app does at a resolution
    nobody baked.
    """
    wide, feat = panel
    monkeypatch.setattr(interp_mod, "production", lambda: {
        "wide": wide, "feat": feat, "card": CARD, "bounds": bounds,
        "lat": feat["lat"].to_numpy(), "lon": feat["lon"].to_numpy(),
    })
    monkeypatch.setattr(interp_mod, "grid_masks", lambda: None)
    in_band = pd.DataFrame([{
        "dist_major_road_m": 300.0, "road_density_500m": 15.0,
        "road_density_1km": 12.0, "dist_coast_m": 4000.0,
        "dist_industrial_m": 1500.0, "building_density_500m": 700.0,
    }] * 10_000)
    _stub_osm(interp_mod, monkeypatch, in_band)
    return wide, feat


def test_predict_surface_rejects_other_pollutants(interp_mod):
    """Only pm25 was fetched in Phase 1."""
    with pytest.raises(ValueError, match="pm25"):
        interp_mod.predict_surface("2024-01-01T00:00Z", pollutant="no2")


def test_predict_surface_shapes(interp_mod, prod, station_features_on_disk):
    wide, _ = prod
    n = 15
    s = interp_mod.predict_surface(wide.index[0], grid_resolution=n)
    assert s.values.shape == (n, n)
    assert s.sigma.shape == (n, n)
    assert s.mask.shape == (n, n)
    assert s.mask_kind.shape == (n, n)
    assert s.lats.shape == (n,)
    assert s.lons.shape == (n,)


def test_predict_surface_stays_within_the_station_readings(interp_mod, prod,
                                                           station_features_on_disk):
    """IDW is a weighted mean, so no cell can exceed the hour's extremes."""
    wide, _ = prod
    row = wide.iloc[0]
    s = interp_mod.predict_surface(wide.index[0], grid_resolution=15)
    assert np.nanmin(s.values) >= row.min() - 1e-9
    assert np.nanmax(s.values) <= row.max() + 1e-9


def test_predict_surface_is_nearest_station_dominated(interp_mod, prod,
                                                       station_features_on_disk):
    """The cell containing a monitor should read close to that monitor."""
    wide, feat = prod
    n = 60
    s = interp_mod.predict_surface(wide.index[0], grid_resolution=n)
    lat, lon = feat["lat"].iloc[0], feat["lon"].iloc[0]
    i = int(np.abs(s.lats - lat).argmin())
    j = int(np.abs(s.lons - lon).argmin())
    assert s.values[i, j] == pytest.approx(wide.iloc[0, 0], abs=5.0)


def test_predict_surface_applies_the_calibrated_sigma(interp_mod, prod,
                                                       station_features_on_disk):
    wide, _ = prod
    s = interp_mod.predict_surface(wide.index[0], grid_resolution=15)
    u = CARD["uncertainty"]
    expected = np.maximum(u["a"] + u["b"] * s.values, u["floor"])
    assert s.sigma == pytest.approx(expected)
    assert (s.sigma >= u["floor"]).all()


def test_predict_surface_metadata(interp_mod, prod, station_features_on_disk):
    wide, _ = prod
    s = interp_mod.predict_surface(wide.index[0], grid_resolution=25)
    assert s.meta["n_stations"] == 4
    assert s.meta["idw_power"] == CARD["idw_power"]
    assert s.meta["cv_rmse"] == CARD["cv"]["rmse"]
    assert s.meta["band"] == ["min", "max"]
    assert 0.0 <= s.meta["masked_fraction"] <= 1.0
    # 25 cells over the ~53 km box is a ~2.1 km cell.
    assert 1.5 < s.meta["cell_km"] < 3.0
    assert s.meta["timestamp"] == wide.index[0]


def test_predict_surface_cell_size_shrinks_with_resolution(interp_mod, prod,
                                                            station_features_on_disk):
    wide, _ = prod
    coarse = interp_mod.predict_surface(wide.index[0], grid_resolution=15)
    fine = interp_mod.predict_surface(wide.index[0], grid_resolution=60)
    assert coarse.meta["cell_km"] > fine.meta["cell_km"]


def test_predict_surface_floors_and_localises_the_timestamp(interp_mod, prod,
                                                             station_features_on_disk):
    """A naive timestamp is read as UTC, and the hour is floored."""
    wide, _ = prod
    s = interp_mod.predict_surface("2024-01-01 00:45:00", grid_resolution=15)
    assert s.meta["timestamp"] == pd.Timestamp("2024-01-01T00:00Z")
    assert s.meta["n_stations"] == 4


def test_predict_surface_blank_when_the_hour_is_missing(interp_mod, prod,
                                                         station_features_on_disk):
    s = interp_mod.predict_surface("1999-01-01T00:00Z", grid_resolution=15)
    assert np.isnan(s.values).all()
    assert np.isnan(s.sigma).all()
    assert s.meta["n_stations"] == 0
    assert s.meta["cell_km"] is None
    assert s.meta["note"] == "no station reported this hour"
    # The mask is still described: it does not depend on the readings.
    assert s.mask.shape == (15, 15)


def test_predict_surface_blank_when_every_station_is_silent(interp_mod, prod,
                                                             station_features_on_disk):
    wide, _ = prod
    s = interp_mod.predict_surface(wide.index[0], grid_resolution=15,
                                   readings={1: np.nan, 2: np.nan,
                                             3: np.nan, 4: np.nan})
    assert np.isnan(s.values).all()
    assert s.meta["n_stations"] == 0


def test_predict_surface_accepts_live_readings(interp_mod, prod,
                                               station_features_on_disk):
    """`readings` overrides the stored panel, so the same interpolation runs on
    a live feed. A constant field must come back constant."""
    wide, _ = prod
    s = interp_mod.predict_surface(wide.index[0], grid_resolution=15,
                                   readings={1: 40.0, 2: 40.0, 3: 40.0, 4: 40.0})
    assert s.meta["n_stations"] == 4
    assert s.values == pytest.approx(np.full((15, 15), 40.0))


def test_predict_surface_live_readings_ignore_unknown_stations(interp_mod, prod,
                                                               station_features_on_disk):
    """A live feed may carry stations the model does not know; they are dropped
    rather than silently shifting the surface."""
    wide, _ = prod
    s = interp_mod.predict_surface(wide.index[0], grid_resolution=15,
                                   readings={1: 40.0, 2: 40.0, 3: 40.0, 4: 40.0,
                                             99999: 900.0})
    assert s.meta["n_stations"] == 4
    assert np.nanmax(s.values) == pytest.approx(40.0)


def test_predict_surface_with_one_reporting_station(interp_mod, prod,
                                                     station_features_on_disk):
    wide, _ = prod
    s = interp_mod.predict_surface(wide.index[0], grid_resolution=15,
                                   readings={1: 55.0})
    assert s.meta["n_stations"] == 1
    assert s.values == pytest.approx(np.full((15, 15), 55.0))


def test_predict_surface_uses_the_baked_mask_when_there_is_one(interp_mod, prod,
                                                               monkeypatch):
    """The deployed app never loads the OSM point clouds -- not even for the
    bbox, which comes out of the baked file too."""
    wide, _ = prod
    n = 15
    kind = np.full((n, n), interp_mod.MASK_EMPTY, dtype=np.int8)
    kind[0, 0] = interp_mod.MASK_NONE
    monkeypatch.setattr(interp_mod, "grid_masks", lambda: {
        "bbox": BBOX, "band": ("min", "max"),
        "reasons": {str(n): {"_empty": n * n - 1}},
        "kind": {n: kind},
    })
    # Any call into the OSM module from here is a bug, so make one fail loudly.
    monkeypatch.setattr(interp_mod, "osm_features",
                        lambda: pytest.fail("predict_surface loaded the OSM layers"))

    s = interp_mod.predict_surface(wide.index[0], grid_resolution=n)
    assert s.mask_kind[0, 0] == interp_mod.MASK_NONE
    assert not s.mask[0, 0]
    assert s.mask.sum() == n * n - 1
    assert s.meta["mask_reasons"] == {"_empty": n * n - 1}


def test_predict_surface_falls_through_for_an_unbaked_resolution(interp_mod, prod,
                                                                  station_features_on_disk,
                                                                  monkeypatch):
    wide, _ = prod
    baked_n = 15
    monkeypatch.setattr(interp_mod, "grid_masks", lambda: {
        "bbox": BBOX, "band": ("min", "max"),
        "reasons": {str(baked_n): {}},
        "kind": {baked_n: np.zeros((baked_n, baked_n), dtype=np.int8)},
    })
    # 23 was never baked, so the mask is computed from the (stubbed) OSM layers.
    s = interp_mod.predict_surface(wide.index[0], grid_resolution=23)
    assert s.values.shape == (23, 23)
    assert not s.mask.any()


def test_surface_is_a_named_tuple(interp_mod, prod, station_features_on_disk):
    """The app unpacks it by name; keep the contract explicit."""
    wide, _ = prod
    s = interp_mod.predict_surface(wide.index[0], grid_resolution=15)
    assert s._fields == ("lats", "lons", "values", "sigma", "mask",
                         "mask_kind", "meta")


# --------------------------------------------------------------------------
# Cross-validation, end to end on the fixture panel
# --------------------------------------------------------------------------

def test_cross_validate_holds_out_whole_cv_groups(interp_mod, panel):
    wide, feat = panel
    cv, (pred, obs) = interp_mod.cross_validate(wide, feat)

    assert set(cv["model"]) == {"regional_mean", "idw", "kriging", "trees",
                                "trees+idw"}
    # One fold per cv_group, and each station scored exactly once per model.
    assert cv["fold"].nunique() == feat["cv_group"].nunique() == 4
    assert len(cv) == 4 * 5
    assert set(cv["station_id"]) == set(feat.index)
    for _, g in cv.groupby("model"):
        assert len(g) == 4
        assert g["station_id"].is_unique

    assert pred.shape == obs.shape
    assert len(pred) == int(cv.loc[cv["model"] == "idw", "n"].sum())
    assert np.isfinite(pred).all()
    assert np.isfinite(obs).all()


def test_cross_validate_scores_look_sane(interp_mod, panel):
    """The fixture panel is a shared city-wide level plus a small per-station
    offset, so every model should land within a few ug/m3."""
    wide, feat = panel
    cv, _ = interp_mod.cross_validate(wide, feat)
    assert (cv["mae"] > 0).all()
    assert (cv["rmse"] >= cv["mae"]).all()
    assert cv["rmse"].max() < 25.0
    assert (cv["n"] == 200).all()
    assert cv["idw_power"].isin(interp_mod.IDW_POWERS).all()


def test_cross_validate_feeds_summarise(interp_mod, panel):
    wide, feat = panel
    cv, _ = interp_mod.cross_validate(wide, feat)
    table = interp_mod.summarise(cv)
    assert len(table) == 5
    assert table["n_station_hours"].min() == 800
    assert table["RMSE"].is_monotonic_increasing
