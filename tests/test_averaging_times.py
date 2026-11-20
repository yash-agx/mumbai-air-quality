"""The machinery behind scripts/11_averaging_times.py.

Synthetic panels whose answer is known by construction: a stable spatial
pattern buried in independent hourly noise, where reliability has to rise
with averaging time, and pure noise, where it has to stay near zero. No data
files and no cross-validation of the real network here.
"""

import numpy as np
import pandas as pd
import pytest

TZ = "Asia/Kolkata"


@pytest.fixture(scope="module")
def at(script_module):
    return script_module("11_averaging_times.py", "averaging_times")


def _panel(n_stations=30, days=84, offset_sd=1.0, noise_sd=8.0, seed=0):
    """Hourly wide panel in UTC: shared city signal + stable offsets + noise.

    Stamps are whole UTC hours, so in IST they sit on :30, as the real data
    does. 19:00 UTC is 00:30 IST, so the record starts on an IST day boundary.
    """
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2025-01-01 19:00", periods=days * 24, freq="h",
                          tz="UTC")
    city = 40 + 15 * np.sin(np.arange(len(hours)) / 30.0)
    offsets = rng.normal(0, offset_sd, n_stations)
    noise = rng.normal(0, noise_sd, (len(hours), n_stations))
    wide = pd.DataFrame(city[:, None] + offsets[None, :] + noise,
                        index=hours, columns=np.arange(1, n_stations + 1))
    wide.index.name = "timestamp"
    groups = np.array([f"g{i}" for i in wide.columns])
    return wide, groups


def _reliability(at, grid, anchor, groups, spec):
    X, meta = at.window_means(grid, spec, anchor)
    z, _ = at.anomalies(X)
    Y = at.by_group(z, groups[: X.shape[1]]).to_numpy()
    i, j = at.eligible_pairs(meta["start"], meta["end"])
    _, _, r = at.pair_correlations(Y, i, j)
    return float(r.mean()), len(r), Y, X


# --------------------------------------------------------------------------
# Windows and coverage
# --------------------------------------------------------------------------

def test_grid_keeps_the_half_hour_ist_clock(at):
    wide, _ = _panel(n_stations=3, days=3)
    grid, anchor = at.hourly_grid(wide)
    assert anchor.minute == 30 and anchor.hour == 0
    assert (grid.index.minute == 30).all()
    assert str(grid.index.tz) == TZ


def test_fixed_windows_are_anchored_and_sized(at):
    wide, _ = _panel(n_stations=12, days=4)
    grid, anchor = at.hourly_grid(wide)
    X, meta = at.window_means(grid, 24, anchor)
    assert (meta["nominal_hours"] == 24).all()
    assert meta["start"].iloc[0] == anchor
    # every full day passes; the stub at the end does not
    assert len(X) == len(grid) // 24


def test_coverage_needs_75_percent_of_nominal_hours(at):
    wide, _ = _panel(n_stations=12, days=2)
    grid, anchor = at.hourly_grid(wide)
    day = grid.index < anchor + pd.Timedelta("24h")
    col17, col18 = grid.columns[0], grid.columns[1]
    g = grid.copy()
    g.loc[g.index[day][:7], col17] = np.nan     # 17 of 24 valid
    g.loc[g.index[day][:6], col18] = np.nan     # 18 of 24 valid
    X, _ = at.window_means(g, 24, anchor)
    assert np.isnan(X.iloc[0][col17])
    assert np.isfinite(X.iloc[0][col18])


def test_a_window_needs_ten_stations(at):
    wide, _ = _panel(n_stations=12, days=2)
    grid, anchor = at.hourly_grid(wide)
    g = grid.copy()
    first = g.index < anchor + pd.Timedelta("24h")
    g.loc[first, g.columns[:3]] = np.nan        # 9 stations left on day one
    X, _ = at.window_means(g, 24, anchor)
    assert X.index[0] != 0


def test_a_partial_month_fails_on_its_nominal_length(at):
    """Ten days of a 31-day month is not a month mean, however complete."""
    wide, _ = _panel(n_stations=12, days=41)    # Jan 1 + Feb 1-10 IST
    grid, anchor = at.hourly_grid(wide)
    X, meta = at.window_means(grid, "month", anchor)
    assert list(X.index) == ["2025-01"]
    assert meta["nominal_hours"].iloc[0] == 31 * 24


def test_seasons_put_december_with_the_next_january(at):
    times = pd.date_range("2025-12-15 00:30", periods=24 * 40, freq="h", tz=TZ)
    labels, meta = at.window_labels(times, "season", times[0])
    assert set(labels) == {"2026-DJF"}
    assert meta.loc["2026-DJF", "nominal_hours"] == (31 + 31 + 28) * 24


def test_anomalies_remove_the_window_mean(at):
    X = pd.DataFrame([[10.0, 20.0, 30.0], [50.0, 50.0, 80.0]])
    z, raw = at.anomalies(X)
    assert np.allclose(raw.mean(axis=1), 0)
    assert np.allclose(z.mean(axis=1), 0)
    assert np.allclose(z.std(axis=1, ddof=0), 1)


# --------------------------------------------------------------------------
# (b) reliability: the noise check
# --------------------------------------------------------------------------

def test_reliability_rises_with_averaging_time_for_a_stable_pattern(at):
    """A real, stable pattern under heavy noise shows up as windows lengthen."""
    wide, groups = _panel(offset_sd=1.0, noise_sd=8.0)
    grid, anchor = at.hourly_grid(wide)
    r = {spec: _reliability(at, grid, anchor, groups, spec)[0]
         for spec in (1, 24, 168)}
    assert r[1] < 0.05
    assert r[1] < r[24] < r[168]
    assert r[168] > 0.5
    # independent noise: Spearman-Brown should be close
    assert r[24] == pytest.approx(at.spearman_brown(1 / 65, 24), abs=0.1)


def test_reliability_is_near_zero_for_pure_noise(at):
    # 24 weeks: with only a dozen weekly windows the mean pair correlation
    # wanders by ~0.05 around zero on sampling alone.
    wide, groups = _panel(offset_sd=0.0, noise_sd=8.0, days=168, seed=1)
    grid, anchor = at.hourly_grid(wide)
    for spec in (1, 24, 168):
        r, n, _, _ = _reliability(at, grid, anchor, groups, spec)
        assert n > 0
        assert abs(r) < 0.05, (spec, r)


def test_split_half_does_not_track_averaging_time(at):
    """Why b1 is not the primary: each half pools half the record regardless.

    On a stable pattern it is about the same at 1 h and 1 week, so it cannot
    show structure 'appearing' with averaging time.
    """
    wide, groups = _panel(offset_sd=1.0, noise_sd=8.0)
    grid, anchor = at.hourly_grid(wide)
    r = {}
    for spec in (1, 168):
        _, _, Y, _ = _reliability(at, grid, anchor, groups, spec)
        r[spec] = at.split_half(Y, np.arange(len(Y)) % 2)
    assert r[1] > 0.8
    assert abs(r[1] - r[168]) < 0.1


def test_window_pairs_respect_the_gap(at):
    starts = pd.date_range("2025-01-01", periods=30, freq="D", tz=TZ)
    ends = starts + pd.Timedelta("1D")
    i, j = at.eligible_pairs(pd.Series(starts), pd.Series(ends))
    assert len(i) > 0
    assert ((starts[j] - ends[i]) >= pd.Timedelta("7D")).all()
    # 30 daily windows: partners of window k are k+8 onward
    assert len(i) == sum(range(1, 30 - 7))


def test_window_pairs_are_sampled_when_there_are_too_many(at):
    starts = pd.date_range("2025-01-01", periods=2000, freq="h", tz=TZ)
    ends = starts + pd.Timedelta("1h")
    i, j = at.eligible_pairs(pd.Series(starts), pd.Series(ends),
                             max_pairs=500, rng=np.random.default_rng(0))
    assert len(i) == 500
    assert len(set(zip(i, j))) == 500
    assert ((starts[j] - ends[i]) >= pd.Timedelta("7D")).all()


def test_pair_correlation_needs_enough_groups(at):
    Y = np.array([[1.0, 2.0, 3.0] + [np.nan] * 9,
                  [1.0, 2.0, 3.0] + [np.nan] * 9])
    _, _, r = at.pair_correlations(Y, np.array([0]), np.array([1]))
    assert len(r) == 0


# --------------------------------------------------------------------------
# (a) spatial share
# --------------------------------------------------------------------------

def test_icc_is_near_zero_for_noise_and_raw_share_is_not(at):
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(5, 35))                # five windows of pure noise
    raw, icc, _ = at.spatial_share(Y)
    assert raw > 0.1                            # ~1/(n+1): the bias
    assert abs(icc) < 0.15


def test_icc_recovers_a_known_stable_share(at):
    rng = np.random.default_rng(0)
    stable = rng.normal(0, 1, 35)
    stable = (stable - stable.mean()) / stable.std()    # variance exactly 1
    Y = stable[None, :] + rng.normal(0, 1, (400, 35))   # true share 0.5
    _, icc, _ = at.spatial_share(Y)
    assert icc == pytest.approx(0.5, abs=0.08)


def test_one_window_has_no_icc(at):
    _, icc, n = at.spatial_share(np.random.default_rng(0).normal(size=(1, 20)))
    assert n == 1 and np.isnan(icc)


# --------------------------------------------------------------------------
# (c) distance pattern
# --------------------------------------------------------------------------

def test_a_smooth_field_makes_near_pairs_alike(at):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 40, 30)                  # stations on a 40 km line
    dist = np.abs(x[:, None] - x[None, :])
    field = np.sin(x / 8.0)                     # smooth over ~10 km
    Z = field[None, :] + rng.normal(0, 0.3, (50, 30))
    sq, a, b = at.pair_sq_diffs(Z)
    corr, _ = at.pooled_corr(sq)
    same = np.zeros((30, 30), bool)
    contrast, n_near, n_far = at.near_far(corr, dist[a, b], same[a, b])
    assert n_near > 0 and n_far > 0
    assert contrast > 0.5
    assert at.mantel_p(corr, a, b, dist, same, n_perm=199) < 0.05


def test_no_field_means_no_distance_pattern(at):
    rng = np.random.default_rng(1)
    x = np.linspace(0, 40, 30)
    dist = np.abs(x[:, None] - x[None, :])
    Z = rng.normal(0, 1, (200, 30))
    sq, a, b = at.pair_sq_diffs(Z)
    corr, _ = at.pooled_corr(sq)
    same = np.zeros((30, 30), bool)
    contrast, _, _ = at.near_far(corr, dist[a, b], same[a, b])
    assert abs(contrast) < 0.1


def test_same_group_pairs_are_kept_out_of_the_contrast(at):
    corr = np.array([5.0, 0.2, 0.0])
    d = np.array([0.1, 3.0, 20.0])
    same = np.array([True, False, False])
    contrast, n_near, _ = at.near_far(corr, d, same)
    assert n_near == 1
    assert contrast == pytest.approx(0.2)


# --------------------------------------------------------------------------
# (d) prediction skill
# --------------------------------------------------------------------------

def test_skill_weights_reproduce_repetition(at):
    rng = np.random.default_rng(0)
    e1, e2 = rng.normal(size=(6, 4)), rng.normal(size=(6, 4))
    w = np.array([2, 0, 1, 1, 3, 0])
    rep = np.repeat(np.arange(6), w)
    assert np.allclose(at.skill(e1, e2, w), at.skill(e1[rep], e2[rep]))


def test_thin_data_labels(at):
    assert at.thin_label(72) == "interpretable"
    assert at.thin_label(30) == "interpretable"
    assert at.thin_label(17) == "thin"
    assert at.thin_label(5) == "too thin"
    assert at.thin_label(1) == "too thin"
