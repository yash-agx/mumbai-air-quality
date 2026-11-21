"""The machinery behind scripts/13_no2_landuse.py, on synthetic inputs.

The closed-form leave-one-out operators are checked against explicit refits,
including the bootstrap case where a group appears more than once and every
copy must be held out together. No data files.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def lu(script_module):
    return script_module("13_no2_landuse.py", "no2_landuse")


def _units(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"log_dist_major_road": rng.uniform(0.5, 3, n),
                         "road_density_500m": rng.uniform(1, 30, n)})


def _explicit_loo(X, y, copies):
    """Refit without each distinct group (all its copies) and predict it."""
    e, e0 = [], []
    for i in range(len(copies)):
        keep = np.repeat(np.arange(len(copies)) != i, copies)
        Xr, yr = np.repeat(X, copies, 0)[keep], np.repeat(y, copies)[keep]
        beta, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
        e.append(y[i] - X[i] @ beta)
        e0.append(y[i] - yr.mean())
    return np.array(e), np.array(e0)


def test_loo_operator_matches_refits(lu):
    u = _units()
    X = lu.design(u)
    y = np.random.default_rng(1).normal(size=len(u))
    M, M0, _ = lu.loo_operators(X)
    e, e0 = _explicit_loo(X, y, np.ones(len(u), int))
    assert np.allclose(M @ y, e)
    assert np.allclose(M0 @ y, e0)


def test_weighted_operator_holds_all_copies_out(lu):
    u = _units()
    X = lu.design(u)
    y = np.random.default_rng(2).normal(size=len(u))
    w = np.array([1, 3, 1, 2, 1, 1, 2, 1, 1, 4, 1, 1])
    M, M0, _ = lu.loo_operators(X, w)
    e, e0 = _explicit_loo(X, y, w)
    assert np.allclose(M @ y, e)
    assert np.allclose(M0 @ y, e0)


def test_skill_positive_on_signal_and_noise_gives_large_p(lu):
    u = _units(n=35, seed=3)
    rng = np.random.default_rng(4)
    signal = -2 * u["log_dist_major_road"] + 0.1 * u["road_density_500m"]
    r2, p = lu.perm_p(u, signal + 0.05 * rng.normal(size=35), 999, 0)
    assert r2 > 0.9 and p < 0.01
    r2n, pn = lu.perm_p(u, rng.normal(size=35), 999, 0)
    assert r2n < 0.2 and pn > 0.05


def test_right_sign(lu):
    assert lu.right_sign({"log_dist_major_road": (-1.0, -3.0),
                          "road_density_500m": (0.5, 1.0)})
    # one expected-sign predictor, the other wrong but weak: still right
    assert lu.right_sign({"log_dist_major_road": (-1.0, -3.0),
                          "road_density_500m": (-0.5, -1.0)})
    # a strong wrong-signed coefficient disqualifies
    assert not lu.right_sign({"log_dist_major_road": (-1.0, -3.0),
                              "road_density_500m": (-0.5, -2.5)})
    assert not lu.right_sign({"log_dist_major_road": (1.0, 1.0),
                              "road_density_500m": (-0.5, -1.0)})


@pytest.mark.parametrize("args, prefix", [
    ((0.4, 0.01, True, 0.0, 0.6, True, 0.1, 0.7), "L1"),
    ((0.4, 0.01, True, 0.1, 0.6, True, -0.1, 0.7), "L2"),
    ((0.05, 0.3, True, 0.0, 0.6, True, -0.3, 0.4), "L3"),
    ((0.0, 0.6, True, 0.3, 0.01, True, -0.6, 0.1), "L4"),
    ((0.0, 0.6, True, 0.1, 0.5, True, -0.6, -0.05), "L3/L4 overlap"),
    ((0.0, 0.6, True, 0.1, 0.04, False, -0.6, -0.05), "L5"),
    ((0.4, 0.01, False, 0.0, 0.6, True, 0.1, 0.7), "L5"),
])
def test_reading_rules(lu, args, prefix):
    assert lu.reading(*args).startswith(prefix)


def test_station_outcomes_anomaly_and_hour_rules(lu):
    idx = pd.date_range("2025-03-01", periods=3000, freq="h", tz="UTC")
    cols = list(range(12))
    base = np.linspace(10, 30, len(idx))[:, None]
    V = base + np.arange(12)[None, :]            # station j sits j above base
    panel = pd.DataFrame(V, index=idx, columns=cols)
    panel.iloc[:, 0] = np.nan
    panel.iloc[:500, 1] = np.nan                  # 2500 hours: still in
    panel.iloc[:1500, 2] = np.nan                 # 1500 hours: dropped
    out = lu.station_outcomes(panel)
    assert np.isnan(out.loc[0, "anomaly"]) and np.isnan(out.loc[2, "level"])
    # anomaly is offset from the city mean of the stations present
    a = out["anomaly"].dropna()
    assert np.allclose(np.diff(a.loc[3:].to_numpy()), 1.0)
