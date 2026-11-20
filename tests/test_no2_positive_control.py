"""The machinery behind scripts/12_no2_positive_control.py.

Cleaning rules, the like-for-like masking, and the pre-registered decision
rules, on small synthetic inputs whose answer is known. No data files.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def nc(script_module):
    return script_module("12_no2_positive_control.py", "no2_positive_control")


START = pd.Timestamp("2025-03-01 00:00", tz="UTC")


def _raw(values_by_station, parameter="no2", start=START, flagged=None):
    frames = []
    for sid, values in values_by_station.items():
        n = len(values)
        frames.append(pd.DataFrame({
            "timestamp": pd.date_range(start, periods=n, freq="h"),
            "value": pd.array(values, dtype="float64"),
            "flagged": [False] * n if flagged is None else flagged.get(sid, [False] * n),
            "station_id": sid,
            "sensor_id": sid * 10,
            "parameter": parameter,
            "units": "ppb",
        }))
    return pd.concat(frames, ignore_index=True)


def _wiggle(n, base=20.0, seed=0):
    """A series that never repeats a value, so the stuck mask leaves it alone."""
    rng = np.random.default_rng(seed)
    return list(base + rng.uniform(-3, 3, n))


def _clean(nc, raw, **kw):
    ts = raw["timestamp"]
    return nc.clean_no2(raw, raw["parameter"].iloc[0], ts.min(), ts.max(), **kw)


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def test_negatives_dropped_and_zeros_kept(nc):
    v = _wiggle(48)
    v[5], v[9] = -1.0, 0.0
    df, rep = _clean(nc, _raw({1: v}))
    assert rep["out_of_range"] == 1
    assert rep["zeros"] == 1
    assert (df["value"] == 0).sum() == 1
    assert (df["value"] < 0).sum() == 0


def test_zeros_can_be_dropped_for_the_sensitivity(nc):
    v = _wiggle(48)
    v[9] = 0.0
    df, _ = _clean(nc, _raw({1: v}), drop_zeros=True)
    assert (df["value"] == 0).sum() == 0


def test_flagged_null_and_absurd_values_go(nc):
    v = _wiggle(48)
    v[3], v[4] = None, 5000.0
    flags = {1: [i == 7 for i in range(48)]}
    df, rep = _clean(nc, _raw({1: v}, flagged=flags))
    assert rep["null"] == 1 and rep["out_of_range"] == 1 and rep["flagged"] == 1
    assert len(df) == 45


def test_only_the_requested_parameter_is_cleaned(nc):
    raw = pd.concat([_raw({1: _wiggle(24)}, "no2"),
                     _raw({1: _wiggle(30, seed=1)}, "nox")])
    df, _ = nc.clean_no2(raw, "nox", raw["timestamp"].min(), raw["timestamp"].max())
    assert len(df) == 30


def test_saturation_is_a_maximum_shared_by_three_monitors(nc):
    df = pd.DataFrame({"station_id": [1, 1, 2, 2, 3, 3, 4, 4],
                       "value": [10, 500, 12, 500, 9, 500, 8, 80.0]})
    assert nc.saturation_values(df) == [500.0]
    assert nc.saturation_values(df[df["station_id"] != 3]) == []


def test_lone_spike_is_caught(nc):
    v = _wiggle(200)
    v[100] = 400.0
    df, rep = _clean(nc, _raw({1: v}))
    assert rep["lone_spike"] == 1
    assert 400.0 not in df["value"].to_numpy()


def test_a_sustained_local_plume_is_kept(nc):
    """Six high hours at one monitor while the others stay low: a local plume
    the spatial PM2.5 rule would delete, and the temporal rule must keep."""
    v = _wiggle(200)
    for i in range(100, 106):
        v[i] = 300.0 + i
    others = {s: _wiggle(200, seed=s) for s in (2, 3, 4)}
    df, rep = _clean(nc, _raw({1: v, **others}))
    assert rep["lone_spike"] == 0
    assert (df.loc[df["station_id"] == 1, "value"] > 300).sum() == 6


def test_spike_needs_enough_neighbours(nc):
    df = pd.DataFrame({
        "station_id": 1,
        "timestamp": pd.date_range(START, periods=3, freq="3h"),
        "value": [10.0, 400.0, 10.0]})
    assert not nc.lone_spikes(df).any()


def test_stuck_runs_are_masked(nc):
    v = _wiggle(100)
    v[20:50] = [42.0] * 30
    df, rep = _clean(nc, _raw({1: v}))
    assert rep["stuck_masked"] == 30
    assert len(df) == 70


def test_cleaning_is_scale_free_for_the_structure_measures(nc):
    """Every rule but the fixed 1000 ceiling is relative, so a unit error that
    multiplies every monitor alike changes nothing that is kept."""
    v = {s: _wiggle(200, seed=s) for s in (1, 2, 3)}
    v[1][50] = 300.0
    a, _ = _clean(nc, _raw(v))
    b, _ = _clean(nc, _raw({s: [x * 1.88 for x in vals] for s, vals in v.items()}))
    assert len(a) == len(b)
    np.testing.assert_allclose(a["value"].to_numpy() * 1.88, b["value"].to_numpy())


# --------------------------------------------------------------------------
# Like for like
# --------------------------------------------------------------------------

def test_matched_panels_share_exactly_the_same_station_hours(nc):
    idx = pd.date_range(START, periods=4, freq="h")
    a = pd.DataFrame({1: [1, np.nan, 3, 4], 2: [5, 6, 7, np.nan]}, index=idx)
    b = pd.DataFrame({1: [9, 9, np.nan, 9], 2: [9, 9, 9, 9]}, index=idx)
    ma, mb = nc.matched(a, b)
    assert (ma.notna() == mb.notna()).all().all()
    assert int(ma.notna().sum().sum()) == 5
    assert ma.loc[idx[0], 1] == 1 and mb.loc[idx[0], 1] == 9


def test_to_wide_aligns_to_the_reference_grid(nc):
    df = pd.DataFrame({"station_id": [2, 2],
                       "timestamp": pd.date_range(START, periods=2, freq="h"),
                       "value": [1.0, 2.0]})
    idx = pd.date_range(START, periods=3, freq="h")
    w = nc.to_wide(df, idx, [1, 2])
    assert list(w.columns) == [1, 2] and len(w) == 3
    assert w[1].isna().all() and w[2].notna().sum() == 2


# --------------------------------------------------------------------------
# Decision rules
# --------------------------------------------------------------------------

def _table(b2_no2=0.40, b2_pm=0.14, nf=0.0, mantel=0.5, gap=-0.5,
           gap_lo=-1.0, gap_hi=0.1):
    rows = []
    for panel, b2, g, lo, hi, nfv, mp in (
            ("no2", b2_no2, gap, gap_lo, gap_hi, nf, mantel),
            ("pm", b2_pm, -0.5, -0.9, -0.3, 0.06, 0.2)):
        for w in ("1h", "24h", "1w"):
            rows.append({"panel": panel, "window": w, "reliability_mean": b2,
                         "near_minus_far": nfv, "mantel_p": mp, "gap_pct": g,
                         "gap_pct_lo": lo, "gap_pct_hi": hi})
    return pd.DataFrame(rows)


def _diffs(b2=(0.26, 0.2, 0.3), nf=(0.0, -0.1, 0.1), gap=(0.0, -0.5, 0.5)):
    return {"b2_1h": b2, "b2_24h": b2, "b2_1w": b2, "near_far_1h": nf,
            "gap_pct_1h": gap, "gap_pct_24h": gap, "gap_pct_1w": gap}


def test_stability_alone_is_reading_r2_and_not_clearly_more(nc):
    r = nc.reading(_table(), _diffs(), "no2", "pm")
    assert r["S"] and not r["G"] and not r["P"]
    assert r["verdict"].startswith("R2")
    assert not r["clearly_more_structure"]


def test_stability_plus_prediction_is_a_partial_pass(nc):
    t = _table(gap=-3.0, gap_lo=-4.0, gap_hi=-2.0)
    r = nc.reading(t, _diffs(gap=(-2.5, -3.5, -1.5)), "no2", "pm")
    assert r["verdict"] == "R3: partial pass (P)" and r["clearly_more_structure"]


def test_everything_met_is_a_full_pass(nc):
    t = _table(nf=0.3, mantel=0.01, gap=-3.0, gap_lo=-4.0, gap_hi=-2.0)
    d = _diffs(nf=(0.25, 0.1, 0.4), gap=(-2.5, -3.5, -1.5))
    assert nc.reading(t, d, "no2", "pm")["verdict"] == "R1: full pass"


def test_a_significant_but_small_b2_gain_does_not_count(nc):
    """Interval above zero but under twice PM2.5's b2: not 'clearly more'."""
    r = nc.reading(_table(b2_no2=0.20), _diffs(b2=(0.06, 0.02, 0.1)), "no2", "pm")
    assert not r["S"] and r["verdict"].startswith("R4")


def test_a_small_idw_gain_misses_the_two_percent_floor(nc):
    t = _table(gap=-1.5, gap_lo=-2.0, gap_hi=-1.0)
    r = nc.reading(t, _diffs(gap=(-1.0, -1.5, -0.5)), "no2", "pm")
    assert not r["P"]


def test_distance_or_skill_without_stability_is_mixed(nc):
    t = _table(b2_no2=0.15, nf=0.3, mantel=0.01)
    d = _diffs(b2=(0.01, -0.05, 0.07), nf=(0.25, 0.1, 0.4))
    assert nc.reading(t, d, "no2", "pm")["verdict"] == "mixed"


def test_paired_interval_uses_the_same_resamples(nc):
    a = {"1h": {"k": np.arange(100, dtype=float) + 1.0}}
    b = {"1h": {"k": np.arange(100, dtype=float)}}
    lo, hi = nc.paired_diff(a, b, "1h", "k")
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
