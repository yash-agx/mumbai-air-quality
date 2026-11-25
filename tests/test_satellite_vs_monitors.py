import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "s15", ROOT / "scripts" / "15_satellite_vs_monitors.py")
s15 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s15)


def test_citywide_share_is_one_for_pure_day_signal():
    day = np.array([10.0, 30.0, 50.0])[:, None]
    assert s15.citywide_share(np.repeat(day, 4, axis=1)) == pytest.approx(1.0)


def test_citywide_share_is_zero_for_pure_place_signal():
    place = np.array([1.0, 2.0, 3.0, 4.0])[None, :]
    assert s15.citywide_share(np.repeat(place, 5, axis=0)) == pytest.approx(0.0)


def test_citywide_share_decomposition_is_exact_with_gaps():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(40, 6)) + rng.normal(size=(40, 1)) * 3
    Y[rng.random(Y.shape) < 0.2] = np.nan
    ok = np.isfinite(Y)
    xbar = Y[ok].mean()
    m = np.nanmean(Y, axis=1, keepdims=True)
    ssw = np.nansum((Y - m) ** 2)
    sst = np.sum((Y[ok] - xbar) ** 2)
    assert s15.citywide_share(Y) == pytest.approx(1 - ssw / sst)


def test_first_differences_skip_non_consecutive_days():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04"])
    Y = np.array([[1.0], [3.0], [10.0]])
    D, d = s15.first_differences(Y, dates)
    assert D[0, 0] == 2.0 and np.isnan(D[1, 0])
    assert list(d) == list(dates[1:])


@pytest.mark.parametrize("lo, hi, A, want", [
    (0.31, 0.6, 3.0, "supported"),
    (0.31, 0.6, 2.9, "counts against"),
    (0.10, 0.29, 5.0, "counts against"),
    (0.22, 0.43, 3.2, "inconclusive"),
])
def test_score_h1(lo, hi, A, want):
    assert s15.score_h1(lo, hi, A) == want


@pytest.mark.parametrize("cm, cml, cs, csl, want", [
    (0.73, 0.62, 0.95, 0.92, "supported"),
    (0.50, 0.40, 0.95, 0.92, "counts against"),
    (0.54, 0.48, 0.87, 0.84, "inconclusive"),
])
def test_score_h2(cm, cml, cs, csl, want):
    assert s15.score_h2(cm, cml, cs, csl) == want


@pytest.mark.parametrize("med, lo, want", [
    (0.70, 0.60, "supported"),
    (0.75, 0.59, "weak"),
    (0.59, 0.50, "counts against"),
])
def test_score_h3(med, lo, want):
    assert s15.score_h3(med, lo) == want
