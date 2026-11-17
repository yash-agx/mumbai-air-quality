"""The pure helpers behind the breeze and swing tests.

No network and no data files: the Open-Meteo call is never made here, and the
statistics come from hand-built arrays whose answer is known by construction.
What these guard is the handful of places where a quiet mistake would produce
a plausible-looking result - the wind sectors, the residualising that stops the
test re-measuring the diurnal cycle, and the swing statistic whose denominator
is the whole reason test 2 is a split-half.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def breeze(script_module):
    return script_module("06_breeze.py", "breeze")


@pytest.fixture(scope="module")
def swing(script_module):
    return script_module("07_baseline_swing.py", "baseline_swing")


# --------------------------------------------------------------- wind sectors

def _wind(dirs, speed=5.0):
    return pd.DataFrame({"wind_dir_deg": np.asarray(dirs, float),
                         "wind_speed_ms": np.full(len(dirs), speed, float)})


def test_westerlies_are_onshore_and_easterlies_offshore(breeze):
    """The sea is west of Mumbai and direction is where wind comes FROM."""
    k = breeze.classify(_wind([270, 230, 310]))
    assert list(k) == ["onshore"] * 3
    k = breeze.classify(_wind([90, 50, 130]))
    assert list(k) == ["offshore"] * 3


def test_along_shore_winds_are_not_classified(breeze):
    """North and south winds are neither breeze; they must not be forced."""
    k = breeze.classify(_wind([0, 180, 20, 200, 350]))
    assert set(k) == {"unclassified"}


def test_calm_hours_are_dropped_whatever_the_direction(breeze):
    """Below the speed floor the direction is noise, not a breeze."""
    k = breeze.classify(_wind([270, 90], speed=0.2))
    assert list(k) == ["unclassified", "unclassified"]


def test_missing_wind_is_unclassified(breeze):
    k = breeze.classify(_wind([np.nan, 270]))
    assert k[0] == "unclassified"
    assert k[1] == "onshore"


# ---------------------------------------------------------- hour residualising

def test_residualising_removes_a_pure_diurnal_cycle(breeze):
    """A monitor that only ever repeats its daily shape must residual to zero.

    This is the one that matters: offshore hours are mostly night, so without
    this the breeze test would re-measure the diurnal cycle and call it wind.
    """
    hours = np.tile(np.arange(24), 30)
    cycle = 20 + 10 * np.sin(2 * np.pi * hours / 24)
    values = cycle.reshape(-1, 1)
    out = breeze.residualise(values, hours)
    assert np.allclose(out, 0.0, atol=1e-9)


def test_residualising_keeps_a_departure_from_the_usual_hour(breeze):
    hours = np.tile(np.arange(24), 30)
    values = (20 + 10 * np.sin(2 * np.pi * hours / 24)).reshape(-1, 1)
    values[5, 0] += 7.0                      # one unusual hour
    out = breeze.residualise(values, hours)
    assert out[5, 0] > 6.0                   # survives, minus its own share
    others = np.delete(out[:, 0], 5)
    assert np.abs(others).max() < 1.0


def test_block_residualising_also_removes_a_season(breeze):
    """The exploratory variant: month blocks take the dirty season out too."""
    hours = np.tile(np.arange(24), 60)
    months = np.where(np.arange(len(hours)) < len(hours) // 2, "2025-12",
                      "2026-06")
    values = (20 + 10 * np.sin(2 * np.pi * hours / 24)).reshape(-1, 1)
    values[:len(hours) // 2] += 30.0         # winter is dirtier everywhere
    plain = breeze.residualise(values, hours)
    blocked = breeze.residualise(values, hours, months)
    assert np.abs(plain).max() > 10.0        # the season survives hour-only
    assert np.allclose(blocked, 0.0, atol=1e-9)


# -------------------------------------------------------------- swing statistic

def test_percentage_swing_is_relative_to_the_mean(swing):
    flat = np.full(24, 20.0)
    assert swing.pct_swing(flat) == pytest.approx(0.0)
    profile = np.where(np.arange(24) < 12, 10.0, 30.0)   # mean 20, range 20
    assert swing.pct_swing(profile) == pytest.approx(100.0)


def test_the_same_shape_on_a_lower_baseline_scores_higher(swing):
    """Why test 2 needs a split-half: halve the level, double the percentage.

    Both profiles have the same absolute swing. The percentage statistic ranks
    the quieter monitor as swingier purely because the mean is its denominator.
    """
    high = np.where(np.arange(24) < 12, 30.0, 50.0)      # mean 40, range 20
    low = np.where(np.arange(24) < 12, 10.0, 30.0)       # mean 20, range 20
    assert swing.abs_swing(high) == swing.abs_swing(low)
    # Exactly double, because the only thing that changed is the denominator.
    assert swing.pct_swing(low) == pytest.approx(2 * swing.pct_swing(high))
    assert swing.pct_swing(low) == pytest.approx(100.0)
    assert swing.pct_swing(high) == pytest.approx(50.0)


def test_swing_is_nan_when_the_mean_is_not_usable(swing):
    assert np.isnan(swing.pct_swing(np.full(24, np.nan)))
    assert np.isnan(swing.pct_swing(np.zeros(24)))
