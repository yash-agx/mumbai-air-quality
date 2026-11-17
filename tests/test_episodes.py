"""The episode machinery behind scripts/10_episodes.py.

The whole episode analysis turns on one judgement: a run of bad hours is ONE
observation, not twenty. These check the code that makes that judgement, on
hand-built timestamps whose answer is known by construction. No data files and
no cross-validation here.
"""

import numpy as np
import pandas as pd
import pytest

TZ = "Asia/Kolkata"


@pytest.fixture(scope="module")
def ep(script_module):
    return script_module("10_episodes.py", "episodes")


def _hours(*specs):
    """Timestamps from ("2025-12-01 00:00", n_hours) pairs."""
    out = []
    for start, n in specs:
        out.extend(pd.date_range(pd.Timestamp(start, tz=TZ), periods=n,
                                 freq="h"))
    return pd.DatetimeIndex(out)


def test_one_unbroken_run_is_one_episode(ep):
    labels = ep.episodes_of(_hours(("2025-12-01 00:00", 30)))
    assert labels.nunique() == 1
    assert len(labels) == 30


def test_a_long_gap_splits_episodes(ep):
    """Two spells a week apart are two observations, not one."""
    labels = ep.episodes_of(_hours(("2025-12-01 00:00", 12),
                                   ("2025-12-08 00:00", 12)))
    assert labels.nunique() == 2


def test_a_short_gap_is_bridged(ep):
    """A few quiet hours inside a spell do not end it.

    The bridge is 6 hours; a 4-hour dip keeps one episode, which is the point
    of merging - otherwise one weather system would be counted several times
    and every interval would come out too narrow.
    """
    labels = ep.episodes_of(_hours(("2025-12-01 00:00", 10),
                                   ("2025-12-01 14:00", 10)))
    assert labels.nunique() == 1


def test_a_gap_just_past_the_bridge_splits(ep):
    labels = ep.episodes_of(_hours(("2025-12-01 00:00", 10),
                                   ("2025-12-01 17:00", 10)))
    assert labels.nunique() == 2


def test_episode_labels_are_ordered_in_time(ep):
    labels = ep.episodes_of(_hours(("2025-12-08 00:00", 5),
                                   ("2025-12-01 00:00", 5)))
    assert list(labels.index) == sorted(labels.index)
    assert labels.iloc[0] == 0


def test_rmse_is_the_root_mean_square(ep):
    assert ep.rmse(np.array([3.0, 4.0])) == pytest.approx(3.5355339)
    assert ep.rmse(np.zeros(5)) == 0.0


def test_citywide_drops_hours_with_too_few_monitors(ep):
    """An hour carried by three monitors is not a city-wide average."""
    idx = pd.date_range("2025-12-01", periods=3, freq="h", tz="UTC")
    wide = pd.DataFrame(np.full((3, 12), 50.0), index=idx,
                        columns=range(12))
    wide.iloc[1, 3:] = np.nan          # only 3 monitors reporting that hour
    city = ep.citywide(wide)
    assert len(city) == 2
    assert idx[1].tz_convert(TZ) not in city.index
    assert city.iloc[0] == pytest.approx(50.0)
