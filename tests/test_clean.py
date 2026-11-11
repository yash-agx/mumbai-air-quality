"""Tests for data/clean.py -- the drop rules and the quality report they feed."""

import numpy as np
import pandas as pd
import pytest

from conftest import raw_rows


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def test_haversine_km_zero_for_same_point(clean_mod):
    assert clean_mod.haversine_km(19.0627, 72.84614, 19.0627, 72.84614) == 0.0


def test_haversine_km_matches_documented_bandra_pair(clean_mod):
    """The two Bandra East stations are ~70 m apart, per the CV group note."""
    d = clean_mod.haversine_km(19.0627, 72.84614, 19.06321, 72.84563)
    assert 0.060 < d < 0.080


def test_haversine_km_is_symmetric(clean_mod):
    a = clean_mod.haversine_km(19.0, 72.8, 19.2, 73.1)
    b = clean_mod.haversine_km(19.2, 73.1, 19.0, 72.8)
    assert a == pytest.approx(b)


@pytest.mark.parametrize("name, expected", [
    ("Kurla, Mumbai - MPCB", ("Kurla, Mumbai", "MPCB")),
    ("Bandra Kurla Complex - IITM", ("Bandra Kurla Complex", "IITM")),
    # Only the last " - " separates site from agency.
    ("Sion - Dharavi - MPCB", ("Sion - Dharavi", "MPCB")),
    # No separator at all: the whole string is the site, agency is empty.
    ("Mumbai", ("Mumbai", "")),
    ("  Worli  ", ("Worli", "")),
])
def test_split_name(clean_mod, name, expected):
    assert clean_mod.split_name(name) == expected


def test_run_lengths_counts_unbroken_identical_hours(clean_mod):
    df = raw_rows(1, "A - X", 19.0, 72.8, [41.0, 41.0, 41.0, 50.0, 50.0])
    lengths = clean_mod.run_lengths(df)
    assert list(lengths) == [3, 3, 3, 2, 2]


def test_run_lengths_breaks_a_run_across_a_missing_hour(clean_mod):
    """A station reading 41.0 either side of an outage was not stuck through it."""
    ts = ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z",
          "2024-01-05T00:00:00Z", "2024-01-05T01:00:00Z"]
    df = raw_rows(1, "A - X", 19.0, 72.8, [41.0] * 4, timestamps=ts)
    assert list(clean_mod.run_lengths(df)) == [2, 2, 2, 2]


def test_run_lengths_separates_stations_with_the_same_value(clean_mod):
    df = pd.concat([
        raw_rows(1, "A - X", 19.0, 72.8, [41.0, 41.0]),
        raw_rows(2, "B - Y", 19.1, 72.9, [41.0, 41.0]),
    ], ignore_index=True)
    assert list(clean_mod.run_lengths(df)) == [2, 2, 2, 2]


# --------------------------------------------------------------------------
# clean(): the drop rules, one at a time
# --------------------------------------------------------------------------

def test_clean_keeps_a_clean_pull_intact(clean_mod, two_station_raw):
    df, rep = clean_mod.clean(two_station_raw)
    assert rep["rows_in"] == 96
    assert rep["rows_out"] == 96
    assert rep["excluded"] == 0
    assert rep["flagged"] == 0
    assert rep["null"] == 0
    assert rep["out_of_range"] == 0
    assert rep["duplicate_hours"] == 0
    assert rep["stuck_masked"] == 0
    assert set(df["station_id"]) == {1001, 1002}


def test_clean_output_columns(clean_mod, two_station_raw):
    df, _ = clean_mod.clean(two_station_raw)
    assert set(df.columns) == {"station_id", "timestamp", "station_name", "lat",
                               "lon", "pollutant", "value", "site", "agency",
                               "cv_group"}
    # flagged is consumed, not carried.
    assert "flagged" not in df.columns
    assert str(df["timestamp"].dtype).endswith("UTC]")
    assert df.index.equals(pd.RangeIndex(len(df)))


def test_clean_drops_the_excluded_station(clean_mod, make_raw, two_station_raw):
    """Station 8039 is dropped outright -- see EXCLUDE_STATIONS."""
    assert 8039 in clean_mod.EXCLUDE_STATIONS
    raw = make_raw(two_station_raw,
                   raw_rows(8039, "Mumbai", 19.07283, 72.88261, [25.0] * 4))
    df, rep = clean_mod.clean(raw)
    assert rep["excluded"] == 4
    assert rep["excluded_ids"] == [8039]
    assert 8039 not in set(df["station_id"])


def test_clean_drops_qa_flagged_rows(clean_mod, make_raw, two_station_raw):
    flagged = raw_rows(1003, "Gamma - MPCB", 19.15, 72.90, [22.0, 23.0, 24.0],
                       flagged=[True, False, True])
    df, rep = clean_mod.clean(make_raw(two_station_raw, flagged))
    assert rep["flagged"] == 2
    assert list(df.loc[df["station_id"] == 1003, "value"]) == [23.0]


def test_clean_drops_null_values(clean_mod, make_raw, two_station_raw):
    nulls = raw_rows(1003, "Gamma - MPCB", 19.15, 72.90, [22.0, None, None])
    df, rep = clean_mod.clean(make_raw(two_station_raw, nulls))
    assert rep["null"] == 2
    assert len(df[df["station_id"] == 1003]) == 1


@pytest.mark.parametrize("value, dropped", [
    (-999.0, True),     # OpenAQ's sentinel
    (0.0, True),        # the range is exclusive at the bottom
    (0.1, False),
    (999.9, False),
    (1000.0, False),    # inclusive at the top
    (1000.1, True),
])
def test_clean_range_boundaries(clean_mod, make_raw, two_station_raw, value, dropped):
    """What the range rule itself removes.

    Survival is not asserted here: an in-range reading above SPIKE_MIN still has
    the artefact screen to get past, which is a different rule and is tested on
    its own below.
    """
    assert clean_mod.VALID_RANGE == (0.0, 1000.0)
    one = raw_rows(1003, "Gamma - MPCB", 19.15, 72.90, [value])
    _, rep = clean_mod.clean(make_raw(two_station_raw, one))
    assert rep["out_of_range"] == int(dropped)


def test_clean_keeps_an_ordinary_in_range_reading(clean_mod, make_raw,
                                                  two_station_raw):
    one = raw_rows(1003, "Gamma - MPCB", 19.15, 72.90, [0.1])
    df, rep = clean_mod.clean(make_raw(two_station_raw, one))
    assert rep["out_of_range"] == 0
    assert list(df.loc[df["station_id"] == 1003, "value"]) == [0.1]


def test_clean_floors_timestamps_to_the_hour(clean_mod, make_raw, two_station_raw):
    """OpenAQ stamps some series at :15 or :30 past."""
    odd = raw_rows(1003, "Gamma - MPCB", 19.15, 72.90, [22.0, 23.0],
                   timestamps=["2024-02-01T05:15:00Z", "2024-02-01T06:30:00Z"])
    df, _ = clean_mod.clean(make_raw(two_station_raw, odd))
    got = df.loc[df["station_id"] == 1003, "timestamp"]
    assert list(got) == [pd.Timestamp("2024-02-01T05:00:00Z"),
                         pd.Timestamp("2024-02-01T06:00:00Z")]


def test_clean_averages_duplicate_station_hours(clean_mod, make_raw, two_station_raw):
    """Two overlapping sensors at one station yield two rows for one hour."""
    dup = raw_rows(1003, "Gamma - MPCB", 19.15, 72.90, [20.0, 30.0],
                   timestamps=["2024-02-01T05:10:00Z", "2024-02-01T05:40:00Z"])
    df, rep = clean_mod.clean(make_raw(two_station_raw, dup))
    assert rep["duplicate_hours"] == 1
    row = df[df["station_id"] == 1003]
    assert len(row) == 1
    assert row["value"].iloc[0] == 25.0


def test_clean_splits_site_and_agency(clean_mod, two_station_raw):
    df, _ = clean_mod.clean(two_station_raw)
    alpha = df[df["station_id"] == 1001].iloc[0]
    assert (alpha["site"], alpha["agency"]) == ("Alpha, Mumbai", "MPCB")
    beta = df[df["station_id"] == 1002].iloc[0]
    assert (beta["site"], beta["agency"]) == ("Beta, Mumbai", "IITM")


def test_clean_assigns_cv_groups(clean_mod, make_raw, two_station_raw):
    """Grouped ids get the shared label; everything else gets its own id."""
    pair = make_raw(
        raw_rows(3409328, "Bandra Kurla Complex - IITM", 19.0627, 72.84614, [30.0] * 3),
        raw_rows(3409486, "Kherwadi_Bandra East - MPCB", 19.06321, 72.84563, [31.0] * 3),
    )
    df, _ = clean_mod.clean(make_raw(two_station_raw, pair))
    groups = df.drop_duplicates("station_id").set_index("station_id")["cv_group"]
    assert groups[3409328] == "bandra_east"
    assert groups[3409486] == "bandra_east"
    # Ungrouped stations fall back to their own id as a string, so holding out
    # a cv_group still holds out exactly one station.
    assert groups[1001] == "1001"
    assert groups[1002] == "1002"


# --------------------------------------------------------------------------
# Artefact screen
# --------------------------------------------------------------------------

def test_artefact_screen_drops_the_saturation_ceiling(clean_mod):
    df = pd.DataFrame({
        "station_id": [1, 1, 2],
        "timestamp": pd.to_datetime(["2024-01-01T00:00Z"] * 3, utc=True),
        "value": [985.0, 30.0, 985.0],
    })
    out, rep = clean_mod.artefact_screen(df)
    assert rep["saturation"] == 2
    assert rep["saturation_stations"] == 2
    assert list(out["value"]) == [30.0]


def test_artefact_screen_drops_a_lone_spike(clean_mod):
    """One station at 690 while the rest of the city sits at ~23 is a sensor."""
    hour = pd.Timestamp("2024-01-01T00:00Z")
    df = pd.DataFrame({
        "station_id": [1, 2, 3, 4],
        "timestamp": [hour] * 4,
        "value": [690.0, 23.0, 24.0, 22.0],
    })
    out, rep = clean_mod.artefact_screen(df)
    assert rep["above_spike_min"] == 1
    assert rep["lone_spike"] == 1
    assert rep["lone_spike_stations"] == 1
    assert 690.0 not in set(out["value"])


def test_artefact_screen_keeps_a_corroborated_episode(clean_mod):
    """A high reading the whole city agrees with is weather, and is kept."""
    hour = pd.Timestamp("2024-01-01T00:00Z")
    df = pd.DataFrame({
        "station_id": [1, 2, 3, 4],
        "timestamp": [hour] * 4,
        "value": [690.0, 300.0, 280.0, 310.0],
    })
    out, rep = clean_mod.artefact_screen(df)
    assert rep["lone_spike"] == 0
    assert len(out) == 4
    assert len(rep["survivors"]) == 4


def test_artefact_screen_keeps_what_it_cannot_check(clean_mod):
    """No other station reporting that hour: keep it and count it as unchecked."""
    df = pd.DataFrame({
        "station_id": [1],
        "timestamp": pd.to_datetime(["2024-01-01T00:00Z"], utc=True),
        "value": [690.0],
    })
    out, rep = clean_mod.artefact_screen(df)
    assert rep["uncorroborated"] == 1
    assert rep["lone_spike"] == 0
    assert list(out["value"]) == [690.0]


def test_artefact_screen_with_no_candidates_reports_zeros(clean_mod):
    df = pd.DataFrame({
        "station_id": [1, 2],
        "timestamp": pd.to_datetime(["2024-01-01T00:00Z"] * 2, utc=True),
        "value": [20.0, 30.0],
    })
    out, rep = clean_mod.artefact_screen(df)
    assert rep["above_spike_min"] == 0
    assert rep["lone_spike"] == 0
    assert rep["uncorroborated"] == 0
    assert rep["survivors"].empty
    assert len(out) == 2


def test_other_station_median_leaves_the_candidate_out(clean_mod):
    hour = pd.Timestamp("2024-01-01T00:00Z")
    df = pd.DataFrame({
        "station_id": [1, 2, 3],
        "timestamp": [hour] * 3,
        "value": [500.0, 10.0, 30.0],
    })
    cand = df[df["value"] > clean_mod.SPIKE_MIN]
    # Median of {10, 30}, not of {500, 10, 30}.
    assert clean_mod.other_station_median(df, cand) == pytest.approx([20.0])


def test_other_station_median_is_nan_with_no_neighbour(clean_mod):
    df = pd.DataFrame({
        "station_id": [1],
        "timestamp": pd.to_datetime(["2024-01-01T00:00Z"], utc=True),
        "value": [500.0],
    })
    assert np.isnan(clean_mod.other_station_median(df, df)).all()


def test_clean_screens_saturation_before_the_stuck_mask(clean_mod, make_raw,
                                                        two_station_raw):
    """A sensor pinned at 985.0 is reported as saturation, not as a stuck run."""
    pinned = raw_rows(1003, "Gamma - MPCB", 19.15, 72.90, [985.0] * 48)
    _, rep = clean_mod.clean(make_raw(two_station_raw, pinned))
    assert rep["artefact"]["saturation"] == 48
    assert rep["stuck_masked"] == 0


# --------------------------------------------------------------------------
# Stuck-run mask
# --------------------------------------------------------------------------

def test_clean_masks_a_stuck_run_at_the_threshold(clean_mod, make_raw,
                                                  two_station_raw):
    assert clean_mod.STUCK_RUN_HOURS == 24
    stuck = raw_rows(1003, "Gamma - MPCB", 19.15, 72.90, [41.0] * 24)
    df, rep = clean_mod.clean(make_raw(two_station_raw, stuck))
    assert rep["stuck_masked"] == 24
    assert df[df["station_id"] == 1003].empty


def test_clean_keeps_a_run_one_hour_short(clean_mod, make_raw, two_station_raw):
    short = raw_rows(1003, "Gamma - MPCB", 19.15, 72.90, [41.0] * 23)
    df, rep = clean_mod.clean(make_raw(two_station_raw, short))
    assert rep["stuck_masked"] == 0
    assert len(df[df["station_id"] == 1003]) == 23


def test_clean_keeps_the_station_when_it_masks_a_run(clean_mod, make_raw,
                                                    two_station_raw):
    """The masked hours become missing hours; the rest of the series survives."""
    series = [41.0] * 30 + list(50.0 + np.arange(10))
    partial = raw_rows(1003, "Gamma - MPCB", 19.15, 72.90, series)
    df, rep = clean_mod.clean(make_raw(two_station_raw, partial))
    assert rep["stuck_masked"] == 30
    kept = df[df["station_id"] == 1003]
    assert len(kept) == 10
    assert 41.0 not in set(kept["value"])


def test_stuck_cost_tallies_per_station(clean_mod, make_raw, two_station_raw):
    stuck = raw_rows(1003, "Gamma - MPCB", 19.15, 72.90,
                     [41.0] * 25 + [12.0] * 5)
    _, rep = clean_mod.clean(make_raw(two_station_raw, stuck))
    cost = rep["stuck"]
    assert list(cost["station_id"]) == [1003]
    row = cost.iloc[0]
    assert row["masked_h"] == 25
    assert row["longest_run_h"] == 25
    assert row["of_obs"] == 30
    assert row["pct"] == pytest.approx(83.3, abs=0.1)


def test_stuck_cost_is_empty_when_nothing_is_masked(clean_mod, two_station_raw):
    _, rep = clean_mod.clean(two_station_raw)
    assert rep["stuck"].empty
    assert list(rep["stuck"].columns) == ["station_id", "site", "masked_h",
                                          "longest_run_h", "of_obs", "pct"]


# --------------------------------------------------------------------------
# per_station() and proximity_check()
# --------------------------------------------------------------------------

def test_per_station_summary(clean_mod, two_station_raw):
    df, _ = clean_mod.clean(two_station_raw)
    stations, window_hours = clean_mod.per_station(df)
    assert window_hours == 48
    assert set(stations["station_id"]) == {1001, 1002}
    alpha = stations.set_index("station_id").loc[1001]
    assert alpha["n_obs"] == 48
    assert alpha["coverage_pct"] == 100.0
    assert alpha["max_gap_h"] == 0
    assert alpha["site"] == "Alpha, Mumbai"
    assert alpha["agency"] == "MPCB"
    assert alpha["cv_group"] == "1001"


def test_per_station_measures_the_longest_outage(clean_mod, make_raw,
                                                 two_station_raw):
    """A station reporting at hour 0 then hour 72 was absent for 71 hours."""
    gappy = raw_rows(1003, "Gamma - MPCB", 19.15, 72.90, [20.0, 21.0],
                     timestamps=["2024-01-01T00:00:00Z", "2024-01-04T00:00:00Z"])
    df, _ = clean_mod.clean(make_raw(two_station_raw, gappy))
    stations, _ = clean_mod.per_station(df)
    assert stations.set_index("station_id").loc[1003, "max_gap_h"] == 71


def test_per_station_coverage_is_a_share_of_the_whole_window(clean_mod, make_raw,
                                                             two_station_raw):
    sparse = raw_rows(1003, "Gamma - MPCB", 19.15, 72.90, [20.0] * 12)
    df, _ = clean_mod.clean(make_raw(two_station_raw, sparse))
    stations, window_hours = clean_mod.per_station(df)
    assert window_hours == 48
    assert stations.set_index("station_id").loc[1003, "coverage_pct"] == 25.0


def test_per_station_is_sorted_by_coverage(clean_mod, make_raw, two_station_raw):
    sparse = raw_rows(1003, "Gamma - MPCB", 19.15, 72.90, [20.0] * 12)
    df, _ = clean_mod.clean(make_raw(two_station_raw, sparse))
    stations, _ = clean_mod.per_station(df)
    assert list(stations["coverage_pct"]) == sorted(stations["coverage_pct"],
                                                    reverse=True)


def test_proximity_check_flags_an_ungrouped_close_pair(clean_mod, make_raw):
    """~70 m apart with no cv_group between them is what this warns about."""
    raw = make_raw(
        raw_rows(9001, "Near A - MPCB", 19.0627, 72.84614, [20.0] * 3),
        raw_rows(9002, "Near B - IITM", 19.06321, 72.84563, [21.0] * 3),
    )
    df, _ = clean_mod.clean(raw)
    stations, _ = clean_mod.per_station(df)
    warn = clean_mod.proximity_check(stations)
    assert len(warn) == 1
    d, _site_a, id_a, _site_b, id_b = warn[0]
    assert d < clean_mod.PROXIMITY_WARN_KM
    assert {id_a, id_b} == {9001, 9002}


def test_proximity_check_ignores_pairs_already_grouped(clean_mod, make_raw):
    raw = make_raw(
        raw_rows(3409328, "Bandra Kurla Complex - IITM", 19.0627, 72.84614, [20.0] * 3),
        raw_rows(3409486, "Kherwadi_Bandra East - MPCB", 19.06321, 72.84563, [21.0] * 3),
    )
    df, _ = clean_mod.clean(raw)
    stations, _ = clean_mod.per_station(df)
    assert clean_mod.proximity_check(stations) == []


def test_proximity_check_ignores_distant_pairs(clean_mod, two_station_raw):
    df, _ = clean_mod.clean(two_station_raw)
    stations, _ = clean_mod.per_station(df)
    assert clean_mod.proximity_check(stations) == []


def test_cv_group_table_is_keyed_by_int_station_ids(clean_mod):
    """map() is given int ids, which is what the station_id column holds."""
    assert all(isinstance(k, int) for k in clean_mod.CV_STATION_GROUPS)
    assert set(clean_mod.CV_STATION_GROUPS.values()) == {
        "bandra_east", "borivali_east", "worli", "ulhasnagar"}
