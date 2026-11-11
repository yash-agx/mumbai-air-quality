"""Tests for data/live.py -- the AQI conversion and the two-source fallback.

No network: requests.get is replaced for every test that reaches a feed, and
the credentials are passed in rather than read from .env.
"""

from types import SimpleNamespace

import pandas as pd
import pytest
import requests


# --------------------------------------------------------------------------
# HTTP doubles
# --------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = "fake response"

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


@pytest.fixture
def http(live_mod, monkeypatch):
    """Route requests.get by URL fragment, and record what was called.

    A URL no handler claims is an error rather than a silent miss, so a test
    that expects the fallback not to be reached fails loudly if it is.
    """
    calls, handlers = [], {}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append((url, params or {}))
        for fragment, handler in handlers.items():
            if fragment in url:
                if isinstance(handler, Exception):
                    raise handler
                return handler(params) if callable(handler) else handler
        raise AssertionError(f"unexpected request to {url}")

    monkeypatch.setattr(live_mod.requests, "get", fake_get)
    return SimpleNamespace(calls=calls, handlers=handlers,
                           datagov=lambda: [u for u, _ in calls
                                            if "data.gov.in" in u],
                           waqi=lambda: [u for u, _ in calls if "waqi" in u])


@pytest.fixture
def stations():
    """Three of our monitors, far enough apart to match unambiguously."""
    return pd.DataFrame({
        "station_id": [1001, 1002, 1003],
        "lat": [19.00, 19.10, 19.20],
        "lon": [72.82, 72.88, 72.90],
    })


def datagov_payload(records):
    return FakeResponse({"records": records})


def datagov_record(lat, lon, value, stamp=None, pollutant="PM2.5"):
    stamp = stamp or pd.Timestamp.now(tz="Asia/Kolkata").strftime("%d-%m-%Y %H:%M:%S")
    return {"pollutant_id": pollutant, "pollutant_avg": str(value),
            "latitude": str(lat), "longitude": str(lon), "last_update": stamp}


def waqi_payload(rows, status="ok"):
    return FakeResponse({"status": status, "data": rows})


def waqi_row(lat, lon, aqi, stamp=None, name="Some Station"):
    stamp = stamp or pd.Timestamp.now(tz="Asia/Kolkata").isoformat()
    return {"lat": lat, "lon": lon, "uid": 1234, "aqi": str(aqi),
            "station": {"name": name, "time": stamp}}


# --------------------------------------------------------------------------
# AQI -> ug/m3
# --------------------------------------------------------------------------

@pytest.mark.parametrize("aqi, ugm3", [
    # Both edges of every band in EPA_PM25_BREAKPOINTS.
    (0, 0.0), (50, 12.0),
    (51, 12.1), (100, 35.4),
    (101, 35.5), (150, 55.4),
    (151, 55.5), (200, 150.4),
    (201, 150.5), (300, 250.4),
    (301, 250.5), (400, 350.4),
    (401, 350.5), (500, 500.4),
])
def test_aqi_to_ugm3_hits_every_breakpoint(live_mod, aqi, ugm3):
    assert live_mod.aqi_to_ugm3(aqi) == pytest.approx(ugm3)


@pytest.mark.parametrize("aqi, ugm3", [
    (25, 6.0),      # 0.0 + 25 * 12.0/50
    (75, 23.5),     # 12.1 + 24 * 23.3/49
    (125, 45.2),    # 35.5 + 24 * 19.9/49
    (175, 102.0),   # 55.5 + 24 * 94.9/49
])
def test_aqi_to_ugm3_interpolates_inside_a_band(live_mod, aqi, ugm3):
    assert live_mod.aqi_to_ugm3(aqi) == pytest.approx(ugm3, abs=0.05)


def test_aqi_to_ugm3_is_monotone(live_mod):
    out = [live_mod.aqi_to_ugm3(a) for a in range(0, 501)]
    assert all(b >= a for a, b in zip(out, out[1:]))


def test_aqi_to_ugm3_matches_the_epa_formula(live_mod):
    """Independent recomputation of the piecewise-linear inverse."""
    for lo_i, hi_i, lo_c, hi_c in live_mod.EPA_PM25_BREAKPOINTS:
        for aqi in (lo_i, (lo_i + hi_i) // 2, hi_i):
            expected = lo_c + (aqi - lo_i) * (hi_c - lo_c) / (hi_i - lo_i)
            assert live_mod.aqi_to_ugm3(aqi) == pytest.approx(expected, abs=0.05)


def test_aqi_to_ugm3_clamps_above_the_scale(live_mod):
    """The AQI scale stops at 500; nothing is extrapolated past the table."""
    top = live_mod.EPA_PM25_BREAKPOINTS[-1][3]
    assert live_mod.aqi_to_ugm3(501) == top
    assert live_mod.aqi_to_ugm3(9999) == top


@pytest.mark.parametrize("aqi", ["-", "", None, "nope", -5, float("nan")])
def test_aqi_to_ugm3_rejects_unusable_values(live_mod, aqi):
    """WAQI reports "-" for a station with no current reading."""
    assert live_mod.aqi_to_ugm3(aqi) is None


def test_aqi_to_ugm3_accepts_strings(live_mod):
    """map/bounds sends the AQI as a string."""
    assert live_mod.aqi_to_ugm3("151") == pytest.approx(55.5)


def test_breakpoint_table_is_contiguous_and_ordered(live_mod):
    bands = live_mod.EPA_PM25_BREAKPOINTS
    assert bands[0][0] == 0
    assert bands[-1][1] == 500
    for (_, hi_i, _, hi_c), (lo_i2, _, lo_c2, _) in zip(bands, bands[1:]):
        assert lo_i2 == hi_i + 1
        assert lo_c2 > hi_c


# --------------------------------------------------------------------------
# parse_waqi()
# --------------------------------------------------------------------------

def test_parse_waqi_matches_by_coordinates_and_converts(live_mod, stations):
    rows = [waqi_row(19.001, 72.821, 151), waqi_row(19.101, 72.881, 75)]
    readings, observed, unmatched = live_mod.parse_waqi(rows, stations)
    assert dict(readings) == {1001: pytest.approx(55.5), 1002: pytest.approx(23.5)}
    assert unmatched == 0
    assert observed is not None


def test_parse_waqi_ignores_stations_beyond_match_km(live_mod, stations):
    """Same rule as the primary feed: MATCH_KM or it is not one of ours."""
    far = waqi_row(19.60, 73.60, 120)
    readings, _, unmatched = live_mod.parse_waqi([far], stations)
    assert readings.empty
    assert unmatched == 1


def test_parse_waqi_drops_rows_without_a_usable_reading(live_mod, stations):
    rows = [waqi_row(19.001, 72.821, "-"), waqi_row(19.101, 72.881, 101)]
    readings, _, unmatched = live_mod.parse_waqi(rows, stations)
    assert dict(readings) == {1002: pytest.approx(35.5)}
    assert unmatched == 0


def test_parse_waqi_drops_rows_without_coordinates(live_mod, stations):
    rows = [{"aqi": "80", "station": {"time": "2026-11-11T14:00:00+05:30"}}]
    readings, _, _ = live_mod.parse_waqi(rows, stations)
    assert readings.empty


def test_parse_waqi_keeps_the_first_of_two_rows_on_one_monitor(live_mod, stations):
    rows = [waqi_row(19.000, 72.820, 101), waqi_row(19.002, 72.822, 300)]
    readings, _, _ = live_mod.parse_waqi(rows, stations)
    assert dict(readings) == {1001: pytest.approx(35.5)}


def test_parse_waqi_reads_the_newest_timestamp_in_utc(live_mod, stations):
    rows = [waqi_row(19.001, 72.821, 101, stamp="2026-11-11T14:00:00+05:30"),
            waqi_row(19.101, 72.881, 101, stamp="2026-11-11T15:00:00+05:30")]
    _, observed, _ = live_mod.parse_waqi(rows, stations)
    assert observed == pd.Timestamp("2026-11-11T09:30:00Z")


def test_parse_waqi_survives_an_unparseable_timestamp(live_mod, stations):
    rows = [waqi_row(19.001, 72.821, 101, stamp="whenever")]
    readings, observed, _ = live_mod.parse_waqi(rows, stations)
    assert dict(readings) == {1001: pytest.approx(35.5)}
    assert observed is None


def test_parse_waqi_on_an_empty_response(live_mod, stations):
    readings, observed, unmatched = live_mod.parse_waqi([], stations)
    assert readings.empty
    assert observed is None
    assert unmatched == 0


# --------------------------------------------------------------------------
# Fallback order
# --------------------------------------------------------------------------

def test_primary_wins_and_the_fallback_is_never_called(live_mod, stations, http):
    http.handlers["data.gov.in"] = datagov_payload([
        datagov_record(19.001, 72.821, 41.0), datagov_record(19.101, 72.881, 52.0)])

    res = live_mod.fetch_live(stations, key="k", token="t")
    assert res.ok
    assert res.source == live_mod.SOURCE_DATAGOV
    assert dict(res.readings) == {1001: 41.0, 1002: 52.0}
    assert res.attribution is None
    assert http.waqi() == []


@pytest.mark.parametrize("failure, handler", [
    ("unreachable", requests.exceptions.ConnectionError("blocked")),
    ("timeout", requests.exceptions.ReadTimeout("slow")),
    ("unauthorised key", FakeResponse({}, status_code=403)),
    ("server error", FakeResponse({}, status_code=502)),
    ("unreadable body", FakeResponse(ValueError("not json"))),
    ("no PM2.5 at all", datagov_payload([])),
])
def test_waqi_takes_over_whenever_the_primary_fails(live_mod, stations, http,
                                                    failure, handler):
    """Streamlit Cloud sees the first of these: data.gov.in refuses its IP."""
    http.handlers["data.gov.in"] = handler
    http.handlers["waqi"] = waqi_payload([waqi_row(19.001, 72.821, 151)])

    res = live_mod.fetch_live(stations, key="k", token="t")
    assert res.ok, f"{failure}: {res.error}"
    assert res.source == live_mod.SOURCE_WAQI
    assert dict(res.readings) == {1001: pytest.approx(55.5)}
    assert res.n_reporting == 1
    assert res.n_total == 3
    assert http.waqi()


def test_waqi_takes_over_when_the_primary_is_stale(live_mod, stations, http):
    old = (pd.Timestamp.now(tz="Asia/Kolkata") - pd.Timedelta(hours=30))
    http.handlers["data.gov.in"] = datagov_payload([
        datagov_record(19.001, 72.821, 41.0,
                       stamp=old.strftime("%d-%m-%Y %H:%M:%S"))])
    http.handlers["waqi"] = waqi_payload([waqi_row(19.101, 72.881, 101)])

    res = live_mod.fetch_live(stations, key="k", token="t")
    assert res.ok
    assert res.source == live_mod.SOURCE_WAQI
    assert dict(res.readings) == {1002: pytest.approx(35.5)}


def test_waqi_is_used_when_no_primary_key_is_configured(live_mod, stations, http,
                                                        monkeypatch):
    monkeypatch.setattr(live_mod, "api_key", lambda: None)
    http.handlers["waqi"] = waqi_payload([waqi_row(19.001, 72.821, 151)])

    res = live_mod.fetch_live(stations, token="t")
    assert res.ok
    assert res.source == live_mod.SOURCE_WAQI
    assert http.datagov() == []


def test_waqi_result_carries_the_attribution(live_mod, stations, http):
    """WAQI's terms require it wherever the data is shown."""
    http.handlers["data.gov.in"] = requests.exceptions.ConnectionError("blocked")
    http.handlers["waqi"] = waqi_payload([waqi_row(19.001, 72.821, 151)])

    res = live_mod.fetch_live(stations, key="k", token="t")
    assert res.attribution == live_mod.WAQI_ATTRIBUTION
    assert "World Air Quality Index" in res.attribution
    assert "waqi.info" in res.attribution


def test_waqi_request_uses_the_map_bounds_endpoint_and_bbox(live_mod, stations,
                                                            http):
    http.handlers["data.gov.in"] = requests.exceptions.ConnectionError("blocked")
    http.handlers["waqi"] = waqi_payload([waqi_row(19.001, 72.821, 151)])

    live_mod.fetch_live(stations, key="k", token="tok")
    url, params = next((u, p) for u, p in http.calls if "waqi" in u)
    assert url == f"{live_mod.WAQI_BASE}/map/bounds/"
    assert params["token"] == "tok"
    lat1, lon1, lat2, lon2 = live_mod.WAQI_BBOX
    assert params["latlng"] == f"{lat1},{lon1},{lat2},{lon2}"


def test_waqi_stale_reading_is_not_served(live_mod, stations, http):
    """Same STALE_HOURS rule as the primary source."""
    http.handlers["data.gov.in"] = requests.exceptions.ConnectionError("blocked")
    http.handlers["waqi"] = waqi_payload([
        waqi_row(19.001, 72.821, 151, stamp="2020-01-01T10:00:00+05:30")])

    res = live_mod.fetch_live(stations, key="k", token="t")
    assert not res.ok
    assert "hours old" in res.error


def test_waqi_counts_unmatched_stations(live_mod, stations, http):
    http.handlers["data.gov.in"] = requests.exceptions.ConnectionError("blocked")
    http.handlers["waqi"] = waqi_payload([waqi_row(19.001, 72.821, 151),
                                          waqi_row(19.60, 73.60, 90)])

    res = live_mod.fetch_live(stations, key="k", token="t")
    assert res.ok
    assert res.unmatched == 1
    assert res.n_reporting == 1


# --------------------------------------------------------------------------
# Both sources down
# --------------------------------------------------------------------------

def test_both_failing_reports_each_reason(live_mod, stations, http):
    http.handlers["data.gov.in"] = requests.exceptions.ConnectionError("blocked")
    http.handlers["waqi"] = requests.exceptions.ConnectionError("down too")

    res = live_mod.fetch_live(stations, key="k", token="t")
    assert not res.ok
    assert "Could not reach data.gov.in" in res.error
    assert "WAQI fallback also failed" in res.error
    assert "Could not reach WAQI" in res.error
    # The primary stays the named source when neither answered.
    assert res.source == live_mod.SOURCE_DATAGOV
    assert res.readings.empty
    assert res.n_total == 3


def test_both_failing_keeps_a_hint_for_the_reader(live_mod, stations, http):
    http.handlers["data.gov.in"] = FakeResponse({}, status_code=403)
    http.handlers["waqi"] = requests.exceptions.ConnectionError("down too")

    res = live_mod.fetch_live(stations, key="k", token="t")
    assert not res.ok
    assert "Key not authorised" in res.error
    assert res.hint and "data.gov.in" in res.hint


def test_no_credentials_at_all(live_mod, stations, monkeypatch):
    monkeypatch.setattr(live_mod, "api_key", lambda: None)
    monkeypatch.setattr(live_mod, "waqi_token", lambda: None)

    res = live_mod.fetch_live(stations)
    assert not res.ok
    assert "No DATAGOV_API_KEY configured" in res.error
    assert "No WAQI_TOKEN configured" in res.error
    assert res.hint


def test_rejected_waqi_token_is_reported(live_mod, stations, http):
    """WAQI answers HTTP 200 with status "error" for a bad token."""
    http.handlers["data.gov.in"] = requests.exceptions.ConnectionError("blocked")
    http.handlers["waqi"] = waqi_payload("Invalid key", status="error")

    res = live_mod.fetch_live(stations, key="k", token="bad")
    assert not res.ok
    assert "WAQI rejected the request (Invalid key)" in res.error


def test_waqi_with_nothing_matching_our_monitors(live_mod, stations, http):
    http.handlers["data.gov.in"] = requests.exceptions.ConnectionError("blocked")
    http.handlers["waqi"] = waqi_payload([waqi_row(19.60, 73.60, 90)])

    res = live_mod.fetch_live(stations, key="k", token="t")
    assert not res.ok
    assert "WAQI answered but reported no PM2.5" in res.error
    assert res.unmatched == 1


# --------------------------------------------------------------------------
# Shared helpers still behave for the primary source
# --------------------------------------------------------------------------

def test_nearest_station_respects_match_km(live_mod, stations):
    assert live_mod.nearest_station(19.001, 72.821, stations) == 1001
    assert live_mod.nearest_station(19.60, 73.60, stations) is None


def test_nearest_station_at_the_match_boundary(live_mod, stations):
    """MATCH_KM is inclusive, so a station exactly on it still matches."""
    lat = stations["lat"].iloc[0] + live_mod.MATCH_KM / 111.2
    assert live_mod.nearest_station(lat, stations["lon"].iloc[0], stations) == 1001
    further = stations["lat"].iloc[0] + (live_mod.MATCH_KM + 0.2) / 111.2
    assert live_mod.nearest_station(further, stations["lon"].iloc[0],
                                    stations) is None


def test_primary_parse_still_filters_on_pollutant(live_mod, stations, http):
    http.handlers["data.gov.in"] = datagov_payload([
        datagov_record(19.001, 72.821, 900.0, pollutant="PM10"),
        datagov_record(19.101, 72.881, 52.0)])

    res = live_mod.fetch_live(stations, key="k", token="t")
    assert res.ok
    assert dict(res.readings) == {1002: 52.0}


def test_source_strings_are_the_two_the_app_shows(live_mod):
    assert live_mod.SOURCE_DATAGOV == "data.gov.in (CPCB)"
    assert live_mod.SOURCE_WAQI == "WAQI (CPCB stations, converted from AQI)"
