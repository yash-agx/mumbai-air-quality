"""Tests for scripts/02_features.py -- the OSM land-use feature generation.

No Overpass call and no data/interim cache is touched. The geometry helpers are
exercised directly, and features_for() runs against a synthetic index built out
of the module's own cKDTree and compound_path, so the projection, the resampling
and the per-feature arithmetic are all the real code -- only the OSM download is
replaced.
"""

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------

def test_to_xy_puts_the_bbox_centre_at_the_origin(features_mod):
    x, y = features_mod.to_xy(features_mod.LAT0, features_mod.LON0)
    assert float(x) == pytest.approx(0.0)
    assert float(y) == pytest.approx(0.0)


def test_to_xy_scales_a_degree_of_latitude_to_about_111_km(features_mod):
    _, y = features_mod.to_xy(features_mod.LAT0 + 1.0, features_mod.LON0)
    assert float(y) == pytest.approx(111_195.0, rel=1e-3)


def test_to_xy_shrinks_longitude_by_the_cosine_of_the_centre_latitude(features_mod):
    x, _ = features_mod.to_xy(features_mod.LAT0, features_mod.LON0 + 1.0)
    expected = 111_195.0 * np.cos(np.radians(features_mod.LAT0))
    assert float(x) == pytest.approx(expected, rel=1e-3)


def test_to_xy_is_vectorised_and_index_aligned(features_mod):
    lats = [19.0, 19.1, 19.2]
    lons = [72.8, 72.9, 73.0]
    x, y = features_mod.to_xy(lats, lons)
    assert x.shape == y.shape == (3,)
    # Monotone in both axes, north- and east-positive.
    assert list(np.argsort(y)) == [0, 1, 2]
    assert list(np.argsort(x)) == [0, 1, 2]


# --------------------------------------------------------------------------
# Resampling lines into point clouds
# --------------------------------------------------------------------------

def test_resample_spaces_points_along_a_segment(features_mod):
    """A 100 m segment at 10 m spacing is 10 points, the first at the midpoint
    of the first 10 m, so no vertex is double-counted between segments."""
    x0 = np.array([0.0])
    y0 = np.array([0.0])
    x1 = np.array([100.0])
    y1 = np.array([0.0])
    px, py = features_mod.resample(x0, y0, x1, y1)
    assert len(px) == 10
    assert px == pytest.approx(np.arange(10) * 10.0 + 5.0)
    assert py == pytest.approx(np.zeros(10))


def test_resample_gives_a_short_segment_one_point(features_mod):
    px, py = features_mod.resample(np.array([0.0]), np.array([0.0]),
                                   np.array([2.0]), np.array([0.0]))
    assert len(px) == 1
    assert px[0] == pytest.approx(1.0)
    assert py[0] == pytest.approx(0.0)


def test_resample_handles_many_segments_at_once(features_mod):
    x0 = np.array([0.0, 0.0])
    y0 = np.array([0.0, 0.0])
    x1 = np.array([100.0, 50.0])
    y1 = np.array([0.0, 0.0])
    px, _ = features_mod.resample(x0, y0, x1, y1)
    assert len(px) == 10 + 5


def test_resample_point_count_tracks_length(features_mod):
    """Counting points is how road length is measured, so n * SPACING_M has to
    come back to the length of the line."""
    length = 1234.0
    px, _ = features_mod.resample(np.array([0.0]), np.array([0.0]),
                                  np.array([length]), np.array([0.0]))
    assert len(px) * features_mod.SPACING_M == pytest.approx(length, rel=0.01)


# --------------------------------------------------------------------------
# Parsing Overpass geometry
# --------------------------------------------------------------------------

def _geom(points):
    return [{"lat": lat, "lon": lon} for lat, lon in points]


def test_way_geometries_yields_ways(features_mod):
    payload = {"elements": [
        {"type": "way", "tags": {"highway": "primary"},
         "geometry": _geom([(19.0, 72.8), (19.01, 72.81)])},
    ]}
    out = list(features_mod.way_geometries(payload))
    assert len(out) == 1
    el, lat, lon = out[0]
    assert el["tags"]["highway"] == "primary"
    assert lat == pytest.approx([19.0, 19.01])
    assert lon == pytest.approx([72.8, 72.81])


def test_way_geometries_descends_into_relation_members(features_mod):
    """Industrial land-use comes back as relations as well as ways."""
    payload = {"elements": [
        {"type": "relation", "tags": {"landuse": "industrial"}, "members": [
            {"type": "way", "geometry": _geom([(19.0, 72.8), (19.0, 72.81)])},
            {"type": "node"},
        ]},
    ]}
    out = list(features_mod.way_geometries(payload))
    assert len(out) == 1
    assert out[0][0]["tags"]["landuse"] == "industrial"


def test_way_geometries_skips_elements_without_geometry(features_mod):
    payload = {"elements": [
        {"type": "way", "tags": {}},
        {"type": "node", "lat": 19.0, "lon": 72.8},
        {"type": "way", "geometry": []},
    ]}
    assert list(features_mod.way_geometries(payload)) == []


def test_line_points_filters_on_tags(features_mod):
    payload = {"elements": [
        {"type": "way", "tags": {"highway": "primary"},
         "geometry": _geom([(19.0, 72.8), (19.0, 72.81)])},
        {"type": "way", "tags": {"highway": "residential"},
         "geometry": _geom([(19.1, 72.8), (19.1, 72.81)])},
    ]}
    everything = features_mod.line_points(payload)
    major_only = features_mod.line_points(
        payload, keep=lambda t: t.get("highway") == "primary")
    assert everything.shape[1] == 2
    assert len(major_only) < len(everything)
    assert len(major_only) == pytest.approx(len(everything) / 2, rel=0.05)


def test_line_points_skips_one_point_ways(features_mod):
    payload = {"elements": [
        {"type": "way", "tags": {}, "geometry": _geom([(19.0, 72.8)])},
    ]}
    assert features_mod.line_points(payload).shape == (0, 2)


def test_line_points_on_an_empty_payload(features_mod):
    assert features_mod.line_points({"elements": []}).shape == (0, 2)


# --------------------------------------------------------------------------
# Industrial polygons
# --------------------------------------------------------------------------

def _square_payload(lat0=19.0, lon0=72.8, d=0.01, closed=True):
    pts = [(lat0, lon0), (lat0, lon0 + d), (lat0 + d, lon0 + d), (lat0 + d, lon0)]
    if closed:
        pts = pts + [(lat0, lon0)]
    return {"elements": [{"type": "way", "tags": {"landuse": "industrial"},
                          "geometry": _geom(pts)}]}


def test_polygon_rings_closes_an_open_ring(features_mod):
    rings = features_mod.polygon_rings(_square_payload(closed=False))
    assert len(rings) == 1
    ring = rings[0]
    assert ring[0] == pytest.approx(ring[-1])


def test_polygon_rings_normalises_the_winding(features_mod):
    """Two polygons wound opposite ways would cancel under nonzero winding, so
    every ring comes back with the same orientation however OSM wound it."""
    clockwise = features_mod.polygon_rings(_square_payload())
    pts = _square_payload()["elements"][0]["geometry"][::-1]
    reversed_payload = {"elements": [{"type": "way", "tags": {}, "geometry": pts}]}
    anticlockwise = features_mod.polygon_rings(reversed_payload)

    def shoelace(r):
        x, y = r[:, 0], r[:, 1]
        return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))

    assert shoelace(clockwise[0]) <= 0
    assert shoelace(anticlockwise[0]) <= 0


def test_polygon_rings_skips_degenerate_ways(features_mod):
    payload = {"elements": [
        {"type": "way", "tags": {}, "geometry": _geom([(19.0, 72.8), (19.0, 72.81),
                                                       (19.01, 72.81)])},
    ]}
    assert features_mod.polygon_rings(payload) == []


def test_compound_path_contains_points_inside_the_polygon(features_mod):
    rings = features_mod.polygon_rings(_square_payload(lat0=19.0, lon0=72.8, d=0.01))
    path = features_mod.compound_path(rings)
    inside_x, inside_y = features_mod.to_xy([19.005], [72.805])
    outside_x, outside_y = features_mod.to_xy([19.05], [72.9])
    assert path.contains_points(np.column_stack([inside_x, inside_y]))[0]
    assert not path.contains_points(np.column_stack([outside_x, outside_y]))[0]


# --------------------------------------------------------------------------
# Road density
# --------------------------------------------------------------------------

def test_road_density_counts_points_as_road_length(features_mod):
    """n points within the radius is n * SPACING_M metres of road inside it."""
    from scipy.spatial import cKDTree

    # 100 points at 10 m spacing along a line through the origin = 1 km of road,
    # all of it inside a 500 m radius centred on the line's midpoint.
    pts = np.column_stack([np.arange(100) * 10.0 - 495.0, np.zeros(100)])
    tree = cKDTree(pts)
    density = features_mod.road_density(tree, np.array([[0.0, 0.0]]), 500.0)
    expected = (100 * features_mod.SPACING_M / 1000.0) / (np.pi * 0.5 ** 2)
    assert density[0] == pytest.approx(expected)


def test_road_density_is_zero_far_from_any_road(features_mod):
    from scipy.spatial import cKDTree

    tree = cKDTree(np.array([[0.0, 0.0]]))
    assert features_mod.road_density(tree, np.array([[50_000.0, 0.0]]), 500.0)[0] == 0.0


def test_road_density_falls_as_the_radius_grows_past_the_road(features_mod):
    """The same 1 km of road spread over four times the area is a quarter as dense."""
    from scipy.spatial import cKDTree

    pts = np.column_stack([np.arange(100) * 10.0 - 495.0, np.zeros(100)])
    tree = cKDTree(pts)
    at_500 = features_mod.road_density(tree, np.array([[0.0, 0.0]]), 500.0)[0]
    at_1000 = features_mod.road_density(tree, np.array([[0.0, 0.0]]), 1000.0)[0]
    assert at_1000 == pytest.approx(at_500 / 4.0)


# --------------------------------------------------------------------------
# features_for(), against a synthetic index
# --------------------------------------------------------------------------

@pytest.fixture
def fake_index(features_mod, monkeypatch):
    """Replace the OSM-backed index with a small hand-built one.

    A major road runs east-west through the bbox centre, minor roads parallel
    to it 300 m north, a coastline 5 km west, one industrial square around the
    centre, and a block of buildings at the centre.
    """
    from scipy.spatial import cKDTree

    def line(x_from, x_to, y, spacing=features_mod.SPACING_M):
        xs = np.arange(x_from, x_to, spacing)
        return np.column_stack([xs, np.full(len(xs), float(y))])

    major = line(-5_000, 5_000, 0.0)
    minor = np.vstack([major, line(-5_000, 5_000, 300.0)])
    coast = np.column_stack([np.full(200, -5_000.0), np.arange(200) * 10.0 - 1_000.0])
    # A 1 km square of industrial land centred 2 km east of the origin.
    ring = np.array([[1_500.0, -500.0], [2_500.0, -500.0],
                     [2_500.0, 500.0], [1_500.0, 500.0], [1_500.0, -500.0]])
    industrial_edge = np.vstack([
        line(1_500, 2_500, -500.0), line(1_500, 2_500, 500.0),
        np.column_stack([np.full(100, 1_500.0), np.arange(100) * 10.0 - 500.0]),
        np.column_stack([np.full(100, 2_500.0), np.arange(100) * 10.0 - 500.0]),
    ])
    buildings = np.column_stack([
        np.repeat(np.arange(10) * 40.0 - 180.0, 10),
        np.tile(np.arange(10) * 40.0 - 180.0, 10),
    ])

    index = {
        "major_roads": cKDTree(major),
        "all_roads": cKDTree(minor),
        "coast": cKDTree(coast),
        "industrial": cKDTree(industrial_edge),
        "buildings": cKDTree(buildings),
        "industrial_area": features_mod.compound_path([ring]),
        "counts": {},
    }
    monkeypatch.setattr(features_mod, "_index", lambda: index)
    return index


def _latlon(features_mod, x, y):
    """Invert to_xy, so a test can name a point in metres from the bbox centre."""
    lat = features_mod.LAT0 + np.degrees(y / features_mod.R_EARTH)
    lon = features_mod.LON0 + np.degrees(
        x / (features_mod.R_EARTH * np.cos(np.radians(features_mod.LAT0))))
    return lat, lon


def test_features_for_returns_the_six_documented_columns(features_mod, fake_index):
    out = features_mod.features_for([features_mod.LAT0], [features_mod.LON0])
    assert list(out.columns) == features_mod.FEATURES
    assert len(out) == 1
    assert out.notna().all().all()


def test_features_for_is_index_aligned_to_its_input(features_mod, fake_index):
    lats, lons = _latlon(features_mod, np.array([0.0, 1_000.0, 2_000.0]),
                         np.array([0.0, 0.0, 0.0]))
    out = features_mod.features_for(lats, lons)
    assert len(out) == 3
    assert list(out.index) == [0, 1, 2]


def test_features_for_measures_distance_to_the_nearest_major_road(features_mod,
                                                                 fake_index):
    """A point 1 km north of the only major road is 1 km from a major road."""
    lat, lon = _latlon(features_mod, np.array([0.0]), np.array([1_000.0]))
    out = features_mod.features_for(lat, lon)
    assert out["dist_major_road_m"].iloc[0] == pytest.approx(1_000.0, abs=15.0)


def test_features_for_counts_minor_roads_in_the_density(features_mod, fake_index):
    """Density uses all drivable roads, so it sees the minor road the distance
    feature ignores."""
    lat, lon = _latlon(features_mod, np.array([0.0]), np.array([300.0]))
    out = features_mod.features_for(lat, lon)
    assert out["dist_major_road_m"].iloc[0] == pytest.approx(300.0, abs=15.0)
    assert out["road_density_500m"].iloc[0] > 0


def test_features_for_puts_a_point_inside_an_industrial_estate_at_zero(features_mod,
                                                                      fake_index):
    """Inside the estate the answer is 0, not the distance to its perimeter."""
    lat, lon = _latlon(features_mod, np.array([2_000.0]), np.array([0.0]))
    out = features_mod.features_for(lat, lon)
    assert out["dist_industrial_m"].iloc[0] == 0.0


def test_features_for_measures_to_the_industrial_boundary_from_outside(features_mod,
                                                                       fake_index):
    lat, lon = _latlon(features_mod, np.array([0.0]), np.array([0.0]))
    out = features_mod.features_for(lat, lon)
    assert out["dist_industrial_m"].iloc[0] == pytest.approx(1_500.0, abs=25.0)


def test_features_for_measures_distance_to_the_coast(features_mod, fake_index):
    lat, lon = _latlon(features_mod, np.array([0.0]), np.array([0.0]))
    out = features_mod.features_for(lat, lon)
    assert out["dist_coast_m"].iloc[0] == pytest.approx(5_000.0, abs=25.0)


def test_features_for_building_density_is_per_square_km(features_mod, fake_index):
    """100 footprints inside 500 m, over the pi*0.5^2 km2 that circle covers."""
    lat, lon = _latlon(features_mod, np.array([0.0]), np.array([0.0]))
    out = features_mod.features_for(lat, lon)
    expected = round(100 / (np.pi * 0.5 ** 2), 1)
    assert out["building_density_500m"].iloc[0] == pytest.approx(expected, abs=0.2)


def test_features_for_reports_empty_land_as_far_and_undeveloped(features_mod,
                                                                fake_index):
    """The far corner has no road, no building and no industry near it -- the
    signature the extrapolation mask later keys on."""
    lat, lon = _latlon(features_mod, np.array([40_000.0]), np.array([40_000.0]))
    out = features_mod.features_for(lat, lon).iloc[0]
    assert out["road_density_500m"] == 0.0
    assert out["road_density_1km"] == 0.0
    assert out["building_density_500m"] == 0.0
    assert out["dist_major_road_m"] > 10_000
    assert out["dist_industrial_m"] > 10_000


def test_features_for_is_consistent_between_the_two_road_radii(features_mod,
                                                               fake_index):
    """Both radii see the same road, so the 1 km figure cannot exceed 4x the
    500 m one -- the area ratio is the only thing that differs."""
    lat, lon = _latlon(features_mod, np.array([0.0]), np.array([0.0]))
    out = features_mod.features_for(lat, lon).iloc[0]
    assert out["road_density_1km"] <= out["road_density_500m"]


def test_features_at_returns_one_point_as_a_dict(features_mod, fake_index):
    got = features_mod.features_at(features_mod.LAT0, features_mod.LON0)
    assert set(got) == set(features_mod.FEATURES)
    assert all(isinstance(v, float) for v in got.values())


def test_features_for_handles_many_points_at_once(features_mod, fake_index):
    """The same code serves 39 stations and a prediction grid."""
    xs = np.linspace(-4_000, 4_000, 50)
    lat, lon = _latlon(features_mod, xs, np.zeros(50))
    out = features_mod.features_for(lat, lon)
    assert len(out) == 50
    assert out.notna().all().all()
    # On the road line itself, every point is essentially on a major road.
    assert (out["dist_major_road_m"] < 15.0).all()


# --------------------------------------------------------------------------
# variation()
# --------------------------------------------------------------------------

def test_variation_summarises_each_feature(features_mod):
    feat = pd.DataFrame({
        "dist_major_road_m": [100.0, 200.0, 300.0, 400.0],
        "road_density_500m": [10.0, 10.0, 10.0, 10.0],
        "road_density_1km": [1.0, 2.0, 3.0, 4.0],
        "dist_coast_m": [0.0, 0.0, 1000.0, 2000.0],
        "dist_industrial_m": [0.0, 500.0, 1000.0, 1500.0],
        "building_density_500m": [100.0, 200.0, 300.0, 400.0],
    })
    out = features_mod.variation(feat).set_index("feature")
    assert set(out.index) == set(features_mod.FEATURES)

    road = out.loc["dist_major_road_m"]
    assert road["min"] == 100.0
    assert road["max"] == 400.0
    assert road["median"] == 250.0
    assert road["IQR"] == 150.0
    assert road["n_unique"] == 4
    assert road["n_zero"] == 0

    # A feature that never varies has zero spread, which is the case this
    # summary exists to surface.
    assert out.loc["road_density_500m", "CV"] == 0.0
    assert out.loc["road_density_500m", "n_unique"] == 1
    assert out.loc["dist_coast_m", "n_zero"] == 2


def test_variation_is_sorted_worst_first(features_mod):
    feat = pd.DataFrame({c: [1.0, 2.0, 3.0, 4.0] for c in features_mod.FEATURES})
    feat["dist_coast_m"] = [1.0, 1.0, 1.0, 1.0]
    out = features_mod.variation(feat)
    assert out["CV"].iloc[0] <= out["CV"].iloc[-1]
