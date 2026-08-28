"""Spatial land-use features for any point in the Mumbai region, from OpenStreetMap.

Run directly: python scripts/02_features.py [--refresh]

Reads  data/processed/pm25_clean.parquet   (station coordinates)
Writes data/processed/station_features.parquet

features_for(lats, lons) is the reusable entry point: it takes arrays of any
length, so the same code serves the 39 stations here and a prediction grid in
Phase 2. Six features per point:

    dist_major_road_m       to the nearest motorway/trunk/primary/secondary
    road_density_500m       km of drivable road per km2 within 500 m
    road_density_1km        same, within 1 km
    dist_coast_m            to the nearest natural=coastline way
    dist_industrial_m       to the nearest landuse=industrial polygon (0 inside)
    building_density_500m   building footprints per km2 within 500 m

building_density_500m counts mapped OSM footprints, and OSM under-maps
informal and high-density housing in Mumbai, often as one polygon per block
rather than per structure. Read it as a relative proxy for built form, not
as a structure count.

OSM is fetched once from Overpass and cached twice: the raw response under
data/raw/osm/, and the projected point clouds the features actually need under
data/interim/. A warm run touches neither the network nor the JSON.
"""

import argparse
import gzip
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "processed" / "pm25_clean.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "station_features.parquet"
OSM_DIR = ROOT / "data" / "raw" / "osm"
INTERIM_DIR = ROOT / "data" / "interim"

# Same region as data/fetch.py: min_lon, min_lat, max_lon, max_lat.
BBOX = (72.70, 18.85, 73.20, 19.40)

# Fetch a ring beyond the bbox. A station 22 m inside the eastern edge would
# otherwise get "distance to nearest industrial land" measured only against the
# industry we happened to download, and the true nearest could sit just outside.
# 0.05 deg is ~5.5 km; main() checks whether any answer is still censored.
PAD_DEG = 0.05

# Coastline needs far more room. Inland stations sit 20 km+ from the sea, so a
# 5.5 km ring cannot prove it found the nearest coast, and the layer is ~100
# ways -- widening it to ~55 km costs nothing and makes the answer defensible.
COAST_PAD_DEG = 0.5
PADS = {"roads": PAD_DEG, "coastline": COAST_PAD_DEG,
        "industrial": PAD_DEG, "buildings": PAD_DEG}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Overpass answers 406 to the default python-requests User-Agent.
USER_AGENT = "mumbai-air-quality-dashboard/0.1"
OVERPASS_TIMEOUT = 300
REQUEST_SPACING = 5.0

# Roads and coastlines are lines, but nearest-distance and radius counts want
# points, so every way is resampled into points this far apart. Doubles as the
# length quantum for road density: n points within a radius ~= n * SPACING_M
# metres of road inside it.
SPACING_M = 10.0

MAJOR_ROADS = ("motorway", "trunk", "primary", "secondary")
MINOR_ROADS = ("tertiary", "unclassified", "residential")

R_EARTH = 6371008.8
LAT0 = (BBOX[1] + BBOX[3]) / 2
LON0 = (BBOX[0] + BBOX[2]) / 2


def overpass_bbox(pad):
    """Padded bbox in Overpass order: south, west, north, east."""
    lon_min, lat_min, lon_max, lat_max = BBOX
    return (f"{lat_min - pad},{lon_min - pad},"
            f"{lat_max + pad},{lon_max + pad}")


def queries():
    bb = overpass_bbox(PAD_DEG)
    road_re = "|".join(MAJOR_ROADS + MINOR_ROADS)
    return {
        # One download covers both road features; major roads are a tag filter
        # on the same set, so there is no reason to ask Overpass twice.
        "roads": f'way["highway"~"^({road_re})(_link)?$"]({bb});out tags geom;',
        "coastline": (f'way["natural"="coastline"]'
                      f'({overpass_bbox(COAST_PAD_DEG)});out geom;'),
        "industrial": (f'(way["landuse"="industrial"]({bb});'
                       f'relation["landuse"="industrial"]({bb}););out geom;'),
        # Density only needs one point per building, and "out center" turns a
        # 240k-polygon download into a 240k-point one.
        "buildings": f'way["building"]({bb});out center;',
    }


def fetch_layer(name, body, refresh=False):
    """Fetch one Overpass layer, caching the raw response gzipped."""
    path = OSM_DIR / f"{name}.json.gz"
    if path.exists() and not refresh:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)

    OSM_DIR.mkdir(parents=True, exist_ok=True)
    query = f"[out:json][timeout:{OVERPASS_TIMEOUT}];{body}"
    print(f"  fetching {name:<11} ", end="", flush=True)
    t0 = time.time()
    r = requests.post(OVERPASS_URL, data={"data": query},
                      headers={"User-Agent": USER_AGENT}, timeout=OVERPASS_TIMEOUT + 60)
    if r.status_code != 200:
        sys.exit(f"\nOverpass returned HTTP {r.status_code} for {name}:\n{r.text[:400]}")
    payload = r.json()
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"{len(payload.get('elements', [])):>7,} elements  "
          f"({time.time() - t0:.0f}s, {path.stat().st_size / 1e6:.1f} MB cached)")
    time.sleep(REQUEST_SPACING)
    return payload


def to_xy(lat, lon):
    """Equirectangular metres about the bbox centre.

    Good to a few parts in a thousand over a 60 km box, which is far below the
    precision these features are used at, and it keeps the dependency list to
    numpy rather than pulling in a projection library.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = R_EARTH * np.cos(np.radians(LAT0)) * np.radians(lon - LON0)
    y = R_EARTH * np.radians(lat - LAT0)
    return x, y


def way_geometries(payload):
    """Yield (lat, lon) arrays for every way, including relation members."""
    for el in payload.get("elements", []):
        if el.get("type") == "way" and el.get("geometry"):
            g = el["geometry"]
            yield el, np.array([p["lat"] for p in g]), np.array([p["lon"] for p in g])
        elif el.get("type") == "relation":
            for m in el.get("members", []):
                if m.get("type") == "way" and m.get("geometry"):
                    g = m["geometry"]
                    yield el, np.array([p["lat"] for p in g]), np.array([p["lon"] for p in g])


def resample(x0, y0, x1, y1):
    """Points spaced ~SPACING_M along every segment, vectorised over all of them.

    Sampling at segment midpoints rather than endpoints means consecutive
    segments do not both claim the vertex they share, so a point is never
    counted twice when measuring road length by counting points.
    """
    n = np.maximum(1, np.round(np.hypot(x1 - x0, y1 - y0) / SPACING_M)).astype(np.int64)
    idx = np.repeat(np.arange(len(n)), n)
    starts = np.concatenate(([0], np.cumsum(n)[:-1]))
    t = (np.arange(n.sum()) - np.repeat(starts, n) + 0.5) / n[idx]
    return x0[idx] + t * (x1[idx] - x0[idx]), y0[idx] + t * (y1[idx] - y0[idx])


def line_points(payload, keep=None):
    """Every way in the payload as one resampled point cloud."""
    xs, ys = [], []
    for el, lat, lon in way_geometries(payload):
        if keep is not None and not keep(el.get("tags") or {}):
            continue
        if len(lat) < 2:
            continue
        x, y = to_xy(lat, lon)
        xs.append(np.column_stack([x[:-1], x[1:]]))
        ys.append(np.column_stack([y[:-1], y[1:]]))
    if not xs:
        return np.empty((0, 2))
    xs, ys = np.vstack(xs), np.vstack(ys)
    px, py = resample(xs[:, 0], ys[:, 0], xs[:, 1], ys[:, 1])
    return np.column_stack([px, py])


def polygon_rings(payload):
    """Closed rings as (x, y) arrays, each wound counter-clockwise."""
    rings = []
    for _, lat, lon in way_geometries(payload):
        if len(lat) < 4:
            continue
        x, y = to_xy(lat, lon)
        if x[0] != x[-1] or y[0] != y[-1]:
            x, y = np.append(x, x[0]), np.append(y, y[0])
        # matplotlib fills a compound path by nonzero winding, so two polygons
        # wound opposite ways would cancel where they overlap. Normalise first.
        if np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1])) > 0:
            x, y = x[::-1], y[::-1]
        rings.append(np.column_stack([x, y]))
    return rings


def compound_path(rings):
    verts, codes = [], []
    for r in rings:
        verts.append(r)
        codes.append(np.array([MplPath.MOVETO] + [MplPath.LINETO] * (len(r) - 1)))
    return MplPath(np.vstack(verts), np.concatenate(codes))


def build_layers(refresh=False):
    """Fetch, project and cache the point clouds every feature is measured off."""
    cache = {name: INTERIM_DIR / f"osm_{name}.parquet"
             for name in ("major_roads", "all_roads", "coast", "industrial", "buildings")}
    rings_cache = INTERIM_DIR / "osm_industrial_rings.parquet"

    if not refresh and all(p.exists() for p in cache.values()) and rings_cache.exists():
        out = {k: pd.read_parquet(p)[["x", "y"]].to_numpy() for k, p in cache.items()}
        r = pd.read_parquet(rings_cache)
        out["industrial_rings"] = [g[["x", "y"]].to_numpy() for _, g in r.groupby("ring")]
        return out

    print("Building OSM layers for "
          f"{BBOX} padded by {PAD_DEG} deg (~{PAD_DEG * 111:.0f} km):")
    q = queries()
    roads = fetch_layer("roads", q["roads"], refresh)
    coast = fetch_layer("coastline", q["coastline"], refresh)
    indus = fetch_layer("industrial", q["industrial"], refresh)
    builds = fetch_layer("buildings", q["buildings"], refresh)

    major = tuple(MAJOR_ROADS) + tuple(f"{r}_link" for r in MAJOR_ROADS)
    out = {
        "major_roads": line_points(roads, keep=lambda t: t.get("highway") in major),
        "all_roads": line_points(roads),
        "coast": line_points(coast),
        "industrial": line_points(indus),
    }

    bx, by = [], []
    for el in builds.get("elements", []):
        c = el.get("center")
        if c:
            bx.append(c["lat"])
            by.append(c["lon"])
    x, y = to_xy(bx, by)
    out["buildings"] = np.column_stack([x, y])

    out["industrial_rings"] = polygon_rings(indus)

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    for name, path in cache.items():
        pd.DataFrame(out[name], columns=["x", "y"]).to_parquet(path, index=False)
    pd.concat([pd.DataFrame(r, columns=["x", "y"]).assign(ring=i)
               for i, r in enumerate(out["industrial_rings"])]).to_parquet(
        rings_cache, index=False)
    return out


@lru_cache(maxsize=1)
def _index():
    """KD-trees and the industrial polygon test, built once per process."""
    layers = build_layers()
    return {
        "major_roads": cKDTree(layers["major_roads"]),
        "all_roads": cKDTree(layers["all_roads"]),
        "coast": cKDTree(layers["coast"]),
        "industrial": cKDTree(layers["industrial"]),
        "buildings": cKDTree(layers["buildings"]),
        "industrial_area": compound_path(layers["industrial_rings"]),
        "counts": {k: len(v) for k, v in layers.items()},
    }


def road_density(tree, pts, radius_m):
    """km of road per km2 within radius_m."""
    n = tree.query_ball_point(pts, radius_m, return_length=True)
    length_km = n * SPACING_M / 1000.0
    area_km2 = np.pi * (radius_m / 1000.0) ** 2
    return length_km / area_km2


def features_for(lats, lons):
    """Six OSM land-use features for any set of points. Index-aligned to input."""
    x, y = to_xy(lats, lons)
    pts = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
    ix = _index()

    dist_industrial, _ = ix["industrial"].query(pts)
    # A point inside an industrial estate is at distance 0, not at its distance
    # to the perimeter, which is what the boundary tree alone would report.
    dist_industrial = np.where(ix["industrial_area"].contains_points(pts),
                               0.0, dist_industrial)

    n_build = ix["buildings"].query_ball_point(pts, 500.0, return_length=True)

    return pd.DataFrame({
        "dist_major_road_m": ix["major_roads"].query(pts)[0].round(1),
        "road_density_500m": road_density(ix["all_roads"], pts, 500.0).round(2),
        "road_density_1km": road_density(ix["all_roads"], pts, 1000.0).round(2),
        "dist_coast_m": ix["coast"].query(pts)[0].round(1),
        "dist_industrial_m": np.round(dist_industrial, 1),
        "building_density_500m": np.round(n_build / (np.pi * 0.5 ** 2), 1),
    })


def features_at(lat, lon):
    return features_for([lat], [lon]).iloc[0].to_dict()


FEATURES = ["dist_major_road_m", "road_density_500m", "road_density_1km",
            "dist_coast_m", "dist_industrial_m", "building_density_500m"]


def variation(feat):
    """Spread of each feature across the stations, worst-first."""
    rows = []
    for c in FEATURES:
        v = feat[c]
        iqr = v.quantile(0.75) - v.quantile(0.25)
        rows.append({
            "feature": c,
            "min": round(v.min(), 1),
            "median": round(v.median(), 1),
            "max": round(v.max(), 1),
            "IQR": round(iqr, 1),
            # Spread relative to level: the only scale-free number here, and the
            # one that says whether a model can tell the stations apart at all.
            "CV": round(v.std() / v.mean(), 3) if v.mean() else float("nan"),
            "n_unique": int(v.nunique()),
            "n_zero": int((v == 0).sum()),
        })
    return pd.DataFrame(rows).sort_values("CV")


def edge_distance_m(lats, lons, pad):
    """Metres from each point to the nearest edge of a layer's fetched area.

    A nearest-feature distance is only trustworthy if it is shorter than this:
    beyond it, something closer could be sitting just outside what we fetched.
    """
    lon_min, lat_min, lon_max, lat_max = BBOX
    x, y = to_xy(np.asarray(lats), np.asarray(lons))
    xmin, ymin = to_xy(lat_min - pad, lon_min - pad)
    xmax, ymax = to_xy(lat_max + pad, lon_max + pad)
    return np.minimum.reduce([x - xmin, xmax - x, y - ymin, ymax - y])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="refetch OSM, ignoring the cache")
    args = ap.parse_args()

    if not CLEAN_PATH.exists():
        sys.exit(f"{CLEAN_PATH.relative_to(ROOT)} not found - run data/clean.py first.")

    clean = pd.read_parquet(CLEAN_PATH)
    stations = (clean.groupby("station_id", as_index=False)
                     .agg(site=("site", "first"), cv_group=("cv_group", "first"),
                          lat=("lat", "first"), lon=("lon", "first")))

    if args.refresh:
        build_layers(refresh=True)
        _index.cache_clear()

    feat = features_for(stations["lat"].to_numpy(), stations["lon"].to_numpy())
    out = pd.concat([stations, feat], axis=1)

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FEATURES_PATH, index=False)

    counts = _index()["counts"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 100)

    print("=" * 118)
    print("SPATIAL FEATURES  (OpenStreetMap via Overpass)")
    print("=" * 118)
    print(f"source points   {counts['major_roads']:,} major-road / "
          f"{counts['all_roads']:,} all-road / {counts['coast']:,} coastline / "
          f"{counts['industrial']:,} industrial-boundary, sampled every {SPACING_M:.0f} m")
    print(f"                {counts['buildings']:,} building centroids, "
          f"{counts['industrial_rings']:,} industrial polygons")

    print("\n" + "-" * 118)
    print("PER STATION")
    print("-" * 118)
    print(out.drop(columns=["cv_group"]).to_string(index=False))

    print("\n" + "-" * 118)
    print("VARIATION ACROSS THE 39 STATIONS  (lowest CV first)")
    print("-" * 118)
    print(variation(feat).to_string(index=False))

    print("\n" + "-" * 118)
    print("NOTES")
    print("-" * 118)

    weak = variation(feat).query("CV < 0.25")
    if not weak.empty:
        print("Near-constant across stations - little for a model to learn from:")
        for r in weak.itertuples():
            print(f"  CV {r.CV:<6} {r.feature:<22} "
                  f"median {r.median:>9,.1f}, IQR {r.IQR:>9,.1f}")
    else:
        print("Every feature varies meaningfully across the stations (CV >= 0.25).")

    corr = feat.corr()
    pairs = [(corr.loc[a, b], a, b)
             for i, a in enumerate(FEATURES) for b in FEATURES[i + 1:]
             if abs(corr.loc[a, b]) >= 0.8]
    if pairs:
        print("\nFeature pairs that are near-duplicates of each other "
              "(|r| >= 0.8 across stations):")
        for r, a, b in sorted(pairs, key=lambda t: -abs(t[0])):
            print(f"  r = {r:+.3f}   {a}  <->  {b}")

    for c in FEATURES:
        n_zero = int((feat[c] == 0).sum())
        if n_zero >= 5:
            print(f"\n{c}: {n_zero} of {len(feat)} stations sit at exactly 0, so the "
                  f"feature is really a flag for those and a distance for the rest.")

    for c, layer in (("dist_major_road_m", "roads"), ("dist_coast_m", "coastline"),
                     ("dist_industrial_m", "industrial")):
        edge = edge_distance_m(stations["lat"], stations["lon"], PADS[layer])
        censored = stations.loc[feat[c].to_numpy() > edge, "site"]
        if len(censored):
            print(f"\n{c}: {len(censored)} station(s) are farther from the nearest "
                  f"feature than from the edge of the fetched area, so the true "
                  f"nearest may lie outside it: {', '.join(censored)}")

    print(f"\nWrote {len(out)} rows x {len(FEATURES)} features to "
          f"{FEATURES_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
