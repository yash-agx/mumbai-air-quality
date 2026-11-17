"""Test 3 of reports/preregistration_2026-11.md: the monitor table.

Descriptive, no hypothesis and no p-value. Prints a markdown table of every
monitor - area, agency, coordinates, distance to the nearest major road, and
distance to the coast - for pasting into the README.

The table exists to make one thing unmissable: every curve in this project
describes ONE INSTRUMENT AT ONE SPOT, not a neighbourhood. Two monitors can
carry the same area name and sit 50 m and 800 m from a highway; the chart
labels are shorthand for the monitor, not for where anyone lives.

Coordinates are rounded to 3 decimals (~100 m). These are published monitor
locations, not anybody's address, but the table does not need more precision
than the point it makes.
"""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_engine():
    spec = importlib.util.spec_from_file_location(
        "interpolate", ROOT / "model" / "interpolate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load_engine()
    wide, feat, _ = mod.load()
    clean = pd.read_parquet(ROOT / "data" / "processed" / "pm25_clean.parquet")
    agency = clean.groupby("station_id")["agency"].first()
    hours = clean.groupby("station_id").size()

    d = feat.copy()
    d["agency"] = d.index.map(agency)
    d["hours"] = d.index.map(hours)
    d = d.sort_values("site")

    print(f"{len(d)} monitors, {d['cv_group'].nunique()} independent units "
          f"(monitors under ~1 km apart share a unit).\n")

    print("| Monitor | Agency | Lat, lon | Nearest major road | Coast | Hours |")
    print("|---|---|---|---|---|---|")
    for sid, r in d.iterrows():
        grouped = (d["cv_group"] == r["cv_group"]).sum() > 1
        name = r["site"] + (" †" if grouped else "")
        print(f"| {name} | {r['agency']} | {r['lat']:.3f}, {r['lon']:.3f} "
              f"| {r['dist_major_road_m']:,.0f} m "
              f"| {r['dist_coast_m'] / 1000:.1f} km "
              f"| {r['hours']:,} |")

    multi = d[d.duplicated("cv_group", keep=False)]
    print(f"\n† {len(multi)} monitors in "
          f"{multi['cv_group'].nunique()} near-coincident pairs, held out "
          f"together in cross-validation and counted once in every test.")
    print(f"\nDistance to a major road ranges "
          f"{d['dist_major_road_m'].min():,.0f}-{d['dist_major_road_m'].max():,.0f} m "
          f"(median {d['dist_major_road_m'].median():,.0f} m); to the coast "
          f"{d['dist_coast_m'].min() / 1000:.1f}-{d['dist_coast_m'].max() / 1000:.1f} km.")


if __name__ == "__main__":
    main()
