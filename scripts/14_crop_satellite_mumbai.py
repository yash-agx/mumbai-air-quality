"""Crop the IIT Delhi daily satellite PM2.5 files (India, 0.01 deg) to Mumbai.

Input:  data/raw/satellite/Rf2_PM2.5_2024-MM-DD Alok.nc  (one file per day)
Output: data/interim/satellite_mumbai_2024.nc  (time x lat x lon)

Box: 18.85-19.30 N, 72.75-73.05 E. Mask == 1 marks a valid retrieval;
cells with Mask == 0 are set to NaN.
"""
import re
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "satellite"
OUT = ROOT / "data" / "interim" / "satellite_mumbai_2024.nc"
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 18.85, 19.30, 72.75, 73.05
EPS = 1e-6


def main():
    files = sorted(RAW.glob("Rf2_PM2.5_*.nc"))
    dated = {}
    for f in files:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
        if m:
            dated[m.group(1)] = f
    dates = sorted(dated)
    if not dates:
        raise SystemExit("no satellite files found")

    with netCDF4.Dataset(dated[dates[0]]) as ds:
        lat = ds["lat"][:].data
        lon = ds["lon"][:].data
    i = np.where((lat >= LAT_MIN - EPS) & (lat <= LAT_MAX + EPS))[0]
    j = np.where((lon >= LON_MIN - EPS) & (lon <= LON_MAX + EPS))[0]
    si, sj = slice(i.min(), i.max() + 1), slice(j.min(), j.max() + 1)

    pm = np.full((len(dates), len(i), len(j)), np.nan, dtype="float32")
    mask = np.zeros_like(pm, dtype="uint8")
    for k, d in enumerate(dates):
        with netCDF4.Dataset(dated[d]) as ds:
            assert np.allclose(ds["lat"][si].data, lat[si])
            assert np.allclose(ds["lon"][sj].data, lon[sj])
            p = np.ma.filled(ds["pm2.5"][si, sj].astype("float64"), np.nan)
            m = np.ma.filled(ds["Mask"][si, sj], 0).astype("uint8")
        p[m != 1] = np.nan
        pm[k], mask[k] = p, m

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(OUT, "w") as out:
        out.createDimension("time", len(dates))
        out.createDimension("lat", len(i))
        out.createDimension("lon", len(j))
        t = out.createVariable("time", "i4", ("time",))
        t.units = "days since 2024-01-01"
        t[:] = (pd.to_datetime(dates) - pd.Timestamp("2024-01-01")).days
        out.createVariable("lat", "f8", ("lat",))[:] = lat[si]
        out.createVariable("lon", "f8", ("lon",))[:] = lon[sj]
        v = out.createVariable("pm25", "f4", ("time", "lat", "lon"), zlib=True, fill_value=np.nan)
        v.units = "ug/m3"
        v.long_name = "daily satellite PM2.5 (IIT Delhi), NaN where Mask != 1"
        v[:] = pm
        out.createVariable("mask", "u1", ("time", "lat", "lon"), zlib=True)[:] = mask

    valid = np.isfinite(pm)
    print(f"days: {len(dates)} ({dates[0]} to {dates[-1]})")
    print(f"grid: {len(i)} x {len(j)} cells")
    print(f"valid cell-days: {valid.mean():.1%}")
    print(f"PM2.5 range: {np.nanmin(pm):.1f} to {np.nanmax(pm):.1f} ug/m3, mean {np.nanmean(pm):.1f}")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
