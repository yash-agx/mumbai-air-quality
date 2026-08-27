"""Probe the OpenAQ v3 API for monitoring stations in the Mumbai region.

Read-only reconnaissance: how many stations exist, what they measure, and how
much history they have. Nothing is written to disk.
"""

import os
import sys
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

# Station names and units contain non-ASCII (ug/m3); the Windows console
# defaults to cp1252 and would mangle them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

API = "https://api.openaq.org/v3"

# Mumbai metropolitan region: Colaba up through Thane/Navi Mumbai.
# OpenAQ wants bbox as min_lon, min_lat, max_lon, max_lat.
BBOX = (72.70, 18.85, 73.20, 19.40)

PAGE_SIZE = 1000
TIMEOUT = 30


def get_key():
    load_dotenv()
    key = os.getenv("OPENAQ_API_KEY")
    if not key:
        sys.exit("OPENAQ_API_KEY not found. Add it to .env in the project root.")
    return key


def fetch_locations(key):
    """Page through /v3/locations for the bbox and return all results."""
    headers = {"X-API-Key": key}
    locations = []
    page = 1

    while True:
        params = {
            "bbox": ",".join(str(c) for c in BBOX),
            "limit": PAGE_SIZE,
            "page": page,
        }
        r = requests.get(f"{API}/locations", headers=headers, params=params, timeout=TIMEOUT)

        if r.status_code == 401:
            sys.exit("401 Unauthorized - OPENAQ_API_KEY is missing or invalid.")
        if r.status_code == 429:
            sys.exit("429 Rate limited - wait a minute and re-run.")
        r.raise_for_status()

        results = r.json().get("results", [])
        locations.extend(results)
        if len(results) < PAGE_SIZE:
            return locations
        page += 1


def parse_ts(node):
    """OpenAQ returns {'utc': ..., 'local': ...} or null for datetimeFirst/Last."""
    if not node:
        return None
    raw = node.get("utc")
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def describe(loc, now):
    sensors = loc.get("sensors") or []
    pollutants = sorted({
        s.get("parameter", {}).get("name", "?") for s in sensors
    })
    units = {
        s.get("parameter", {}).get("name"): s.get("parameter", {}).get("units")
        for s in sensors
    }

    first = parse_ts(loc.get("datetimeFirst"))
    last = parse_ts(loc.get("datetimeLast"))
    coords = loc.get("coordinates") or {}
    site = (coords.get("latitude"), coords.get("longitude"))

    print(f"\n[{loc.get('id')}] {loc.get('name', 'unnamed')}")
    if loc.get("locality"):
        print(f"  locality:   {loc['locality']}")
    print(f"  coords:     {coords.get('latitude')}, {coords.get('longitude')}")
    print(f"  provider:   {(loc.get('provider') or {}).get('name', 'unknown')}")
    print(f"  mobile:     {loc.get('isMobile')}   reference monitor: {loc.get('isMonitor')}")

    if pollutants:
        pretty = ", ".join(f"{p} ({units.get(p) or '?'})" for p in pollutants)
    else:
        pretty = "none listed"
    print(f"  pollutants: {pretty}")

    if first and last:
        span = (last - first).days
        stale = (now - last).days
        print(f"  data range: {first.date()} -> {last.date()}  ({span} days span, last reading {stale} days ago)")
    else:
        print("  data range: not reported")

    return {"pollutants": set(pollutants), "last": last, "site": site}


def main():
    key = get_key()
    print(f"Querying OpenAQ v3 locations in bbox {BBOX} (lon/lat)...")

    locations = fetch_locations(key)
    if not locations:
        print("\nNo stations returned for this bounding box.")
        return

    now = datetime.now(timezone.utc)
    summaries = [describe(loc, now) for loc in locations]

    with_pm25 = [s for s in summaries if "pm25" in s["pollutants"]]
    # 30 days is a loose cutoff for "still reporting" vs. a dead station.
    live_pm25 = [s for s in with_pm25 if s["last"] and (now - s["last"]).days <= 30]

    print("\n" + "=" * 60)
    print(f"stations found:              {len(locations)}")
    print(f"  reporting pm25:            {len(with_pm25)}")
    print(f"  pm25 active in last 30d:   {len(live_pm25)}")

    # The same physical site is often listed twice under different providers
    # (e.g. CPCB and caaqm), so raw location count overstates spatial coverage.
    distinct_sites = {s["site"] for s in live_pm25}
    print(f"  distinct coords among those: {len(distinct_sites)}")

    all_pollutants = sorted(set().union(*(s["pollutants"] for s in summaries)))
    print(f"  pollutants across region:  {', '.join(all_pollutants) or 'none'}")


if __name__ == "__main__":
    main()
