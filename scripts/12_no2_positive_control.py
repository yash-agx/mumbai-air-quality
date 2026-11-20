"""NO2 as a positive control for the PM2.5 "no spatial structure" result.

Pre-registered in reports/preregistration_no2.md before any NO2 value was
looked at.

PM2.5 here is regional and long-lived, so a flat spatial pattern is what it
should show. NO2 comes from local combustion and should vary much more
between places. If the same measures, on the same monitors and hours, find
clearly more structure in NO2, the PM2.5 null is about PM2.5 and not a blind
spot of the method. The measures are the ones already used for PM2.5, taken
unchanged from scripts/11_averaging_times.py:

  (a) distance pattern   are nearby monitors more alike? (1 - semivariance)
  (b) prediction skill   grouped spatial CV, IDW against the city average
  (c) reliability (b2)   how alike two separate windows' patterns are

Primary panel: station-hours carrying both a cleaned PM2.5 and a cleaned NO2
reading, so the two pollutants are compared like for like. Every bootstrap
uses the same month resamples for both, so differences get paired intervals.

Reads  data/raw/no2_raw.parquet (data/fetch_no2.py) and the shipped PM2.5.
Writes reports/no2_positive_control.csv, reports/no2_positive_control_distance.csv,
       reports/figures/no2_vs_pm25.png
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "no2_raw.parquet"
OUT_CSV = ROOT / "reports" / "no2_positive_control.csv"
OUT_DIST = ROOT / "reports" / "no2_positive_control_distance.csv"
OUT_FIG = ROOT / "reports" / "figures" / "no2_vs_pm25.png"

WINDOWS = [("1h", 1), ("24h", 24), ("1w", 168)]

# Cleaning, as pre-registered. Values are in the labelled unit (ppb).
VALID_MAX = 1000.0
SATURATION_MIN_STATIONS = 3
SPIKE_RATIO = 4.0
SPIKE_HALF_WINDOW = 3          # hours either side
SPIKE_MIN_NEIGHBOURS = 3
SPIKE_QUANTILE = 0.999
STUCK_RUN_HOURS = 24
ZERO_SENSITIVITY_SHARE = 0.01
PPB_TO_UGM3 = 1.88             # NO2 at 25 C, 1 atm; for quoting levels only

# Decision rules.
S_RATIO = 2.0
P_FLOOR_PCT = -2.0


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def saturation_values(df, min_stations=SATURATION_MIN_STATIONS):
    """Exact values that are the recorded maximum at >= min_stations monitors."""
    if df.empty:
        return []
    maxima = df.groupby("station_id")["value"].max()
    counts = maxima.value_counts()
    return sorted(counts[counts >= min_stations].index.tolist())


def lone_spikes(df, ratio=SPIKE_RATIO, half=SPIKE_HALF_WINDOW,
                min_neighbours=SPIKE_MIN_NEIGHBOURS, q=SPIKE_QUANTILE):
    """Boolean mask: a single-hour jump its own monitor does not sustain.

    A reading is a spike if it is above `ratio` x the median of the same
    monitor's readings in the surrounding +-`half` hours (itself excluded,
    at least `min_neighbours` present) AND above that monitor's own
    `q` quantile. Deliberately temporal: a spatial corroboration rule would
    delete the local plumes this analysis is testing for.
    """
    out = pd.Series(False, index=df.index)
    for _, g in df.groupby("station_id"):
        s = g.set_index("timestamp")["value"]
        full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
        shifted = [full.shift(k) for k in range(-half, half + 1) if k != 0]
        nb = pd.concat(shifted, axis=1)
        med = nb.median(axis=1, skipna=True)
        enough = nb.notna().sum(axis=1) >= min_neighbours
        thresh = s.quantile(q)
        hit = enough & (full > ratio * med) & (full > thresh)
        out.loc[g.index] = hit.reindex(s.index).to_numpy()
    return out


def clean_no2(raw, parameter, start, end, spike_screen=True, drop_zeros=False):
    """The pre-registered NO2 (or NOx) cleaning. Returns (long df, report)."""
    clean_mod = _load("aq_clean_for_no2", "data/clean.py")
    rep = {"parameter": parameter}
    df = raw[raw["parameter"] == parameter].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    rep["rows_in"] = len(df)

    flagged = df["flagged"].astype(bool)
    rep["flagged"] = int(flagged.sum())
    df = df[~flagged]

    null = df["value"].isna()
    rep["null"] = int(null.sum())
    df = df[~null]

    bad = (df["value"] < 0) | (df["value"] > VALID_MAX)
    rep["out_of_range"] = int(bad.sum())
    df = df[~bad]
    zero = df["value"] == 0
    rep["zeros"] = int(zero.sum())
    if drop_zeros:
        df = df[~zero]

    df["timestamp"] = df["timestamp"].dt.floor("h")
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    rep["rows_in_window"] = len(df)
    rep["duplicate_hours"] = int(df.duplicated(["station_id", "timestamp"]).sum())
    df = df.groupby(["station_id", "timestamp"], as_index=False)["value"].mean()
    rep["zero_share"] = rep["zeros"] / max(len(df), 1)

    sat = saturation_values(df)
    sat_mask = df["value"].isin(sat)
    rep["saturation_values"] = sat
    rep["saturation"] = int(sat_mask.sum())
    df = df[~sat_mask]

    if spike_screen:
        spikes = lone_spikes(df)
        rep["lone_spike"] = int(spikes.sum())
        rep["lone_spike_stations"] = int(df.loc[spikes, "station_id"].nunique())
        df = df[~spikes]
    else:
        rep["lone_spike"] = 0
        rep["lone_spike_stations"] = 0

    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    stuck = clean_mod.run_lengths(df) >= STUCK_RUN_HOURS
    rep["stuck_masked"] = int(stuck.sum())
    df = df[~stuck].reset_index(drop=True)
    rep["rows_out"] = len(df)
    return df, rep


def to_wide(df, index, columns):
    w = df.pivot_table(index="timestamp", columns="station_id", values="value")
    return w.reindex(index=index, columns=columns)


def matched(a, b):
    """Both panels masked to station-hours where both have a reading."""
    both = a.notna() & b.notna()
    return a.where(both), b.where(both)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def analyse_panel(at, mod, panel, feat, dist, name, pollutant, resamples):
    """(a), (b), (c) at 1h / 24h / 1w for one panel. Returns rows, dists, boots."""
    grid, anchor = at.hourly_grid(panel)
    rows, dtabs, boots = [], [], {}
    for label, spec in WINDOWS:
        row, dtab, b = at.analyse_window(mod, grid, anchor, feat, dist, label,
                                         spec, at.COVERAGE, resamples)
        row = {"panel": name, "pollutant": pollutant, **row,
               "station_hours": int(panel.notna().sum().sum())}
        dtab.insert(0, "panel", name)
        dtab.insert(1, "pollutant", pollutant)
        rows.append(row)
        dtabs.append(dtab)
        boots[label] = b
    return rows, dtabs, boots


def paired_diff(boots_a, boots_b, window, key):
    """Bootstrap interval of a - b over the same month resamples."""
    return at_ci(np.asarray(boots_a[window][key]) - np.asarray(boots_b[window][key]))


def at_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan, np.nan
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def contrasts(t, boots, test, ref):
    """NO2-minus-PM2.5 differences (point, lo, hi) on matched panels."""
    def val(panel, window, key):
        r = t[(t["panel"] == panel) & (t["window"] == window)]
        return float(r[key].iloc[0])

    out = {}
    for window, key, name in (("1h", "reliability_mean", "b2_1h"),
                              ("24h", "reliability_mean", "b2_24h"),
                              ("1w", "reliability_mean", "b2_1w"),
                              ("1h", "near_minus_far", "near_far_1h"),
                              ("1h", "gap_pct", "gap_pct_1h"),
                              ("24h", "gap_pct", "gap_pct_24h"),
                              ("1w", "gap_pct", "gap_pct_1w")):
        point = val(test, window, key) - val(ref, window, key)
        lo, hi = paired_diff(boots[test], boots[ref], window, key)
        out[name] = (point, lo, hi)
    return out


def reading(t, d, test, ref):
    """Apply the pre-registered criteria S, G, P and the readings table."""
    def row(panel, window):
        return t[(t["panel"] == panel) & (t["window"] == window)].iloc[0]

    n1, p1 = row(test, "1h"), row(ref, "1h")
    S = (d["b2_1h"][1] > 0 and d["b2_1w"][1] > 0
         and n1["reliability_mean"] >= S_RATIO * p1["reliability_mean"])
    G = (n1["near_minus_far"] > 0 and n1["mantel_p"] < 0.05
         and d["near_far_1h"][1] > 0)
    P = (n1["gap_pct_hi"] < 0 and d["gap_pct_1h"][2] < 0
         and n1["gap_pct"] <= P_FLOOR_PCT)
    if S and G and P:
        verdict = "R1: full pass"
    elif S and (G or P):
        verdict = "R3: partial pass (" + ("G" if G else "P") + ")"
    elif S:
        verdict = "R2: stability only"
    elif G or P:
        verdict = "mixed"
    else:
        verdict = "R4: control fails"
    clearly_more = bool(S and (G or P))
    return {"S": bool(S), "G": bool(G), "P": bool(P), "verdict": verdict,
            "clearly_more_structure": clearly_more,
            "b2_ratio_1h": n1["reliability_mean"] / p1["reliability_mean"]}


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def figure(t, dist_tab, path, pm="pm25_matched", no2="no2_matched"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid_c, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colors = {pm: "#2a78d6", no2: "#eb6834"}
    names = {pm: "PM2.5", no2: "NO2"}
    labels = [w for w, _ in WINDOWS]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    fig.patch.set_facecolor(surface)

    def errbar(ax, panel, key, offset):
        r = t[t["panel"] == panel].set_index("window").loc[labels]
        y = r[key].to_numpy(float)
        lo, hi = r[f"{key}_lo"].to_numpy(float), r[f"{key}_hi"].to_numpy(float)
        ax.errorbar(x + offset, y, yerr=[y - lo, hi - y], fmt="o", ms=8,
                    color=colors[panel], ecolor=colors[panel], elinewidth=2,
                    capsize=0, mec=surface, mew=1.5, label=names[panel],
                    zorder=3)
        return y

    ax = axes[2]
    for panel, off in ((pm, -0.08), (no2, 0.08)):
        errbar(ax, panel, "reliability_mean", off)
    ax.set_title("(c) b2 reliability", loc="left", fontsize=11, color=ink)
    ax.set_ylabel("correlation between two separate windows", color=muted)
    ax.set_xticks(x, labels)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax = axes[0]
    for panel in (pm, no2):
        d = dist_tab[(dist_tab["panel"] == panel) & (dist_tab["window"] == "1h")
                     & (dist_tab["bin"] != "same cv_group")]
        ax.plot(d["mean_km"], d["corr"], color=colors[panel], lw=2,
                marker="o", ms=8, mec=surface, mew=1.5, label=names[panel],
                zorder=3)
    ax.axhline(0, color=muted, lw=1)
    ax.set_title("(a) Hourly anomaly similarity by distance", loc="left",
                 fontsize=11, color=ink)
    ax.set_xlabel("distance between monitors (km)", color=muted)
    ax.set_ylabel("1 - semivariance (0 = no more alike than random)",
                  color=muted)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    ax = axes[1]
    for panel, off in ((pm, -0.08), (no2, 0.08)):
        errbar(ax, panel, "gap_pct", off)
    ax.axhline(0, color=muted, lw=1)
    ax.set_title("(b) IDW RMSE minus city average", loc="left", fontsize=11,
                 color=ink)
    ax.set_ylabel("% of city-average RMSE (below 0 = IDW wins)", color=muted)
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, fontsize=9, loc="lower left")

    for ax in axes:
        ax.grid(True, color=grid_c, lw=0.8)
        ax.set_facecolor(surface)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid_c)
        ax.tick_params(colors=muted, labelsize=9)
    fig.text(0.01, 0.01,
             "Same 39 monitors, 35 cv_groups and station-hours for both "
             "pollutants (hours carrying both readings). Bars: 95% "
             "month-cluster bootstrap. 75% coverage.",
             fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def print_rows(t):
    cols = ["panel", "window", "n_windows", "n_stations", "n_groups",
            "median_groups_per_window", "reliability_mean",
            "reliability_mean_lo", "reliability_mean_hi", "reliability_pairs",
            "near_minus_far", "near_pairs", "far_pairs", "mantel_p",
            "rmse_city", "rmse_idw", "gap_pct", "gap_pct_lo", "gap_pct_hi",
            "group_gap_wilcoxon_p", "cv_groups", "cv_station_windows"]
    with pd.option_context("display.width", 300, "display.max_columns", 40,
                           "display.float_format", "{:.3f}".format):
        print(t[cols].to_string(index=False))


FLAT_CV_FRACTION = 0.5
OUT_FLAT = ROOT / "reports" / "no2_positive_control_flat.csv"


def flat_stations(df, fraction=FLAT_CV_FRACTION):
    """EXPLORATORY, not pre-registered. Monitors whose hourly coefficient of
    variation is under `fraction` of the network median: sensors that wobble
    in a narrow band, which the exact-repeat stuck rule cannot catch."""
    g = df.groupby("station_id")["value"]
    cv = g.std() / g.mean()
    return sorted(cv[cv < fraction * cv.median()].index.tolist()), cv


def flat_check():
    """Re-run the matched NO2 vs PM2.5 comparison without near-flat sensors."""
    global at
    at = _load("aq_averaging_times", "scripts/11_averaging_times.py")
    mod = at.load_engine()
    wide, feat, _ = mod.load()
    dist = mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    raw = pd.read_parquet(RAW_PATH)
    df, _ = clean_no2(raw, "no2", wide.index.min(), wide.index.max())
    flat, cv = flat_stations(df)
    print(f"near-flat NO2 monitors (CV < {FLAT_CV_FRACTION:g} x median "
          f"{cv.median():.2f}): {flat}")
    no2 = to_wide(df, wide.index, wide.columns)
    no2[flat] = np.nan
    pm_m, no2_m = matched(wide, no2)
    grid0, _ = at.hourly_grid(wide)
    resamples = at.month_resamples(
        pd.Index(grid0.index.strftime("%Y-%m")).unique().to_numpy())
    rows, boots = [], {}
    for name, pol, panel in (("pm25_matched_noflat", "pm25", pm_m),
                             ("no2_matched_noflat", "no2", no2_m)):
        r, _, b = analyse_panel(at, mod, panel, feat, dist, name, pol, resamples)
        rows += r
        boots[name] = b
    t = pd.DataFrame(rows)
    d = contrasts(t, boots, "no2_matched_noflat", "pm25_matched_noflat")
    r = reading(t, d, "no2_matched_noflat", "pm25_matched_noflat")
    for k, (pt, lo, hi) in d.items():
        t.loc[t["panel"] == "no2_matched_noflat", f"diff_{k}"] = pt
        t.loc[t["panel"] == "no2_matched_noflat", f"diff_{k}_lo"] = lo
        t.loc[t["panel"] == "no2_matched_noflat", f"diff_{k}_hi"] = hi
        print(f"  D {k:12s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    t["excluded_flat"] = ",".join(map(str, flat))
    t.to_csv(OUT_FLAT, index=False, float_format="%.5g")
    print_rows(t)
    print(f"  b2 ratio 1h {r['b2_ratio_1h']:.2f}; S={r['S']} G={r['G']} "
          f"P={r['P']} -> {r['verdict']} (exploratory)")


def main():
    if "--flat-check" in sys.argv:
        return flat_check()
    global at
    at = _load("aq_averaging_times", "scripts/11_averaging_times.py")
    mod = at.load_engine()
    wide, feat, _ = mod.load()
    dist = mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    start, end = wide.index.min(), wide.index.max()

    raw = pd.read_parquet(RAW_PATH)
    units = sorted(raw["units"].dropna().unique())
    print(f"units in raw pull: {units}")

    cleaned, reports = {}, {}
    for p in ("no2", "nox"):
        df, rep = clean_no2(raw, p, start, end)
        cleaned[p], reports[p] = df, rep
    cleaned["no2_nospike"], reports["no2_nospike"] = clean_no2(
        raw, "no2", start, end, spike_screen=False)
    if reports["no2"]["zero_share"] > ZERO_SENSITIVITY_SHARE:
        cleaned["no2_nozero"], reports["no2_nozero"] = clean_no2(
            raw, "no2", start, end, drop_zeros=True)
    for k, rep in reports.items():
        print(f"cleaning {k}: " + ", ".join(
            f"{a}={b}" for a, b in rep.items() if a != "parameter"))

    w = {k: to_wide(v, wide.index, wide.columns) for k, v in cleaned.items()}
    pm_m, no2_m = matched(wide, w["no2"])
    pmx_m, nox_m = matched(wide, w["nox"])
    panels = [
        ("pm25_matched", "pm25", pm_m),
        ("no2_matched", "no2", no2_m),
        ("pm25_own", "pm25", wide),
        ("no2_own", "no2", w["no2"]),
        ("pm25_matched_nox", "pm25", pmx_m),
        ("nox_matched", "nox", nox_m),
    ]
    pm_s, no2_s = matched(wide, w["no2_nospike"])
    panels += [("pm25_matched_nospike", "pm25", pm_s),
               ("no2_matched_nospike", "no2", no2_s)]
    if "no2_nozero" in w:
        pm_z, no2_z = matched(wide, w["no2_nozero"])
        panels += [("pm25_matched_nozero", "pm25", pm_z),
                   ("no2_matched_nozero", "no2", no2_z)]

    # One set of month resamples for every panel: the grid's months are the
    # same everywhere because every panel sits on the PM2.5 hour index.
    grid0, _ = at.hourly_grid(wide)
    months = pd.Index(grid0.index.strftime("%Y-%m")).unique().to_numpy()
    resamples = at.month_resamples(months)

    rows, dtabs, boots = [], [], {}
    for name, pol, panel in panels:
        t0 = time.time()
        print(f"\n{name}", flush=True)
        r, d, b = analyse_panel(at, mod, panel, feat, dist, name, pol, resamples)
        rows += r
        dtabs += d
        boots[name] = b
        print(f"  done in {time.time() - t0:.0f}s", flush=True)
    t = pd.DataFrame(rows)
    dtab = pd.concat(dtabs, ignore_index=True)

    comparisons = [("no2_matched", "pm25_matched"),
                   ("nox_matched", "pm25_matched_nox"),
                   ("no2_matched_nospike", "pm25_matched_nospike")]
    if "no2_nozero" in w:
        comparisons.append(("no2_matched_nozero", "pm25_matched_nozero"))
    summary = []
    for test, ref in comparisons:
        d = contrasts(t, boots, test, ref)
        r = reading(t, d, test, ref)
        summary.append((test, ref, d, r))
        for k, (pt, lo, hi) in d.items():
            t.loc[t["panel"] == test, f"diff_{k}"] = pt
            t.loc[t["panel"] == test, f"diff_{k}_lo"] = lo
            t.loc[t["panel"] == test, f"diff_{k}_hi"] = hi
        for k in ("S", "G", "P", "verdict", "clearly_more_structure"):
            t.loc[t["panel"] == test, f"reading_{k}"] = r[k]

    # Levels for context only (not a structure measure).
    lv = {p: cleaned[p]["value"] for p in ("no2", "nox")}
    for p, v in lv.items():
        print(f"\n{p} level (context): median {v.median():.1f} ppb"
              + (f" (~{PPB_TO_UGM3 * v.median():.0f} ug/m3)" if p == "no2" else "")
              + f", p90 {v.quantile(0.9):.1f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT_CSV, index=False, float_format="%.5g")
    dtab.to_csv(OUT_DIST, index=False, float_format="%.5g")
    figure(t, dtab, OUT_FIG)

    print()
    print_rows(t)
    for test, ref, d, r in summary:
        print(f"\n{test} vs {ref}")
        for k, (pt, lo, hi) in d.items():
            print(f"  D {k:12s} {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"  b2 ratio 1h {r['b2_ratio_1h']:.2f}; S={r['S']} G={r['G']} "
              f"P={r['P']} -> {r['verdict']}; clearly more structure: "
              f"{r['clearly_more_structure']}")
    with pd.option_context("display.width", 200):
        print("\ndistance pattern, 1h")
        sub = dtab[dtab["window"] == "1h"]
        print(sub.pivot(index="bin", columns="panel", values="corr")
              .round(3).to_string())
        print(sub[sub["panel"] == "no2_matched"][["bin", "n_pairs", "mean_km"]]
              .to_string(index=False))
    print(f"\nwrote {OUT_CSV.relative_to(ROOT)}, {OUT_DIST.relative_to(ROOT)}, "
          f"{OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
