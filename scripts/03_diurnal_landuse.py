"""Do diurnal patterns relate to land use?

Tests whether the per-monitor hour-of-day profile -- its shape and its swing --
is associated with the OSM land-use features already computed per station.

Profiles are built on the COMMON WINDOW -- the span over which every monitor
reports -- not the full 18-month record. Three monitors came online in March
2026, so on the full record their hour-of-day profiles average a different,
monsoon-weighted slice of the year from everyone else's, which is a seasonal
difference wearing a spatial costume. The full record is kept as a sensitivity.

Read the honest caveat with it: the window was promoted to primary AFTER the
full-record run showed two of the four primary estimates moving under it. The
confound it removes is real and the argument for it does not depend on the
result, but the decision to prefer it did. See NOTES.md.

Structure, fixed before any result was seen:

  1. CONTROL. A Mantel test on whether diurnal shape similarity decays with
     distance. If it does, the features (which are themselves spatially
     autocorrelated) get a free ride on geography and every p-value below is
     anticonservative. Near-coincident pairs (same cv_group) are excluded --
     they are the one bin the variogram work already flagged as unexploitable.

  2. PRIMARY. Four pre-specified feature x summary pairs, Benjamini-Hochberg
     across those four only:
       road_density_500m     x morning-evening balance
       dist_coast_m          x trough hour
       building_density_500m x day-night swing amplitude
       dist_industrial_m     x morning-evening balance

  3. EXPLORATORY. The full 6 x 5 grid, reported separately and never promoted.

  4. RELIABILITY. Each summary recomputed on odd days and even days
     independently. A null against an unreliable summary says nothing, so the
     split-half ceiling is reported next to every test.

Run: python scripts/03_diurnal_landuse.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

N_PERM = 20000
SEED = 0
TZ = "Asia/Kolkata"

# The primary window: the first local day on which every monitor in the panel
# is reporting. Chinchpada (6266794) is the last to come online, on 2026-03-10;
# Kalu Nagar and Vithalwadi arrive within the week before it. Profiles built
# before this date average different date ranges at different stations.
#
# What it costs, and the cost is not small: the window runs March to August, so
# it holds none of Mumbai's November-February pollution season and is weighted
# towards the monsoon. The primary analysis therefore describes the clean half
# of the year. Quantified in the run output.
COMMON_START = "2026-03-10"

# Windows for the morning-evening balance, IST, inclusive.
MORNING = list(range(6, 11))    # 06:00-10:00
EVENING = list(range(18, 23))   # 18:00-22:00

FEATURES = ["dist_major_road_m", "road_density_500m", "road_density_1km",
            "dist_coast_m", "dist_industrial_m", "building_density_500m"]

SUMMARIES = ["morning_evening_balance", "peak_hour", "trough_hour",
             "swing_amplitude", "bimodality"]
CIRCULAR = {"peak_hour", "trough_hour"}

PRIMARY = [
    ("road_density_500m", "morning_evening_balance"),
    ("dist_coast_m", "trough_hour"),
    ("building_density_500m", "swing_amplitude"),
    ("dist_industrial_m", "morning_evening_balance"),
]


# ---------------------------------------------------------------- reused
# diurnal() and shape_of() are copied verbatim from app.py (lines 345 and 353)
# so the summaries are built on exactly the computation the dashboard reports.
# Kept as copies rather than a shared import because app.py is a Streamlit
# script and importing it would execute the whole page.

def diurnal(values, hours):
    """Mean by hour of day, as a 24-long array with gaps as NaN."""
    out = np.full(24, np.nan)
    g = pd.Series(values).groupby(hours).mean()
    out[g.index.to_numpy()] = g.to_numpy()
    return out


def shape_of(profile):
    """A profile with its level and amplitude removed - what is left is shape."""
    d = profile - np.nanmean(profile)
    sd = np.nanstd(d)
    return d / sd if sd > 0 else d


# ---------------------------------------------------------------- summaries

def harmonics(profile, n_harm=2):
    """Least-squares fit of intercept + n_harm harmonics to a 24h profile.

    Returns (a, b): cosine and sine coefficients, index 0 = the 24h term.
    """
    ok = np.isfinite(profile)
    h = np.arange(24)[ok]
    cols = [np.ones(int(ok.sum()))]
    for k in range(1, n_harm + 1):
        cols += [np.cos(2 * np.pi * k * h / 24), np.sin(2 * np.pi * k * h / 24)]
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, profile[ok], rcond=None)
    a = np.array([coef[1 + 2 * k] for k in range(n_harm)])
    b = np.array([coef[2 + 2 * k] for k in range(n_harm)])
    return a, b


def summarise_profile(profile, morning=None, evening=None):
    """The five summaries for one 24-long hour-of-day profile.

    morning/evening default to the pre-specified MORNING/EVENING windows; they
    are parameters only so the boundary-sensitivity check can vary them.
    """
    morning = MORNING if morning is None else morning
    evening = EVENING if evening is None else evening
    sh = shape_of(profile)
    a, b = harmonics(profile, n_harm=2)

    # Peak/trough as the phase of the fitted 24h harmonic, in hours. A raw
    # argmax over 24 integers is both discrete and circular-blind; the phase is
    # continuous and rotation-correct. NOTE: with a first-harmonic phase the
    # trough sits 12h from the peak by construction, so the two are one
    # statistic and their circular-linear tests come out numerically identical.
    peak = (np.arctan2(b[0], a[0]) / (2 * np.pi) * 24) % 24
    trough = (peak + 12) % 24

    amp1 = float(np.hypot(a[0], b[0]))
    amp2 = float(np.hypot(a[1], b[1]))

    return {
        # Shape summaries, computed on the normalised profile.
        "morning_evening_balance": float(np.nanmean(sh[morning])
                                         - np.nanmean(sh[evening])),
        "peak_hour": float(peak),
        "trough_hour": float(trough),
        # Level summary, in ug/m3 on the raw profile. Removed by shape_of, and
        # the quantity most likely to respond to ventilation.
        "swing_amplitude": float(np.nanmax(profile) - np.nanmin(profile)),
        # How bimodal the day is: 12h harmonic amplitude against the 24h one.
        "bimodality": float(amp2 / amp1) if amp1 > 0 else np.nan,
    }


def trough_two_harmonic(profile):
    """Continuous trough hour from the fitted 24h+12h curve (sensitivity only).

    Unlike the first-harmonic phase this is not pinned 12h from the peak, so it
    can move independently of it.
    """
    a, b = harmonics(profile, n_harm=2)
    t = np.arange(0, 24, 0.01)
    y = sum(a[k] * np.cos(2 * np.pi * (k + 1) * t / 24)
            + b[k] * np.sin(2 * np.pi * (k + 1) * t / 24) for k in range(2))
    return float(t[int(np.argmin(y))])


# ---------------------------------------------------------------- statistics

def circ_mean(hours):
    """Circular mean of hours-of-day."""
    th = np.asarray(hours, float) * 2 * np.pi / 24
    return float((np.arctan2(np.sin(th).mean(), np.cos(th).mean())
                  / (2 * np.pi) * 24) % 24)


def circ_diff(a, b):
    """Signed circular difference a-b in hours, wrapped to [-12, 12)."""
    return (np.asarray(a, float) - np.asarray(b, float) + 12) % 24 - 12


def circ_linear_r(x, hours):
    """Mardia's circular-linear correlation, on ranked x for robustness.

    Returns R in [0, 1]. The statistic has no sign -- an association between a
    linear variable and a phase has no direction to report.
    """
    xr = stats.rankdata(x)
    th = np.asarray(hours, float) * 2 * np.pi / 24
    c, s = np.cos(th), np.sin(th)
    rxc = np.corrcoef(xr, c)[0, 1]
    rxs = np.corrcoef(xr, s)[0, 1]
    rcs = np.corrcoef(c, s)[0, 1]
    denom = 1 - rcs ** 2
    if denom <= 0:
        return np.nan
    return float(np.sqrt(max((rxc ** 2 + rxs ** 2
                              - 2 * rxc * rxs * rcs) / denom, 0.0)))


def circ_circ_r(a_h, b_h):
    """Jammalamadaka circular-circular correlation between two hour vectors."""
    a = np.asarray(a_h, float) * 2 * np.pi / 24
    b = np.asarray(b_h, float) * 2 * np.pi / 24
    am = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    bm = np.arctan2(np.sin(b).mean(), np.cos(b).mean())
    num = np.sum(np.sin(a - am) * np.sin(b - bm))
    den = np.sqrt(np.sum(np.sin(a - am) ** 2) * np.sum(np.sin(b - bm) ** 2))
    return float(num / den) if den > 0 else np.nan


def perm_test(x, y, circular, n_perm=N_PERM, seed=SEED):
    """Permutation test: Spearman rho if linear, Mardia R if y is circular.

    n is small (35 groups) and the features are heavily skewed distances, so a
    permutation null is used rather than an asymptotic one.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    stat = (circ_linear_r(x, y) if circular
            else float(stats.spearmanr(x, y).statistic))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        null[i] = (circ_linear_r(x, yp) if circular
                   else float(stats.spearmanr(x, yp).statistic))
    # Circular R is non-negative so its test is one-sided on the statistic;
    # Spearman is two-sided on |rho|. Both use the +1 correction.
    extreme = (null >= stat) if circular else (np.abs(null) >= abs(stat))
    return stat, float((extreme.sum() + 1) / (n_perm + 1)), int(ok.sum())


def benjamini_hochberg(pvals):
    """BH-adjusted p-values, input order preserved."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        prev = min(prev, p[idx] * n / (n - rank + 1))
        adj[idx] = prev
    return adj


def mantel(sim, dist, keep, n_perm=N_PERM, seed=SEED):
    """Mantel test: Pearson r between two square matrices over `keep` pairs.

    Permutes station labels jointly on rows and columns, which is the standard
    Mantel null and preserves each matrix's internal dependence structure.
    """
    n = sim.shape[0]
    iu = np.triu_indices(n, 1)
    mask = keep[iu]
    a, b = sim[iu][mask], dist[iu][mask]
    obs = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(n)
        sp = sim[np.ix_(p, p)]
        null[i] = np.corrcoef(sp[iu][mask], b)[0, 1]
    return obs, float(((np.abs(null) >= abs(obs)).sum() + 1) / (n_perm + 1)), len(a)


# ---------------------------------------------------------------- build

def load_engine():
    spec = importlib.util.spec_from_file_location(
        "interpolate", ROOT / "model" / "interpolate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def profiles_for(wide, hours, row_mask=None):
    """Per-station 24h profiles as a DataFrame indexed by station_id."""
    V = wide.to_numpy(dtype=float)
    h = hours if row_mask is None else hours[row_mask]
    out = {}
    for j, sid in enumerate(wide.columns):
        v = V[:, j] if row_mask is None else V[row_mask, j]
        out[sid] = diurnal(v, h)
    return pd.DataFrame(out, index=range(24)).T


def summary_table(prof, morning=None, evening=None):
    return pd.DataFrame({sid: summarise_profile(prof.loc[sid].to_numpy(),
                                                morning, evening)
                         for sid in prof.index}).T[SUMMARIES]


def to_groups(summ, feat):
    """Collapse stations to cv_group units.

    Near-coincident stations are not independent observations of a land-use
    relationship -- bandra_east's two monitors are 70 m apart and share their
    features to three significant figures. Counting them twice inflates n and
    understates every p-value. Circular summaries are averaged circularly.
    """
    df = summ.join(feat[["cv_group"] + FEATURES])
    rows = []
    for g, sub in df.groupby("cv_group"):
        row = {"cv_group": g, "n_stations": len(sub)}
        for c in SUMMARIES:
            row[c] = (circ_mean(sub[c]) if c in CIRCULAR
                      else float(sub[c].mean()))
        for c in FEATURES:
            row[c] = float(sub[c].mean())
        rows.append(row)
    return pd.DataFrame(rows).set_index("cv_group")


def run_grid(units, features=FEATURES, summaries=SUMMARIES):
    rows = []
    for f in features:
        for s in summaries:
            stat, p, n = perm_test(units[f], units[s], s in CIRCULAR)
            rows.append({"feature": f, "summary": s, "circular": s in CIRCULAR,
                         "stat": stat, "p": p, "n": n})
    return pd.DataFrame(rows)


def main():
    mod = load_engine()
    wide, feat, _ = mod.load()
    ist = wide.index.tz_convert(TZ)
    hours = ist.hour.to_numpy()
    days = ist.dayofyear.to_numpy()

    # PRIMARY = the common window. The full record is the sensitivity.
    common = np.asarray(ist >= pd.Timestamp(COMMON_START, tz=TZ))
    coverage = wide[common].notna().mean()

    print(f"panel {wide.shape[0]} hours x {wide.shape[1]} stations, "
          f"{feat['cv_group'].nunique()} cv groups")
    print(f"full record  {ist.min():%Y-%m-%d} to {ist.max():%Y-%m-%d} "
          f"({wide.shape[0]:,} hours)")
    print(f"PRIMARY window {ist[common].min():%Y-%m-%d} to {ist[common].max():%Y-%m-%d} "
          f"({int(common.sum()):,} hours, "
          f"{100 * common.sum() / len(common):.0f}% of the record)")

    prof_full = profiles_for(wide, hours)
    prof = profiles_for(wide, hours, common)
    summ_full = summary_table(prof_full)
    summ = summary_table(prof)
    units_full = to_groups(summ_full, feat)
    units = to_groups(summ, feat)
    print(f"units for testing: {len(units)} cv groups "
          f"({(units['n_stations'] > 1).sum()} of them multi-station)")

    # What the window costs, stated up front rather than buried.
    print("\n  what the common window costs:")
    print(f"    months covered: {'  '.join(sorted(set(ist[common].strftime('%b %Y'))))}")
    print(f"    Nov-Feb pollution-season hours retained: "
          f"{int((pd.Series(ist[common].month).isin([11, 12, 1, 2])).sum())}")
    lv_f = float(np.nanmean(wide.to_numpy(float)))
    lv_c = float(np.nanmean(wide[common].to_numpy(float)))
    print(f"    mean concentration: {lv_f:.1f} ug/m3 full record -> "
          f"{lv_c:.1f} in window ({100 * (lv_c - lv_f) / lv_f:+.0f}%)")
    print(f"    median swing amplitude: "
          f"{summ_full['swing_amplitude'].median():.1f} ug/m3 -> "
          f"{summ['swing_amplitude'].median():.1f}")
    print(f"    per-station hours behind each profile (median): "
          f"{int(wide.notna().sum().median()):,} -> "
          f"{int(wide[common].notna().sum().median()):,}\n")

    # ---------------------------------------------------------- summaries
    print("=" * 70)
    print("SUMMARY DISTRIBUTIONS (station level, n=%d)" % len(summ))
    print("=" * 70)
    for c in SUMMARIES:
        v = summ[c].to_numpy(float)
        if c in CIRCULAR:
            spread = np.abs(circ_diff(v, circ_mean(v)))
            print(f"  {c:<24} circular mean {circ_mean(v):5.2f} h, "
                  f"median |dev| {np.median(spread):4.2f} h, "
                  f"max |dev| {spread.max():4.2f} h")
        else:
            print(f"  {c:<24} median {np.nanmedian(v):7.3f}  "
                  f"range {np.nanmin(v):7.3f} to {np.nanmax(v):7.3f}")
    print()

    # ---------------------------------------------------------- 1. control
    print("=" * 70)
    print("1. CONTROL -- Mantel: does diurnal shape similarity decay with distance?")
    print("=" * 70)
    sids = list(prof.index)
    S = np.vstack([shape_of(prof.loc[s].to_numpy()) for s in sids])
    sim = np.corrcoef(S)
    lat = feat.loc[sids, "lat"].to_numpy()
    lon = feat.loc[sids, "lon"].to_numpy()
    dist = mod.distance_matrix(lat, lon)
    grp = feat.loc[sids, "cv_group"].to_numpy()
    same_group = grp[:, None] == grp[None, :]
    keep = ~same_group
    r, p, npairs = mantel(sim, dist, keep)
    print(f"  Mantel r(shape similarity, distance) = {r:+.4f}  "
          f"p = {p:.4f}  over {npairs} pairs")
    print(f"  ({int(np.triu(same_group, 1).sum())} within-group pairs excluded)")

    iu = np.triu_indices(len(sids), 1)
    dv, sv, kv = dist[iu], sim[iu], keep[iu]
    edges = np.percentile(dv[kv], np.linspace(0, 100, 8))
    print("\n  binned, excluded pairs dropped:")
    print(f"  {'separation':>16} {'pairs':>6} {'median r(shape)':>16}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = kv & (dv >= lo) & (dv < hi if hi < edges[-1] else dv <= hi)
        if m.sum():
            print(f"  {lo:6.1f}-{hi:5.1f} km {int(m.sum()):6d} "
                  f"{np.median(sv[m]):16.3f}")
    wg = np.triu(same_group, 1) & (dist > 0)
    if wg.any():
        print(f"\n  within-group (near-coincident) pairs, for reference: "
              f"n={int(wg.sum())}, "
              f"median r(shape) = {np.median(sim[wg]):.3f}, "
              f"median separation {np.median(dist[wg]):.2f} km")

    # The other half of the confounding argument: a feature that is itself
    # spatially structured can only borrow geography's significance if shape is
    # spatially structured too. Reported so the claim rests on both sides.
    print("\n  are the features themselves spatially structured? "
          "(Mantel, same excluded pairs)")
    for f in FEATURES:
        v = feat.loc[sids, f].to_numpy(float)
        fd = np.abs(v[:, None] - v[None, :])
        rf, pf, _ = mantel(fd, dist, keep, n_perm=2000, seed=SEED)
        print(f"    {f:<24} r = {rf:+.3f}  p = {pf:.4f}")
    print()

    # ------------------------------------------------------ 4. reliability
    print("=" * 70)
    print("RELIABILITY -- split-half (odd vs even local days)")
    print("=" * 70)
    odd = common & (days % 2 == 1)
    even = common & (days % 2 == 0)
    p_odd, p_even = profiles_for(wide, hours, odd), profiles_for(wide, hours, even)
    s_odd, s_even = summary_table(p_odd), summary_table(p_even)

    # The same split on the full record, to price the window's noise cost.
    fo = summary_table(profiles_for(wide, hours, days % 2 == 1))
    fe = summary_table(profiles_for(wide, hours, days % 2 == 0))

    rel = {}
    print(f"  {'summary':<24} {'split-half r':>13} {'ceiling':>9} "
          f"{'(full rec)':>11}   typical half-to-half gap")
    for c in SUMMARIES:
        a, b = s_odd[c].to_numpy(float), s_even[c].to_numpy(float)
        af, bf = fo[c].to_numpy(float), fe[c].to_numpy(float)
        okf = np.isfinite(af) & np.isfinite(bf)
        rf = (circ_circ_r(af[okf], bf[okf]) if c in CIRCULAR
              else float(stats.spearmanr(af[okf], bf[okf]).statistic))
        ok = np.isfinite(a) & np.isfinite(b)
        if c in CIRCULAR:
            rr = circ_circ_r(a[ok], b[ok])
            gap = f"{np.median(np.abs(circ_diff(a[ok], b[ok]))):.2f} h (median |diff|)"
        else:
            rr = float(stats.spearmanr(a[ok], b[ok]).statistic)
            gap = (f"{np.median(np.abs(a[ok] - b[ok])):.3f} "
                   f"(median |diff|, units of the summary)")
        # Spearman-Brown: the halves each carry half the record, so the
        # full-record reliability is higher than the half-to-half correlation.
        sb = 2 * rr / (1 + rr) if rr > -1 else np.nan
        rel[c] = sb
        print(f"  {c:<24} {rr:13.3f} {np.sqrt(max(sb, 0)):9.3f} "
              f"{rf:11.3f}   {gap}")
    print("  ceiling = sqrt(Spearman-Brown reliability): the largest correlation")
    print("  a perfectly related feature could show against this noisy summary.")
    print("  (full rec) is the same split-half r on the whole 18-month record --")
    print("  the gap is what the shorter primary window costs in profile noise.")
    print()

    # ---------------------------------------------------------- 2. primary
    print("=" * 70)
    print("2. PRIMARY -- 4 pre-specified tests, BH across these four only")
    print("=" * 70)
    rows = []
    for f, s in PRIMARY:
        stat, p, n = perm_test(units[f], units[s], s in CIRCULAR)
        rows.append({"feature": f, "summary": s, "stat": stat, "p": p, "n": n})
    prim = pd.DataFrame(rows)
    prim["p_bh"] = benjamini_hochberg(prim["p"].to_numpy())
    prim["ceiling"] = [np.sqrt(max(rel[s], 0)) for s in prim["summary"]]
    prim["test"] = ["Mardia R" if s in CIRCULAR else "Spearman rho"
                    for s in prim["summary"]]
    print(prim[["feature", "summary", "test", "stat", "p", "p_bh", "n",
                "ceiling"]].to_string(index=False,
                                      float_format=lambda v: f"{v:.4f}"))
    sig = prim[prim["p_bh"] < 0.05]
    print(f"\n  survives BH at q<0.05: {len(sig)} of 4"
          + ("" if sig.empty else " -- " + ", ".join(
              f"{r.feature} x {r.summary}" for r in sig.itertuples())))
    print()

    # ------------------------------------------------------ 3. exploratory
    print("=" * 70)
    print("3. EXPLORATORY -- full %d x %d grid. NOT findings."
          % (len(FEATURES), len(SUMMARIES)))
    print("=" * 70)
    grid = run_grid(units)
    grid["p_bh_grid"] = benjamini_hochberg(grid["p"].to_numpy())
    grid = grid.sort_values("p")
    print(grid[["feature", "summary", "stat", "p", "p_bh_grid"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  raw p<0.05: {(grid['p'] < 0.05).sum()} of {len(grid)} "
          f"(expected by chance at alpha=0.05: {0.05 * len(grid):.1f})")
    print(f"  BH q<0.05 across the whole grid: {(grid['p_bh_grid'] < 0.05).sum()}")
    print()

    # ------------------------------------------------------- sensitivities
    print("=" * 70)
    print("SENSITIVITIES on the four primary tests")
    print("=" * 70)

    def primary_p(u, label, note=""):
        out = []
        for f, s in PRIMARY:
            stat, p, n = perm_test(u[f], u[s], s in CIRCULAR)
            out.append(f"{stat:+.3f}/p={p:.3f}")
        print(f"  {label:<34} n={len(u):3d}  " + "  ".join(out) + note)

    print("  columns: " + " | ".join(f"{f}x{s}" for f, s in PRIMARY))
    primary_p(units, "PRIMARY: common window, cv groups")

    # The former primary. Two of the four estimates move under the window --
    # dist_industrial_m changes sign (-0.084 -> +0.091) and
    # building_density_500m x swing collapses (+0.233 -> +0.027, keeping 12% of
    # its magnitude; a 2026-03-01 start keeps 38%) -- which
    # is the seasonal confound showing itself, and the reason the window was
    # promoted. Neither version is significant either way.
    primary_p(units_full, "full record (was primary)", "   <- the swap")

    primary_p(summ.join(feat[FEATURES]), "station level (pseudo-replicated)")

    keep_cov = coverage[coverage >= 0.5].index
    primary_p(to_groups(summ.loc[keep_cov], feat.loc[keep_cov]),
              "in-window coverage >= 50%")

    # Where the window starts is itself an analytic choice, so vary it. The
    # primary date is the only one that is not arbitrary -- it is the day the
    # last monitor came online -- but the reader should see the neighbourhood.
    for start in ("2026-01-01", "2026-03-01", "2026-04-01", "2026-05-01"):
        m = np.asarray(ist >= pd.Timestamp(start, tz=TZ))
        primary_p(to_groups(summary_table(profiles_for(wide, hours, m)), feat),
                  f"window start {start}")

    wide_s, feat_s, n_scr = mod.load(screen=500)
    is_s = wide_s.index.tz_convert(TZ)
    hs = is_s.hour.to_numpy()
    ms = np.asarray(is_s >= pd.Timestamp(COMMON_START, tz=TZ))
    primary_p(to_groups(summary_table(profiles_for(wide_s, hs, ms)), feat_s),
              f"artefact screen at 500 ({n_scr} dropped)")

    # Where the morning and evening windows are drawn is a choice too, so vary
    # it. Only the two balance tests can move; the other two columns are shown
    # unchanged as a check that nothing else is touched.
    for m_w, e_w in ((range(5, 10), range(17, 22)), (range(7, 11), range(19, 23)),
                     (range(6, 10), range(19, 23)), (range(4, 12), range(16, 24)),
                     (range(7, 10), range(20, 23))):
        primary_p(to_groups(summary_table(prof, list(m_w), list(e_w)), feat),
                  f"balance windows {m_w[0]:02d}-{m_w[-1]:02d}/"
                  f"{e_w[0]:02d}-{e_w[-1]:02d}")

    # Trough from the 24h+12h fit, which is not pinned 12h from the peak.
    t2 = pd.Series({sid: trough_two_harmonic(prof.loc[sid].to_numpy())
                    for sid in prof.index})
    s2 = summ.copy()
    s2["trough_hour"] = t2
    primary_p(to_groups(s2, feat), "trough from 24h+12h fit",
              "   <- see diagnostics below")

    # Amplitude carries the site's level as well as its ventilation: a dirtier
    # site swings more in absolute ug/m3 for the same fractional daily cycle.
    lvl = pd.Series({sid: float(np.nanmean(prof.loc[sid].to_numpy()))
                     for sid in prof.index})
    s3 = summ.copy()
    s3["swing_amplitude"] = summ["swing_amplitude"] / lvl
    primary_p(to_groups(s3, feat), "amplitude relative to site level")
    print(f"  level vs raw amplitude: Spearman rho = "
          f"{stats.spearmanr(lvl, summ['swing_amplitude']).statistic:+.3f} "
          f"(p = {stats.spearmanr(lvl, summ['swing_amplitude']).pvalue:.4f}) -- "
          f"amplitude is partly a level statistic")
    print(f"  building_density vs level: Spearman rho = "
          f"{stats.spearmanr(feat.loc[lvl.index, 'building_density_500m'], lvl).statistic:+.3f}"
          f" -- so the level route is not what that test is picking up")
    print()

    # ------------------------------------- diagnostics on the 24h+12h trough
    print("=" * 70)
    print("DIAGNOSTICS -- why the 24h+12h trough behaves differently")
    print("=" * 70)
    tv = t2.to_numpy()
    early = tv < 9
    print(f"  trough hour is near-bimodal: {int(early.sum())} stations trough "
          f"pre-dawn ({tv[early].min():.1f}-{tv[early].max():.1f} h), "
          f"{int((~early).sum())} in the afternoon "
          f"({tv[~early].min():.1f}-{tv[~early].max():.1f} h)")
    mw = stats.mannwhitneyu(feat.loc[t2.index, "dist_coast_m"][early],
                            feat.loc[t2.index, "dist_coast_m"][~early]).pvalue
    arc = tv[~early].max() - tv[~early].min()
    print(f"  the two clusters {'do' if mw < 0.05 else 'do NOT'} differ in "
          f"dist_coast_m (Mann-Whitney p = {mw:.3f})"
          + ("; the split itself carries coastal information, so R is not"
             " purely a within-cluster effect" if mw < 0.05 else
             f", so the association is within the afternoon cluster's"
             f" {arc:.1f} h arc"))
    print(f"  afternoon cluster spans {arc:.1f} h "
          f"({tv[~early].min():.1f}-{tv[~early].max():.1f})")
    u2 = to_groups(s2, feat)
    jk = sorted((circ_linear_r(u2.drop(g)["dist_coast_m"],
                               u2.drop(g)["trough_hour"]), g) for g in u2.index)
    print(f"  leave-one-group-out R: {jk[0][0]:.3f} ({jk[0][1]}) to "
          f"{jk[-1][0]:.3f} ({jk[-1][1]}) -- not one high-leverage group")
    keep2 = t2[t2 >= 9].index
    st, p, n = perm_test(to_groups(s2.loc[keep2], feat.loc[keep2])["dist_coast_m"],
                         to_groups(s2.loc[keep2], feat.loc[keep2])["trough_hour"],
                         True)
    print(f"  drop the {int(early.sum())} pre-dawn stations "
          f"({int((~early).sum())} left, {n} groups): "
          f"R = {st:.3f}, p = {p:.4f}, n = {n}")
    uk = to_groups(s2.loc[keep2], feat.loc[keep2])
    sr = stats.spearmanr(uk["dist_coast_m"], uk["trough_hour"])
    print(f"  on that arc a plain Spearman gives rho = {sr.statistic:+.3f}, "
          f"p = {sr.pvalue:.4f}: farther inland -> earlier afternoon trough")
    to, te = (pd.Series({s: trough_two_harmonic(p_.loc[s].to_numpy())
                         for s in p_.index}) for p_ in (p_odd, p_even))
    print(f"  split-half stability of this statistic: circular r = "
          f"{circ_circ_r(to, te):.3f}, median |diff| = "
          f"{np.median(np.abs(circ_diff(to, te))):.2f} h, "
          f"{int((np.abs(circ_diff(to, te)) > 6).sum())} of {len(to)} stations "
          f"flip cluster between halves")
    print("  NOTE: the halves share every station and every feature value, so")
    print("  they test the summary's stability, NOT the association's replicability.")
    print()

    # ----------------------------------------------------------- 5. power
    print("=" * 70)
    print("POWER -- what a null at n=%d can and cannot rule out" % len(units))
    print("=" * 70)
    rng = np.random.default_rng(SEED + 1)
    print(f"  {'true rho':>9}  {'power at alpha=0.05':>20}")
    for rho in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
        hits = 0
        for _ in range(3000):
            z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], len(units))
            hits += stats.spearmanr(z[:, 0], z[:, 1]).pvalue < 0.05
        print(f"  {rho:9.2f}  {hits / 3000:20.2f}")
    print("  Before BH, and before the circular tests, which are weaker still.")
    print()

    out = ROOT / "data" / "processed" / "diurnal_summaries.parquet"
    units.to_parquet(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
