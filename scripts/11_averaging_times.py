"""Does spatial structure in PM2.5 appear at longer averaging times?

Pre-registered in reports/preregistration_averaging_times.md before running.

The main result is that hourly PM2.5, with the citywide mean removed, has
almost no spatial structure at this network's spacing. The objection is that
hourly anomalies are noisy, and that stable differences between places could
be hidden at 1 h and only show in daily, weekly or monthly means. So the
same questions are asked again at eleven averaging times, from 1 h to the
whole record:

  (a) spatial share      how much of a window's anomaly variance is a stable
                         difference between groups (ANOVA intraclass corr.)
  (b) reliability        (b1) odd/even split-half of the long-run pattern;
                         (b2, primary) how alike two separate windows'
                         patterns are - the noise check
  (c) distance pattern   are nearby monitors more alike than distant ones?
  (d) prediction skill   grouped spatial CV of IDW against the city average

The unit is the cv_group (35), except where distance makes stations the
natural unit. Long windows are few, and every row says so.

Writes reports/averaging_times.csv, reports/averaging_times_distance.csv and
reports/figures/averaging_time_structure.png.
"""

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
TZ = "Asia/Kolkata"
OUT_CSV = ROOT / "reports" / "averaging_times.csv"
OUT_DIST = ROOT / "reports" / "averaging_times_distance.csv"
OUT_FIG = ROOT / "reports" / "figures" / "averaging_time_structure.png"

# (label, spec): an int is a fixed length in hours, anchored at the first IST
# midnight of the record; the rest are calendar units.
WINDOWS = [("1h", 1), ("3h", 3), ("8h", 8), ("24h", 24), ("2d", 48),
           ("3d", 72), ("1w", 168), ("2w", 336), ("month", "month"),
           ("season", "season"), ("all", "all")]

COVERAGE = 0.75            # share of a window's nominal hours a station needs
COVERAGE_SENSITIVITY = 0.5
MIN_STATIONS = 10          # stations passing coverage before a window counts

MIN_GAP = pd.Timedelta("7D")   # (b2) end of one window to start of the other
MIN_PAIR_GROUPS = 10
MAX_PAIRS = 50_000

MIN_SHARED = 3             # (c) windows a station pair must share
DIST_BINS = (0, 2, 5, 10, 15, 20, 30, np.inf)
NEAR_KM, FAR_KM = 5.0, 15.0

N_BOOT = 1000
N_PERM = 999
SEED = 0
CONTRAST = ("1h", "1w")

# IMD seasons. DJF belongs to the year its January falls in.
SEASON_OF = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
             6: "JJAS", 7: "JJAS", 8: "JJAS", 9: "JJAS", 10: "ON", 11: "ON"}
SEASON_MONTHS = {"DJF": (-1, 12, 0, 1, 0, 2), "MAM": (0, 3, 0, 4, 0, 5),
                 "JJAS": (0, 6, 0, 7, 0, 8, 0, 9), "ON": (0, 10, 0, 11)}


def load_engine():
    spec = importlib.util.spec_from_file_location(
        "interpolate", ROOT / "model" / "interpolate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def thin_label(n_windows):
    if n_windows >= 30:
        return "interpretable"
    if n_windows >= 8:
        return "thin"
    return "too thin"


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------

def hourly_grid(wide):
    """The panel on a complete IST hourly grid, starting at the first midnight.

    The source stamps are whole UTC hours, so in IST they sit on :30. The grid
    keeps that offset, and the anchor is the first IST day start at the same
    minute (00:30 here). That can precede the first reading (02:30 IST on
    2025-02-28); the hours before it are simply missing, and the coverage rule
    deals with them like any other gap.
    """
    t = wide.index.tz_convert(TZ)
    offset = t[0] - t[0].floor("h")
    anchor = t[0].floor("D") + offset
    grid = pd.date_range(anchor, t[-1], freq="h")
    out = wide.copy()
    out.index = t
    return out.reindex(grid), anchor


def _month_start(year, month, offset):
    return pd.Timestamp(year=year, month=month, day=1, tz=TZ) + offset


def window_labels(times, spec, anchor):
    """Window label for every hour, and each label's start, end and nominal hours."""
    offset = anchor - anchor.floor("D")
    if isinstance(spec, int):
        k = np.asarray((times - anchor) // pd.Timedelta(hours=spec), dtype=int)
        meta = {i: (anchor + pd.Timedelta(hours=spec * i), spec)
                for i in np.unique(k)}
        labels = k
    elif spec == "month":
        labels = np.asarray(times.strftime("%Y-%m"))
        meta = {}
        for lab in np.unique(labels):
            y, m = map(int, lab.split("-"))
            meta[lab] = (_month_start(y, m, offset),
                         pd.Period(lab, "M").days_in_month * 24)
    elif spec == "season":
        year = np.where(times.month == 12, times.year + 1, times.year)
        labels = np.asarray([f"{y}-{SEASON_OF[m]}"
                             for y, m in zip(year, times.month)])
        meta = {}
        for lab in np.unique(labels):
            y, s = lab.split("-")
            y = int(y)
            ms = SEASON_MONTHS[s]
            months = [(y + ms[i], ms[i + 1]) for i in range(0, len(ms), 2)]
            hours = sum(pd.Period(f"{a}-{b:02d}", "M").days_in_month * 24
                        for a, b in months)
            meta[lab] = (_month_start(*months[0], offset), hours)
    elif spec == "all":
        labels = np.zeros(len(times), dtype=int)
        meta = {0: (times[0], len(times))}
    else:
        raise ValueError(f"unknown window spec {spec!r}")
    m = pd.DataFrame.from_dict(meta, orient="index",
                               columns=["start", "nominal_hours"])
    m["end"] = m["start"] + pd.to_timedelta(m["nominal_hours"], unit="h")
    return labels, m


def window_means(grid, spec, anchor, coverage=COVERAGE,
                 min_stations=MIN_STATIONS):
    """Station means per window, NaN where a station misses the coverage rule.

    Coverage is judged against the window's NOMINAL length, so a stub at
    either end of the record fails rather than passing on a few hours.
    """
    labels, meta = window_labels(grid.index, spec, anchor)
    g = grid.groupby(labels)
    mean, count = g.mean(), g.count()
    meta = meta.loc[mean.index]
    need = np.ceil(coverage * meta["nominal_hours"].to_numpy() - 1e-9)
    X = mean.where(count.to_numpy() >= need[:, None])
    keep = X.notna().sum(axis=1).to_numpy() >= min_stations
    return X[keep], meta[keep]


def anomalies(X):
    """Remove each window's citywide mean; standardised and in ug/m3."""
    c = X.mean(axis=1)
    raw = X.sub(c, axis=0)
    sd = X.std(axis=1, ddof=0)
    z = raw.div(sd.where(sd > 1e-9), axis=0)
    return z, raw


def by_group(df, groups):
    """Mean of member stations, per window and cv_group."""
    return df.T.groupby(np.asarray(groups)).mean().T


# --------------------------------------------------------------------------
# (a) spatial share
# --------------------------------------------------------------------------

def spatial_share(Y):
    """Raw between-group share and the ANOVA intraclass correlation.

    Y is windows x groups with NaN. The raw share is the variance of group
    means over the total; it drifts to 1 as windows get fewer, even for pure
    noise. The ICC subtracts the noise a finite mean carries and stays near 0
    for pure noise at any window count. Returns (raw, icc, n_windows).
    """
    Y = np.asarray(Y, dtype=float)
    n_w = Y.shape[0]
    ok = np.isfinite(Y)
    n_g = ok.sum(axis=0)
    cols = n_g > 0
    Y, ok, n_g = Y[:, cols], ok[:, cols], n_g[cols]
    vals = Y[ok]
    if len(vals) < 2:
        return np.nan, np.nan, n_w
    m = np.nansum(Y, axis=0) / n_g
    total = np.var(vals)
    raw = np.var(m) / total if total > 0 else np.nan

    k, N = len(n_g), n_g.sum()
    if n_w < 2 or N <= k or k < 2:
        return raw, np.nan, n_w
    grand = vals.mean()
    ssb = float(np.sum(n_g * (m - grand) ** 2))
    ssw = float(np.nansum((Y - m) ** 2))
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - np.sum(n_g ** 2) / N) / (k - 1)
    den = msb + (n0 - 1) * msw
    return raw, (msb - msw) / den if den > 0 else np.nan, n_w


# --------------------------------------------------------------------------
# (b) reliability
# --------------------------------------------------------------------------

def split_half(Y, parity):
    """(b1) Correlate group means over odd windows with those over even ones."""
    Y = np.asarray(Y, dtype=float)
    parity = np.asarray(parity)
    if (parity == 0).sum() == 0 or (parity == 1).sum() == 0:
        return np.nan
    with np.errstate(invalid="ignore"):
        a = np.nanmean(np.where(parity[:, None] == 0, Y, np.nan), axis=0)
        b = np.nanmean(np.where(parity[:, None] == 1, Y, np.nan), axis=0)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def eligible_pairs(starts, ends, min_gap=MIN_GAP, max_pairs=MAX_PAIRS,
                   rng=None):
    """Window pairs (i < j in time) separated by at least min_gap.

    All of them if there are few enough, otherwise a uniform random sample of
    max_pairs.
    """
    # asi8 is in the index's own unit; the source data is in microseconds.
    starts = pd.DatetimeIndex(starts).as_unit("ns")
    ends = pd.DatetimeIndex(ends).as_unit("ns")
    order = np.argsort(starts.asi8)
    s, e = starts.asi8[order], ends.asi8[order]
    gap = min_gap.value
    # first j whose start is >= end_i + gap; starts are sorted
    first = np.searchsorted(s, e + gap, side="left")
    first = np.maximum(first, np.arange(len(s)) + 1)
    counts = len(s) - first
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, int), np.empty(0, int)
    if total <= max_pairs:
        i = np.repeat(np.arange(len(s)), counts)
        j = np.concatenate([np.arange(f, len(s)) for f in first])
    else:
        rng = rng or np.random.default_rng(SEED)
        # Sampling uniformly over eligible pairs: pick i in proportion to how
        # many partners it has, then a partner uniformly.
        pick = rng.choice(total, size=max_pairs, replace=False)
        cum = np.cumsum(counts)
        i = np.searchsorted(cum, pick, side="right")
        within = pick - (cum[i] - counts[i])
        j = first[i] + within
    return order[i], order[j]


def pair_correlations(Y, i, j, min_groups=MIN_PAIR_GROUPS):
    """(b2) Across-group Pearson r between window i's pattern and window j's."""
    Y = np.asarray(Y, dtype=float)
    r = np.full(len(i), np.nan)
    for lo in range(0, len(i), 5000):
        a, b = Y[i[lo:lo + 5000]], Y[j[lo:lo + 5000]]
        ok = np.isfinite(a) & np.isfinite(b)
        n = ok.sum(axis=1)
        a0, b0 = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ma = a0.sum(1) / n
            mb = b0.sum(1) / n
            da = np.where(ok, a - ma[:, None], 0.0)
            db = np.where(ok, b - mb[:, None], 0.0)
            rr = (da * db).sum(1) / np.sqrt((da ** 2).sum(1) * (db ** 2).sum(1))
        rr[n < min_groups] = np.nan
        r[lo:lo + 5000] = rr
    keep = np.isfinite(r)
    return i[keep], j[keep], r[keep]


def spearman_brown(r1, hours):
    """Reliability at `hours` if hourly noise were independent."""
    return hours * r1 / (1 + (hours - 1) * r1)


# --------------------------------------------------------------------------
# (c) distance pattern
# --------------------------------------------------------------------------

def pair_sq_diffs(Z):
    """Half squared differences of z, windows x station-pairs (upper triangle)."""
    Z = np.asarray(Z, dtype=float)
    a, b = np.triu_indices(Z.shape[1], k=1)
    sq = 0.5 * (Z[:, a] - Z[:, b]) ** 2
    return sq, a, b


def pooled_corr(sq, weights=None):
    """1 - semivariance per station pair, pooled over (weighted) windows."""
    ok = np.isfinite(sq)
    w = np.ones(sq.shape[0]) if weights is None else np.asarray(weights, float)
    num = w @ np.where(ok, sq, 0.0)
    den = w @ ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - num / den, ok.sum(axis=0)


def near_far(corr, d, same):
    near = (~same) & (d < NEAR_KM)
    far = (~same) & (d >= FAR_KM)
    if near.sum() == 0 or far.sum() == 0:
        return np.nan, int(near.sum()), int(far.sum())
    return (float(np.nanmean(corr[near]) - np.nanmean(corr[far])),
            int(near.sum()), int(far.sum()))


def distance_bins(corr, d, same, n_shared):
    rows = [{"bin": "same cv_group", "n_pairs": int(same.sum()),
             "mean_km": float(d[same].mean()) if same.any() else np.nan,
             "corr": float(np.mean(corr[same])) if same.any() else np.nan}]
    for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
        m = (~same) & (d >= lo) & (d < hi)
        label = f"{lo:g}-{hi:g} km" if np.isfinite(hi) else f">= {lo:g} km"
        rows.append({"bin": label, "n_pairs": int(m.sum()),
                     "mean_km": float(d[m].mean()) if m.any() else np.nan,
                     "corr": float(np.mean(corr[m])) if m.any() else np.nan})
    return pd.DataFrame(rows)


def mantel_p(corr, a, b, dist, same_mat, n_perm=N_PERM, rng=None):
    """Permute stations over locations; how often is near-far this large?"""
    rng = rng or np.random.default_rng(SEED)
    obs, _, _ = near_far(corr, dist[a, b], same_mat[a, b])
    if not np.isfinite(obs):
        return np.nan
    hits = 0
    n = dist.shape[0]
    for _ in range(n_perm):
        p = rng.permutation(n)
        v, _, _ = near_far(corr, dist[p[a], p[b]], same_mat[p[a], p[b]])
        hits += np.isfinite(v) and v >= obs
    return (hits + 1) / (n_perm + 1)


# --------------------------------------------------------------------------
# (d) prediction skill
# --------------------------------------------------------------------------

def cv_errors(mod, X, dist, groups):
    """Leave-one-cv_group-out errors of IDW and the city average, per window.

    Returns windows x stations error matrices (NaN where nothing was scored)
    and the IDW power picked in each fold.
    """
    values = np.asarray(X, dtype=float)
    avail = np.isfinite(values)
    groups = np.asarray(groups)
    e_idw = np.full(values.shape, np.nan)
    e_city = np.full(values.shape, np.nan)
    powers = {}
    for g in sorted(set(groups)):
        targets = np.flatnonzero(groups == g)
        sources = np.flatnonzero(groups != g)
        power = mod.select_idw_power(values, avail, dist, sources, groups)
        powers[g] = power
        n_src = avail[:, sources].sum(1)
        city = np.where(n_src > 0, np.nansum(values[:, sources], 1)
                        / np.maximum(n_src, 1), np.nan)
        for t in targets:
            est = mod.idw_estimate(values, avail, dist[t], sources, power)
            obs = values[:, t]
            ok = np.isfinite(est) & np.isfinite(obs) & np.isfinite(city)
            e_idw[ok, t] = est[ok] - obs[ok]
            e_city[ok, t] = city[ok] - obs[ok]
    return e_idw, e_city, powers


def skill(e_idw, e_city, weights=None):
    """Pooled RMSEs, the gap and the gap as % of the city-average RMSE."""
    w = np.ones(e_idw.shape[0]) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(e_idw)
    n = w @ ok.sum(1)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    r_i = np.sqrt((w @ np.nansum(e_idw ** 2, 1)) / n)
    r_c = np.sqrt((w @ np.nansum(e_city ** 2, 1)) / n)
    return r_i, r_c, r_i - r_c, 100 * (r_i - r_c) / r_c


def group_gaps(e_idw, e_city, groups):
    out = {}
    for g in sorted(set(groups)):
        cols = np.asarray(groups) == g
        a, b = e_idw[:, cols], e_city[:, cols]
        ok = np.isfinite(a)
        if ok.sum() == 0:
            continue
        out[g] = np.sqrt(np.mean(a[ok] ** 2)) - np.sqrt(np.mean(b[ok] ** 2))
    return pd.Series(out)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def month_resamples(months, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    return [rng.choice(months, size=len(months), replace=True)
            for _ in range(n_boot)]


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan, np.nan
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def analyse_window(mod, grid, anchor, feat, dist, label, spec, coverage,
                   resamples, boot=True):
    """Every metric for one averaging time. Returns (row, distance table, boots)."""
    t0 = time.time()
    X, meta = window_means(grid, spec, anchor, coverage)
    stations = feat.index.to_numpy()
    groups = feat.loc[X.columns, "cv_group"].to_numpy()
    z, raw = anomalies(X)
    zg = by_group(z, groups)
    rawg = by_group(raw, groups)
    Y = zg.to_numpy()
    n_w = len(X)
    tl = thin_label(n_w)

    row = {"window": label, "coverage": coverage, "data": tl,
           "n_windows": n_w,
           "n_stations": int(X.notna().any().sum()),
           "n_groups": int(zg.notna().any().sum()),
           "median_stations_per_window": float(X.notna().sum(axis=1).median()),
           "median_groups_per_window": float(zg.notna().sum(axis=1).median()),
           "median_hours": float(meta["nominal_hours"].median())}

    # (a)
    share_raw, icc, _ = spatial_share(Y)
    _, icc_ug, _ = spatial_share(rawg.to_numpy())
    row.update(share_raw=share_raw, share_icc=icc, share_icc_ugm3=icc_ug)

    # (b1)
    parity = np.arange(n_w) % 2
    row["split_half_r"] = split_half(Y, parity) if n_w >= 2 else np.nan

    # (b2)
    rng = np.random.default_rng(SEED)
    pi, pj = eligible_pairs(meta["start"], meta["end"], rng=rng)
    pi, pj, pr = pair_correlations(Y, pi, pj)
    row.update(reliability_mean=float(pr.mean()) if len(pr) else np.nan,
               reliability_median=float(np.median(pr)) if len(pr) else np.nan,
               reliability_pairs=len(pr),
               reliability_windows=len(np.union1d(pi, pj)))

    # (c)
    sq, a, b = pair_sq_diffs(z.to_numpy())
    col = np.asarray([np.flatnonzero(stations == s)[0] for s in X.columns])
    d = dist[np.ix_(col, col)]
    gfull = feat["cv_group"].to_numpy()
    same_mat = gfull[col][:, None] == gfull[col][None, :]
    corr, n_shared = pooled_corr(sq)
    need = 1 if n_w < MIN_SHARED else MIN_SHARED
    use = n_shared >= need
    corr_u, a_u, b_u = corr[use], a[use], b[use]
    dpair, same = d[a_u, b_u], same_mat[a_u, b_u]
    contrast, n_near, n_far = near_far(corr_u, dpair, same)
    p_mantel = mantel_p(corr_u, a_u, b_u, d, same_mat,
                        rng=np.random.default_rng(SEED))
    dtab = distance_bins(corr_u, dpair, same, n_shared[use])
    dtab.insert(0, "window", label)
    dtab.insert(1, "coverage", coverage)
    dtab["n_windows"] = n_w
    row.update(near_minus_far=contrast, near_pairs=n_near, far_pairs=n_far,
               mantel_p=p_mantel)

    # (d)
    e_idw, e_city, powers = cv_errors(mod, X.to_numpy(), d, groups)
    r_i, r_c, gap, pct = skill(e_idw, e_city)
    gg = group_gaps(e_idw, e_city, groups)
    nz = gg[gg.abs() > 1e-12]
    p_w = stats.wilcoxon(nz).pvalue if len(nz) >= 6 else np.nan
    row.update(idw_power_median=float(np.median(list(powers.values()))),
               rmse_city=r_c, rmse_idw=r_i, gap=gap, gap_pct=pct,
               group_gap_mean=float(gg.mean()), group_gap_wilcoxon_p=p_w,
               groups_won_by_idw=int((gg < 0).sum()), cv_groups=len(gg),
               cv_station_windows=int(np.isfinite(e_idw).sum()))

    # Month-cluster bootstrap: windows are weighted by how often the month
    # they start in was drawn. Season and whole-record get none.
    boots = {}
    if boot and label not in ("season", "all") and resamples:
        wmonth = pd.DatetimeIndex(meta["start"]).strftime("%Y-%m").to_numpy()
        keys = ("share_icc", "split_half_r", "reliability_mean",
                "near_minus_far", "gap", "gap_pct")
        boots = {k: np.full(len(resamples), np.nan) for k in keys}
        ok_sq = np.isfinite(sq[:, use])
        sq_u = np.where(ok_sq, sq[:, use], 0.0)
        for bi, draw in enumerate(resamples):
            mcount = pd.Series(draw).value_counts()
            w = mcount.reindex(wmonth).fillna(0).to_numpy()
            if w.sum() == 0:
                continue
            rep = np.repeat(np.arange(n_w), w.astype(int))
            boots["share_icc"][bi] = spatial_share(Y[rep])[1]
            boots["split_half_r"][bi] = split_half(Y[rep], parity[rep])
            pw = w[pi] * w[pj]
            if pw.sum() > 0:
                boots["reliability_mean"][bi] = np.sum(pw * pr) / pw.sum()
            den = w @ ok_sq
            with np.errstate(invalid="ignore", divide="ignore"):
                cb = 1.0 - (w @ sq_u) / den
            boots["near_minus_far"][bi] = near_far(cb, dpair, same)[0]
            _, _, g_b, p_b = skill(e_idw, e_city, w)
            boots["gap"][bi], boots["gap_pct"][bi] = g_b, p_b
        for k in keys:
            row[f"{k}_lo"], row[f"{k}_hi"] = ci(boots[k])

    print(f"  {label:7s} cov {coverage:.2f}  {n_w:6d} windows  "
          f"{row['n_groups']:2d} groups  ({time.time() - t0:5.1f}s)",
          flush=True)
    return row, dtab, boots


def reading(t):
    """Apply the pre-registered decision rules to one coverage level."""
    t = t.set_index("window")
    short, long_ = CONTRAST
    c1 = t.loc[long_, "c1"], t.loc[long_, "c1_lo"], t.loc[long_, "c1_hi"]
    c2 = t.loc[long_, "c2"], t.loc[long_, "c2_lo"], t.loc[long_, "c2_hi"]
    interp = t[t["data"] == "interpretable"]
    b2_1w = t.loc[long_, "reliability_mean"]
    c1_pos = c1[1] > 0
    c2_neg = c2[2] < 0
    mantel = t.loc[long_, "mantel_p"]
    smooth = (t.loc[long_, "near_minus_far"] > 0) and mantel < 0.05

    if (not c1_pos) or (interp["reliability_mean"] < 0.2).all():
        verdict = "1: noise all the way down"
    elif c1_pos and b2_1w >= 0.3 and not smooth and not c2_neg:
        verdict = "2: stable but local"
    elif c1_pos and b2_1w >= 0.3 and smooth and c2_neg:
        verdict = "3: stable and smooth"
    else:
        verdict = "mixed"
    return {"C1": c1, "C2": c2, "b2_1w": b2_1w, "mantel_1w": mantel,
            "near_minus_far_1w": t.loc[long_, "near_minus_far"],
            "verdict": verdict}


def run(mod, grid, anchor, feat, dist, coverage):
    months = pd.Index(grid.index.strftime("%Y-%m")).unique().to_numpy()
    resamples = month_resamples(months)
    rows, dtabs, boots = [], [], {}
    for label, spec in WINDOWS:
        row, dtab, b = analyse_window(mod, grid, anchor, feat, dist, label,
                                      spec, coverage, resamples)
        rows.append(row)
        dtabs.append(dtab)
        boots[label] = b
    t = pd.DataFrame(rows)

    # Spearman-Brown reference from the 1 h reliability.
    r1 = t.loc[t["window"] == "1h", "reliability_mean"].iloc[0]
    t["reliability_spearman_brown"] = spearman_brown(r1, t["median_hours"])

    # The two contrasts, paired over the same month resamples.
    short, long_ = CONTRAST
    t["c1"] = t["c1_lo"] = t["c1_hi"] = np.nan
    t["c2"] = t["c2_lo"] = t["c2_hi"] = np.nan
    s, l_ = t.set_index("window").loc[short], t.set_index("window").loc[long_]
    c1 = l_["reliability_mean"] - s["reliability_mean"]
    c2 = l_["gap_pct"] - s["gap_pct"]
    c1_lo, c1_hi = ci(boots[long_]["reliability_mean"]
                      - boots[short]["reliability_mean"])
    c2_lo, c2_hi = ci(boots[long_]["gap_pct"] - boots[short]["gap_pct"])
    at = t["window"] == long_
    t.loc[at, ["c1", "c1_lo", "c1_hi", "c2", "c2_lo", "c2_hi"]] = \
        [c1, c1_lo, c1_hi, c2, c2_lo, c2_hi]
    return t, pd.concat(dtabs, ignore_index=True)


def window_hours(t):
    return t["median_hours"].to_numpy(dtype=float)


def figure(t, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid_c = "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange, ref = "#2a78d6", "#eb6834", "#8a8984"
    x = window_hours(t)
    solid = (t["data"] == "interpretable").to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")

    def series(ax, y, lo, hi, color, name):
        y = np.asarray(y, float)
        ax.plot(x, y, color=color, lw=2, zorder=3, label=name)
        if lo is not None:
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0, zorder=2)
        ax.scatter(x[solid], y[solid], s=36, color=color, zorder=4,
                   edgecolor="white", linewidth=1.5)
        ax.scatter(x[~solid], y[~solid], s=36, facecolor="white",
                   edgecolor=color, linewidth=1.5, zorder=4)

    ax = axes[0]
    series(ax, t["share_icc"], t.get("share_icc_lo"), t.get("share_icc_hi"),
           blue, "noise-corrected share")
    ax.set_title("(a) Spatial share of anomaly variance", loc="left",
                 fontsize=11, color=ink)
    ax.set_ylabel("intraclass correlation", color=muted)

    ax = axes[1]
    series(ax, t["reliability_mean"], t.get("reliability_mean_lo"),
           t.get("reliability_mean_hi"), blue,
           "b2: two separate windows (primary)")
    series(ax, t["split_half_r"], None, None, orange,
           "b1: odd vs even split-half")
    ax.plot(x, t["reliability_spearman_brown"], color=ref, lw=1.2, ls="--",
            label="b2 if hourly noise were independent")
    ax.set_title("(b) Reliability of the spatial pattern", loc="left",
                 fontsize=11, color=ink)
    ax.set_ylabel("correlation across groups", color=muted)
    ax.set_ylim(0, 1.38)                       # headroom for the legend
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[2]
    series(ax, t["gap_pct"], t.get("gap_pct_lo"), t.get("gap_pct_hi"), blue,
           "IDW vs city average")
    ax.axhline(0, color=muted, lw=1)
    ax.set_title("(d) IDW RMSE minus city average", loc="left", fontsize=11,
                 color=ink)
    ax.set_ylabel("% of city-average RMSE (below 0 = IDW wins)", color=muted)

    ticks = x
    names = t["window"].tolist()
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels(names, rotation=45, fontsize=8, color=muted)
        ax.minorticks_off()
        ax.grid(True, color=grid_c, lw=0.8)
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid_c)
        ax.tick_params(colors=muted, labelsize=8)
        ax.set_xlabel("averaging time (log scale)", color=muted)
    fig.text(0.01, 0.01,
             "Filled markers: >= 30 windows. Hollow: thin (month) or too thin "
             "(season, whole record). Bands: 95% month-cluster bootstrap. "
             "75% coverage, 35 cv_groups.", fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_table(t):
    cols = ["window", "data", "n_windows", "n_groups", "share_raw",
            "share_icc", "split_half_r", "reliability_mean",
            "reliability_pairs", "near_minus_far", "mantel_p", "rmse_city",
            "rmse_idw", "gap", "gap_pct", "group_gap_wilcoxon_p"]
    with pd.option_context("display.width", 250, "display.max_columns", 50,
                           "display.float_format", "{:.3f}".format):
        print(t[cols].to_string(index=False))


def main():
    mod = load_engine()
    wide, feat, _ = mod.load()
    grid, anchor = hourly_grid(wide)
    dist = mod.distance_matrix(feat["lat"].to_numpy(), feat["lon"].to_numpy())
    print(f"{grid.shape[1]} stations, {feat['cv_group'].nunique()} groups, "
          f"{len(grid):,} hours from {anchor:%Y-%m-%d %H:%M} IST")

    tables, dists = [], []
    for cov in (COVERAGE, COVERAGE_SENSITIVITY):
        print(f"\ncoverage {cov:.0%}")
        t, dt = run(mod, grid, anchor, feat, dist, cov)
        tables.append(t)
        dists.append(dt)
    table = pd.concat(tables, ignore_index=True)
    dtab = pd.concat(dists, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False, float_format="%.5g")
    dtab.to_csv(OUT_DIST, index=False, float_format="%.5g")
    figure(tables[0], OUT_FIG)

    for t in tables:
        cov = t["coverage"].iloc[0]
        print("\n" + "=" * 78)
        print(f"coverage {cov:.0%}")
        print("=" * 78)
        print_table(t)
        r = reading(t)
        print(f"\n  C1 b2(1w) - b2(1h) = {r['C1'][0]:+.3f} "
              f"[{r['C1'][1]:+.3f}, {r['C1'][2]:+.3f}]")
        print(f"  C2 gap%(1w) - gap%(1h) = {r['C2'][0]:+.3f} "
              f"[{r['C2'][1]:+.3f}, {r['C2'][2]:+.3f}]")
        print(f"  b2(1w) = {r['b2_1w']:.3f}; near-far(1w) = "
              f"{r['near_minus_far_1w']:+.3f}, Mantel p = {r['mantel_1w']:.3f}")
        print(f"  reading: {r['verdict']}")
    with pd.option_context("display.width", 200):
        print("\ndistance pattern, 75% coverage")
        print(dists[0].pivot(index="bin", columns="window", values="corr")
              [[w for w, _ in WINDOWS]].round(3).to_string())
    print(f"\nwrote {OUT_CSV.relative_to(ROOT)}, {OUT_DIST.relative_to(ROOT)}, "
          f"{OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
