"""Satellite PM2.5 against the monitor network, 2024, as pre-registered.

Run directly: python scripts/15_satellite_vs_monitors.py

Implements docs/preregistration_satellite.md (H1, H2, H3 and the stated
secondary tests) on

  data/interim/monitors_2024_daily.parquet  (data/load_cpcb_2024.py)
  data/interim/satellite_mumbai_2024.nc     (scripts/14_crop_satellite_mumbai.py)

b2 day-pair selection, the distance pattern and the Mantel permutation reuse
scripts/11_averaging_times.py, so b2 here is the same statistic as the 24 h
b2 there. Intervals are a month-cluster bootstrap (1,000 resamples, seed 0),
with the same resamples for both datasets.

Writes reports/satellite_vs_monitors.csv (one row per metric),
reports/satellite_vs_monitors_groups.csv (per group, H3) and
reports/satellite_vs_monitors_distance.csv (distance bins).
"""

import importlib.util
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DAILY = ROOT / "data" / "interim" / "monitors_2024_daily.parquet"
SAT = ROOT / "data" / "interim" / "satellite_mumbai_2024.nc"
OUT_CSV = ROOT / "reports" / "satellite_vs_monitors.csv"
OUT_GROUPS = ROOT / "reports" / "satellite_vs_monitors_groups.csv"
OUT_DIST = ROOT / "reports" / "satellite_vs_monitors_distance.csv"

MIN_GROUPS_PER_DAY = 10
MIN_GROUP_DAYS = 60           # H3 and the confirmatory minimum
MIN_CONF_GROUPS = 15
MIN_KEPT_DAYS = 120

H1_B2, H1_A = 0.30, 3.0
H2_C = 0.50
H3_MED, H3_LO, H3_FAIL = 0.70, 0.60, 0.60

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m11 = _load("averaging_times", ROOT / "scripts" / "11_averaging_times.py")
SEED, N_BOOT, N_PERM = m11.SEED, m11.N_BOOT, m11.N_PERM


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def load_satellite():
    with netCDF4.Dataset(SAT) as ds:
        t = ds["time"][:].data
        lat, lon = ds["lat"][:].data, ds["lon"][:].data
        pm = ds["pm25"][:].filled(np.nan).astype(float)
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(t, unit="D")
    return pd.DatetimeIndex(dates), lat, lon, pm


def match_cells(stations, lat, lon, pm):
    """Nearest land cell (finite on every day) for each station."""
    land = np.isfinite(pm).all(axis=0)
    ii, jj = np.nonzero(land)
    out = []
    for r in stations.itertuples():
        d = haversine_km(r.lat, r.lon, lat[ii], lon[jj])
        k = int(np.argmin(d))
        out.append({"station_id": r.station_id, "cell_i": ii[k], "cell_j": jj[k],
                    "match_m": 1000 * d[k]})
    return pd.DataFrame(out).set_index("station_id")


def build_panels():
    daily = pd.read_parquet(DAILY)
    dates, lat, lon, pm = load_satellite()
    stations = (daily.drop_duplicates("station_id")
                     .set_index("station_id")[["site", "agency", "cv_group", "lat", "lon"]]
                     .sort_index())
    cells = match_cells(stations.reset_index(), lat, lon, pm)
    stations = stations.join(cells)

    # Station level: days x stations, monitor where valid, satellite on those days.
    Ms = (daily.pivot(index="date", columns="station_id", values="pm25")
               .reindex(index=dates, columns=stations.index))
    sat_full = pd.DataFrame({sid: pm[:, r.cell_i, r.cell_j]
                             for sid, r in stations.iterrows()}, index=dates)
    Ss = sat_full.where(Ms.notna())

    # Group level: member mean; satellite over the same members' cells.
    groups = stations["cv_group"]
    Mg = Ms.T.groupby(groups).mean().T
    Sg = Ss.T.groupby(groups).mean().T

    n_groups = Mg.notna().sum(axis=1)
    kept = n_groups >= MIN_GROUPS_PER_DAY
    land = np.isfinite(pm).all(axis=0)
    return dict(stations=stations, dates=dates, kept=kept.to_numpy(),
                Ms=Ms[kept], Ss=Ss[kept], Mg=Mg[kept], Sg=Sg[kept],
                sat_cells=pm[:, land], sat_dates=dates)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def anomalies(Y):
    return Y - np.nanmean(Y, axis=1, keepdims=True)


def amplitude(A):
    """Median over days of the across-unit SD of anomalies (ddof=0, as in script 11)."""
    with np.errstate(invalid="ignore"):
        return float(np.nanmedian(np.nanstd(A, axis=1)))


def day_pairs(dates):
    starts = pd.DatetimeIndex(dates)
    return m11.eligible_pairs(starts, starts + pd.Timedelta("1D"),
                              rng=np.random.default_rng(SEED))


def b2_pairs(A, pi, pj):
    return m11.pair_correlations(A, pi, pj, min_groups=MIN_GROUPS_PER_DAY)


def citywide_share(Y):
    """SS_between / SS_total of the day-wise decomposition."""
    Y = np.asarray(Y, float)
    ok = np.isfinite(Y)
    if ok.sum() < 2:
        return np.nan
    xbar = Y[ok].mean()
    n_d = ok.sum(axis=1)
    rows = n_d > 0
    m_d = np.nansum(Y[rows], axis=1) / n_d[rows]
    ssb = float(np.sum(n_d[rows] * (m_d - xbar) ** 2))
    sst = float(np.sum((Y[ok] - xbar) ** 2))
    return ssb / sst if sst > 0 else np.nan


def first_differences(Y, dates):
    """x_gd - x_g,d-1 for calendar-consecutive kept days; row d carries day d."""
    Y = np.asarray(Y, float)
    dates = pd.DatetimeIndex(dates)
    consec = (dates[1:] - dates[:-1]) == pd.Timedelta("1D")
    D = Y[1:] - Y[:-1]
    D[~consec] = np.nan
    return D, dates[1:]


def loo_r2(Y):
    Y = np.asarray(Y, float)
    out = []
    for g in range(Y.shape[1]):
        with np.errstate(invalid="ignore"):
            other = np.nanmean(np.delete(Y, g, axis=1), axis=1)
        ok = np.isfinite(Y[:, g]) & np.isfinite(other)
        out.append(np.corrcoef(Y[ok, g], other[ok])[0, 1] ** 2 if ok.sum() > 2 else np.nan)
    return np.array(out)


def pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def group_r(M, S, cols):
    return np.array([pearson(M[:, g], S[:, g]) for g in cols])


def spearman_perm(x, y, n_perm=N_PERM, seed=SEED):
    rho = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    perm = np.array([stats.spearmanr(x, rng.permutation(y)).statistic
                     for _ in range(n_perm)])
    p_one = (np.sum(perm >= rho) + 1) / (n_perm + 1)
    p_two = (np.sum(np.abs(perm) >= abs(rho)) + 1) / (n_perm + 1)
    return float(rho), float(p_one), float(p_two)


def distance_pattern(Ys, stations, label):
    """near - far of pooled 1 - gamma on per-day standardised station anomalies."""
    z, _ = m11.anomalies(pd.DataFrame(Ys))
    sq, a, b = m11.pair_sq_diffs(z.to_numpy())
    corr, n_shared = m11.pooled_corr(sq)
    use = n_shared >= m11.MIN_SHARED
    lat, lon = stations["lat"].to_numpy(), stations["lon"].to_numpy()
    dist = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    g = stations["cv_group"].to_numpy()
    same_mat = g[:, None] == g[None, :]
    cu, au, bu = corr[use], a[use], b[use]
    contrast, n_near, n_far = m11.near_far(cu, dist[au, bu], same_mat[au, bu])
    p = m11.mantel_p(cu, au, bu, dist, same_mat, rng=np.random.default_rng(SEED))
    tab = m11.distance_bins(cu, dist[au, bu], same_mat[au, bu], n_shared[use])
    tab.insert(0, "data", label)
    return contrast, n_near, n_far, p, tab


def ci(x):
    return m11.ci(x)


# --------------------------------------------------------------------------
# Decision rules, exactly as pre-registered
# --------------------------------------------------------------------------

def score_h1(b2_lo, b2_hi, A):
    if b2_lo > H1_B2 and A >= H1_A:
        return "supported"
    if b2_hi < H1_B2 or A < H1_A:
        return "counts against"
    return "inconclusive"


def score_h2(c_mon, c_mon_lo, c_sat, c_sat_lo):
    if c_mon_lo > H2_C and c_sat_lo > H2_C:
        return "supported"
    if c_mon <= H2_C or c_sat <= H2_C:
        return "counts against"
    return "inconclusive"


def score_h3(med, lo):
    if med >= H3_MED and lo >= H3_LO:
        return "supported"
    if med < H3_FAIL:
        return "counts against"
    return "weak"


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    P = build_panels()
    st, Mg, Sg, Ms, Ss = P["stations"], P["Mg"], P["Sg"], P["Ms"], P["Ss"]
    dates = Mg.index
    gcols = list(Mg.columns)
    M, S = Mg.to_numpy(), Sg.to_numpy()
    n_days, n_grp = M.shape
    gdays = np.isfinite(M).sum(axis=0)

    confirm = (int((gdays >= MIN_GROUP_DAYS).sum()) >= MIN_CONF_GROUPS
               and n_days >= MIN_KEPT_DAYS)
    status = "confirmatory" if confirm else "EXPLORATORY (minimum not met)"
    print(f"matched panel: {n_days} kept days, {n_grp} groups, "
          f"{len(st)} monitors, {int(np.isfinite(M).sum()):,} group-days -> {status}")

    rows = []

    def put(metric, data, value, lo=np.nan, hi=np.nan, **extra):
        rows.append({"metric": metric, "data": data, "value": value,
                     "lo": lo, "hi": hi, **extra})

    # Lag check on the citywide series.
    cm, cs = np.nanmean(M, axis=1), np.nanmean(S, axis=1)
    lag = {}
    for k in (-1, 0, 1):
        if k < 0:
            lag[k] = pearson(cm[-k:], cs[:k])
        elif k > 0:
            lag[k] = pearson(cm[:-k], cs[k:])
        else:
            lag[k] = pearson(cm, cs)
        put(f"lag_r_citywide_{k:+d}", "monitor vs satellite", lag[k], n_days=n_days)
    lag0_best = lag[0] >= max(lag[-1], lag[1])

    # Anomalies on the matched panel.
    Am, As = anomalies(M), anomalies(S)
    pi, pj = day_pairs(dates)
    pi_s, pj_s, r_s = b2_pairs(As, pi, pj)
    pi_m, pj_m, r_m = b2_pairs(Am, pi, pj)
    b2_s, b2_m = float(r_s.mean()), float(r_m.mean())
    A_s, A_m = amplitude(As), amplitude(Am)

    # H2 inputs.
    C_m, C_s = citywide_share(M), citywide_share(S)
    with np.errstate(invalid="ignore", divide="ignore"):
        LM, LS = np.log(M), np.log(S)
    C_lm, C_ls = citywide_share(LM), citywide_share(LS)
    DM, ddates = first_differences(M, dates)
    DS, _ = first_differences(S, dates)
    C_dm, C_ds = citywide_share(DM), citywide_share(DS)
    loo_m, loo_s = loo_r2(M), loo_r2(S)

    # H3 inputs.
    h3_cols = [k for k in range(n_grp) if gdays[k] >= MIN_GROUP_DAYS]
    r_grp = group_r(M, S, h3_cols)
    med_r = float(np.median(r_grp))
    pooled = pearson(M.ravel(), S.ravel())
    anom_r = pearson(Am.ravel(), As.ravel())
    mean_am, mean_as = np.nanmean(Am, axis=0), np.nanmean(As, axis=0)
    rho, p_rho, p_rho2 = spearman_perm(mean_as, mean_am)

    # Bootstrap: one set of month draws for everything.
    months = pd.DatetimeIndex(dates).strftime("%Y-%m").to_numpy()
    dmonths = pd.DatetimeIndex(ddates).strftime("%Y-%m").to_numpy()
    res = m11.month_resamples(np.unique(months))
    keys = ["b2_s", "b2_m", "A_s", "A_m", "C_m", "C_s", "C_diff", "C_lm", "C_ls",
            "C_dm", "C_ds", "C_ddiff", "med_r", "pooled", "anom_r", "b2_diff"]
    B = {k: np.full(len(res), np.nan) for k in keys}
    for bi, draw in enumerate(res):
        cnt = pd.Series(draw).value_counts()
        w = cnt.reindex(months).fillna(0).to_numpy().astype(int)
        wd = cnt.reindex(dmonths).fillna(0).to_numpy().astype(int)
        rep, drep = np.repeat(np.arange(n_days), w), np.repeat(np.arange(len(ddates)), wd)
        if len(rep) == 0:
            continue
        ps, pm_ = w[pi_s] * w[pj_s], w[pi_m] * w[pj_m]
        if ps.sum() > 0:
            B["b2_s"][bi] = np.sum(ps * r_s) / ps.sum()
        if pm_.sum() > 0:
            B["b2_m"][bi] = np.sum(pm_ * r_m) / pm_.sum()
        B["b2_diff"][bi] = B["b2_s"][bi] - B["b2_m"][bi]
        B["A_s"][bi], B["A_m"][bi] = amplitude(As[rep]), amplitude(Am[rep])
        B["C_m"][bi], B["C_s"][bi] = citywide_share(M[rep]), citywide_share(S[rep])
        B["C_diff"][bi] = B["C_s"][bi] - B["C_m"][bi]
        B["C_lm"][bi], B["C_ls"][bi] = citywide_share(LM[rep]), citywide_share(LS[rep])
        B["C_dm"][bi], B["C_ds"][bi] = citywide_share(DM[drep]), citywide_share(DS[drep])
        B["C_ddiff"][bi] = B["C_ds"][bi] - B["C_dm"][bi]
        B["med_r"][bi] = np.nanmedian(group_r(M[rep], S[rep], h3_cols))
        B["pooled"][bi] = pearson(M[rep].ravel(), S[rep].ravel())
        B["anom_r"][bi] = pearson(Am[rep].ravel(), As[rep].ravel())
    I = {k: ci(v) for k, v in B.items()}

    n_pairs = dict(n_pairs_sat=len(r_s), n_pairs_mon=len(r_m))
    put("H1_b2", "satellite", b2_s, *I["b2_s"], **n_pairs)
    put("H1_amplitude_ugm3", "satellite", A_s, *I["A_s"])
    put("b2", "monitors", b2_m, *I["b2_m"], **n_pairs)
    put("amplitude_ugm3", "monitors", A_m, *I["A_m"])
    put("b2_sat_minus_mon", "paired", b2_s - b2_m, *I["b2_diff"])

    put("H2_C_levels", "monitors", C_m, *I["C_m"])
    put("H2_C_levels", "satellite", C_s, *I["C_s"])
    put("C_levels_sat_minus_mon", "paired", C_s - C_m, *I["C_diff"])
    put("C_log", "monitors", C_lm, *I["C_lm"])
    put("C_log", "satellite", C_ls, *I["C_ls"])
    put("H2b_C_first_diff", "monitors", C_dm, *I["C_dm"],
        n_rows=int(np.isfinite(DM).sum()))
    put("H2b_C_first_diff", "satellite", C_ds, *I["C_ds"],
        n_rows=int(np.isfinite(DS).sum()))
    put("C_first_diff_sat_minus_mon", "paired", C_ds - C_dm, *I["C_ddiff"])
    for lab, v in (("monitors", loo_m), ("satellite", loo_s)):
        put("loo_r2_median", lab, float(np.nanmedian(v)),
            float(np.nanpercentile(v, 25)), float(np.nanpercentile(v, 75)),
            note="lo/hi = IQR across groups, not an interval")

    put("H3_median_group_r", "monitor vs satellite", med_r, *I["med_r"],
        n_groups=len(h3_cols))
    put("pooled_r_levels", "monitor vs satellite", pooled, *I["pooled"])
    put("H3b_pooled_anomaly_r", "monitor vs satellite", anom_r, *I["anom_r"])
    put("H3b_mean_anomaly_spearman", "monitor vs satellite", rho,
        p_one_sided=p_rho, p_two_sided=p_rho2, n_groups=n_grp)
    # EXPLORATORY, not pre-registered: the pooled anomaly r with each group's
    # mean anomaly removed, i.e. agreement on day-to-day departures only.
    put("EXPLORATORY_daily_anomaly_r_group_means_removed", "monitor vs satellite",
        pearson((Am - mean_am).ravel(), (As - mean_as).ravel()))

    # Distance pattern, station level. Satellite as specified; monitors exploratory.
    dc_s, nn, nf, p_s, tab_s = distance_pattern(Ss.to_numpy(), st, "satellite")
    dc_m, _, _, p_m, tab_m = distance_pattern(Ms.to_numpy(), st, "monitors (exploratory)")
    put("near_minus_far", "satellite", dc_s, near_pairs=nn, far_pairs=nf, mantel_p=p_s)
    put("near_minus_far", "monitors (exploratory)", dc_m, near_pairs=nn,
        far_pairs=nf, mantel_p=p_m)

    # All-land-cell sensitivity, all satellite days.
    Ac = anomalies(P["sat_cells"])
    ci_, cj_ = day_pairs(P["sat_dates"])
    _, _, r_c = b2_pairs(Ac, ci_, cj_)
    put("b2_all_land_cells", "satellite", float(r_c.mean()), n_pairs=len(r_c),
        n_cells=Ac.shape[1], n_days=Ac.shape[0])
    put("amplitude_all_land_cells_ugm3", "satellite", amplitude(Ac))

    # Scores.
    h1 = score_h1(I["b2_s"][0], I["b2_s"][1], A_s)
    h2 = score_h2(C_m, I["C_m"][0], C_s, I["C_s"][0])
    h2b = score_h2(C_dm, I["C_dm"][0], C_ds, I["C_ds"][0])
    h3 = score_h3(med_r, I["med_r"][0])
    if not confirm:
        h1 = h2 = h2b = h3 = "not scored (exploratory)"
    for name, v in (("H1", h1), ("H2", h2), ("H2b_first_diff", h2b), ("H3", h3)):
        put(f"SCORE_{name}", v, np.nan)
    put("lag0_is_best", str(lag0_best), np.nan)

    out = pd.DataFrame(rows)
    out.insert(0, "status", status)
    out.to_csv(OUT_CSV, index=False)

    gt = pd.DataFrame({
        "cv_group": [gcols[k] for k in h3_cols],
        "monitors": [", ".join(f"{st.loc[s, 'site']} - {st.loc[s, 'agency']}"
                               for s in st.index[st["cv_group"] == gcols[k]])
                     for k in h3_cols],
        "days": gdays[h3_cols],
        "r": r_grp,
        "rmse": [float(np.sqrt(np.nanmean((S[:, k] - M[:, k]) ** 2))) for k in h3_cols],
        "bias_sat_minus_mon": [float(np.nanmean(S[:, k] - M[:, k])) for k in h3_cols],
        "match_m": [", ".join(f"{st.loc[s, 'match_m']:.0f}"
                              for s in st.index[st["cv_group"] == gcols[k]])
                    for k in h3_cols],
        "mean_anomaly_mon": mean_am[h3_cols],
        "mean_anomaly_sat": mean_as[h3_cols],
    })
    gt.to_csv(OUT_GROUPS, index=False)
    pd.concat([tab_s, tab_m]).to_csv(OUT_DIST, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_colwidth", 60)
    print(out.drop(columns=["status"]).to_string(index=False))
    print()
    print(gt.round(3).to_string(index=False))
    print()
    print(pd.concat([tab_s, tab_m]).round(3).to_string(index=False))
    print(f"\nH1 {h1}   H2 {h2}   (H2b first differences: {h2b})   H3 {h3}")
    if not lag0_best:
        print("NOTE: lag 0 is not the best-aligned lag; stated with every result.")
    print(f"wrote {OUT_CSV.relative_to(ROOT)}, {OUT_GROUPS.relative_to(ROOT)}, "
          f"{OUT_DIST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
