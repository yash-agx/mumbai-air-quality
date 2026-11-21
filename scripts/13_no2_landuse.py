"""Does land use explain NO2's between-monitor differences better than PM2.5's?

Pre-registered in reports/preregistration_no2_landuse.md before any NO2 vs
land-use relationship was computed.

Land-use regression is known to work far better for NO2 than for PM2.5: NO2
falls off within a few hundred metres of a road, and a model that reads each
monitor's own road environment can see that even when monitors are 1-40 km
apart. So on the matched NO2/PM2.5 panel from the NO2 positive control, each
monitor's long-run mean (and its mean anomaly against the hourly city
average) is regressed on two road features fixed in advance:

    log10(dist_major_road_m)   expected sign -
    road_density_500m          expected sign +

Validation is leave-one-cv_group-out against the city mean (R2cv), with a
permutation p. The primary contrast is R2cv(NO2) - R2cv(PM2.5) on the anomaly,
with a paired bootstrap over groups.

The machinery is shared with the PM2.5 land-use tests in
scripts/03_diurnal_landuse.py so the two pollutants are tested the same way:
the cv_group units (to_groups convention), perm_test, benjamini_hochberg, the
Mantel control, odd/even-day reliability, N_PERM and SEED. Cleaning and the
matched panel come unchanged from scripts/12_no2_positive_control.py.

Run: python scripts/13_no2_landuse.py
Writes reports/no2_landuse.csv, reports/no2_landuse_groups.csv,
       reports/no2_landuse_spearman.csv, reports/figures/no2_landuse.png
"""

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "reports" / "no2_landuse.csv"
OUT_GROUPS = ROOT / "reports" / "no2_landuse_groups.csv"
OUT_SPEARMAN = ROOT / "reports" / "no2_landuse_spearman.csv"
OUT_FIG = ROOT / "reports" / "figures" / "no2_landuse.png"

PREDICTORS = ["log_dist_major_road", "road_density_500m"]
EXPECTED_SIGN = {"log_dist_major_road": -1, "road_density_500m": +1}
ALL_FEATURES = ["dist_major_road_m", "road_density_500m", "road_density_1km",
                "dist_coast_m", "dist_industrial_m", "building_density_500m"]

MIN_HOURS = 2000            # matched hours a monitor needs to enter
MIN_CITY_MONITORS = 10      # monitors an hour needs for its city average
N_BOOT = 2000
N_MANTEL = 2000
ALPHA = 0.05

FLAT = [6956, 6945, 12040, 6943]
ODD_RATIO = [6956, 6965, 6967, 6943, 6948, 6945, 12040, 7752, 3409328, 3409329]
POLLUTANTS = ("no2", "pm25")
OUTCOMES = ("anomaly", "level")


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------

def station_outcomes(panel, min_hours=MIN_HOURS, min_city=MIN_CITY_MONITORS,
                     row_mask=None):
    """Per-station level and mean anomaly from a wide (hour x station) panel.

    The city average c_t is the mean over monitors present at hour t, and an
    hour enters the anomaly only if >= min_city monitors are present. A
    monitor with fewer than min_hours readings is dropped (NaN). row_mask
    restricts which hours are averaged (odd/even days); the city average is
    always computed on the hour itself, so it is unchanged by the mask.
    """
    V = panel.to_numpy(float)
    n_present = np.isfinite(V).sum(axis=1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        city = np.where(n_present >= min_city, np.nanmean(V, axis=1), np.nan)
    rows = np.ones(len(V), bool) if row_mask is None else np.asarray(row_mask)
    Vr = V[rows]
    hours = np.isfinite(Vr).sum(axis=0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # Level: every matched hour. Anomaly: only hours with a city average.
        level = np.nanmean(Vr, axis=0)
        anomaly = np.nanmean(Vr - city[rows, None], axis=0)
    out = pd.DataFrame({"hours": hours, "level": level, "anomaly": anomaly},
                       index=panel.columns)
    out.loc[out["hours"] < min_hours, ["level", "anomaly"]] = np.nan
    return out


def group_table(values, feat, columns):
    """Station values -> cv_group means (03's to_groups convention), with the
    group's mean features and mean coordinates."""
    df = values[columns].join(feat[["cv_group", "lat", "lon"] + ALL_FEATURES])
    df = df.dropna(subset=columns)
    g = df.groupby("cv_group")
    out = g[columns + ["lat", "lon"] + ALL_FEATURES].mean()
    out["n_stations"] = g.size()
    out["stations"] = g.apply(lambda s: ",".join(map(str, s.index)))
    out["log_dist_major_road"] = np.log10(out["dist_major_road_m"])
    return out


# --------------------------------------------------------------------------
# Leave-one-group-out skill
# --------------------------------------------------------------------------

def design(units, predictors=PREDICTORS):
    return np.column_stack([np.ones(len(units))] +
                           [units[p].to_numpy(float) for p in predictors])


def loo_operators(X, w=None):
    """Matrices M, M0 with M @ y = leave-one-out residuals of the weighted OLS
    fit and M0 @ y = those of the weighted mean (the city-mean baseline).

    With integer weights (bootstrap copies), deleting row i deletes all w_i
    copies of that group at once, so no group ever predicts itself:
    e_loo_i = e_i / (1 - w_i h_ii), h_ii = x_i' (X'WX)^-1 x_i.
    """
    n = X.shape[0]
    w = np.ones(n) if w is None else np.asarray(w, float)
    A = np.linalg.inv(X.T @ (w[:, None] * X))
    H = X @ A @ (X.T * w)                       # fitted = H y
    lev = w * np.einsum("ij,jk,ik->i", X, A, X)
    M = (np.eye(n) - H) / (1.0 - lev)[:, None]
    W = w.sum()
    M0 = (np.eye(n) - np.outer(np.ones(n), w) / W) / (1.0 - w / W)[:, None]
    return M, M0, w


def r2cv_from(M, M0, w, Y):
    """R2cv = 1 - SSE_model / SSE_baseline, for one y or a column stack."""
    E, E0 = M @ Y, M0 @ Y
    sse = (w[:, None] * E ** 2).sum(0) if Y.ndim == 2 else (w * E ** 2).sum()
    sse0 = (w[:, None] * E0 ** 2).sum(0) if Y.ndim == 2 else (w * E0 ** 2).sum()
    return 1.0 - sse / sse0


def loo_skill(units, y, predictors=PREDICTORS):
    """Point skill on the real groups: R2cv, both RMSEs, LOO predictions."""
    X = design(units, predictors)
    y = np.asarray(y, float)
    M, M0, w = loo_operators(X)
    e, e0 = M @ y, M0 @ y
    rmse, rmse0 = np.sqrt(np.mean(e ** 2)), np.sqrt(np.mean(e0 ** 2))
    return {"r2cv": 1 - (e ** 2).sum() / (e0 ** 2).sum(),
            "rmse_landuse": rmse, "rmse_city": rmse0,
            "gap_pct": 100 * (rmse - rmse0) / rmse0,
            "pred_landuse": y - e, "pred_city": y - e0}


def perm_p(units, y, n_perm, seed, predictors=PREDICTORS):
    """One-sided permutation p for R2cv, outcome shuffled across groups."""
    X = design(units, predictors)
    y = np.asarray(y, float)
    M, M0, w = loo_operators(X)
    obs = r2cv_from(M, M0, w, y)
    rng = np.random.default_rng(seed)
    Y = np.column_stack([rng.permutation(y) for _ in range(n_perm)])
    null = r2cv_from(M, M0, w, Y)
    return float(obs), float(((null >= obs).sum() + 1) / (n_perm + 1))


def bootstrap_r2(units, ys, n_boot, seed, predictors=PREDICTORS):
    """Paired group bootstrap: the same draws for every outcome in `ys`.

    Returns {name: array of R2cv over resamples}. Each draw is expressed as
    integer weights on the distinct groups, and the weighted LOO operator
    holds every copy of a group out together.
    """
    X = design(units, predictors)
    n = len(units)
    rng = np.random.default_rng(seed)
    out = {k: np.full(n_boot, np.nan) for k in ys}
    Ystack = np.column_stack([np.asarray(v, float) for v in ys.values()])
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        k = w > 0
        try:
            M, M0, ww = loo_operators(X[k], w[k])
        except np.linalg.LinAlgError:
            continue
        r = r2cv_from(M, M0, ww, Ystack[k])
        for j, name in enumerate(ys):
            out[name][b] = r[j]
    return out


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def ols_full(units, y, predictors=PREDICTORS):
    """Full-sample OLS: coefficients and classical t values."""
    X = design(units, predictors)
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = (resid ** 2).sum() / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return {p: (float(beta[i + 1]), float(beta[i + 1] / se[i + 1]))
            for i, p in enumerate(predictors)}


def right_sign(coefs):
    """At least one predictor in its expected direction, and no predictor with
    a wrong-signed coefficient at |t| > 2."""
    ok = any(np.sign(b) == EXPECTED_SIGN[p] for p, (b, _) in coefs.items())
    bad = any(np.sign(b) == -EXPECTED_SIGN[p] and abs(t) > 2
              for p, (b, t) in coefs.items())
    return bool(ok and not bad)


def reading(r2n, pn, rsn, r2p, pp, rsp, d_lo, d_hi, alpha=ALPHA):
    """The pre-registered readings, in this order: L1, L2, L3, L5, L4."""
    sig_n = pn < alpha and r2n > 0
    sig_p = pp < alpha and r2p > 0
    if sig_n and rsn and d_lo > 0:
        return "L1: land use explains NO2 more"
    if sig_n and rsn and d_lo <= 0 <= d_hi and not sig_p:
        return "L2: NO2 signal, contrast unresolved"
    if pn >= alpha and pp >= alpha:
        # The pre-registration does not rank L3 against L4's "D upper < 0"
        # arm, so an overlap is flagged rather than resolved after the fact.
        if d_hi < 0:
            return "L3/L4 overlap: described cell by cell"
        return "L3: no detectable land-use signal"
    if (pn < alpha and not rsn) or (pp < alpha and not rsp):
        return "L5: signal, but not the road signal"
    if d_hi < 0 or (pp < alpha and pn >= alpha):
        return "L4: PM2.5 more than NO2"
    return "unlabelled: described cell by cell"


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def build_panels(nc, at):
    mod = at.load_engine()
    wide, feat, _ = mod.load()
    raw = pd.read_parquet(nc.RAW_PATH)
    df, _ = nc.clean_no2(raw, "no2", wide.index.min(), wide.index.max())
    no2 = nc.to_wide(df, wide.index, wide.columns)
    pm_m, no2_m = nc.matched(wide, no2)
    return {"pm25": pm_m, "no2": no2_m}, feat, mod


def analyse(name, panels, feat, mod, dl, row_mask=None, drop=(), spearman=False):
    """Everything pre-registered for one panel variant."""
    keep = [c for c in panels["no2"].columns if c not in set(drop)]
    rows_sel = slice(None) if row_mask is None else np.asarray(row_mask)
    P = {p: panels[p].loc[rows_sel, keep] for p in POLLUTANTS}
    ist = P["no2"].index.tz_convert(dl.TZ)
    days = ist.dayofyear.to_numpy()

    vals, halves = {}, {}
    for p in POLLUTANTS:
        s = station_outcomes(P[p])
        so = station_outcomes(P[p], min_hours=MIN_HOURS / 2, row_mask=days % 2 == 1)
        se = station_outcomes(P[p], min_hours=MIN_HOURS / 2, row_mask=days % 2 == 0)
        for o in OUTCOMES:
            vals[f"{p}_{o}"] = s[o]
            halves[f"{p}_{o}"] = (so[o], se[o])
        vals[f"{p}_hours"] = s["hours"]
    cols = [f"{p}_{o}" for p in POLLUTANTS for o in OUTCOMES]
    units = group_table(pd.DataFrame(vals), feat, cols)
    n_st = int(units["n_stations"].sum())
    print(f"\n{name}: {n_st} monitors, {len(units)} groups, "
          f"{int(P['no2'].notna().sum().sum()):,} matched station-hours")

    # Odd/even split-half ceiling, at group level.
    rel = {}
    for k, (a, b) in halves.items():
        ga = pd.DataFrame({"v": a}).join(feat["cv_group"]).groupby("cv_group")["v"].mean()
        gb = pd.DataFrame({"v": b}).join(feat["cv_group"]).groupby("cv_group")["v"].mean()
        j = pd.concat([ga, gb], axis=1).loc[units.index].dropna()
        rel[k] = float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1])

    gdist = mod.distance_matrix(units["lat"].to_numpy(), units["lon"].to_numpy())
    keep_pairs = ~np.eye(len(units), dtype=bool)

    boots = bootstrap_r2(units, {k: units[k] for k in cols}, N_BOOT, dl.SEED)
    rows = []
    for o in OUTCOMES:
        res = {}
        for p in POLLUTANTS:
            k = f"{p}_{o}"
            y = units[k].to_numpy(float)
            sk = loo_skill(units, y)
            _, pval = perm_p(units, y, dl.N_PERM, dl.SEED)
            coefs = ols_full(units, y)
            dy = np.abs(y[:, None] - y[None, :])
            mr, mp, _ = dl.mantel(dy, gdist, keep_pairs, n_perm=N_MANTEL, seed=dl.SEED)
            lo, hi = ci(boots[k])
            res[p] = dict(sk, perm_p=pval, coefs=coefs, rs=right_sign(coefs),
                          lo=lo, hi=hi, mantel_r=mr, mantel_p=mp)
            units[f"{k}_pred_landuse"] = sk["pred_landuse"]
            units[f"{k}_pred_city"] = sk["pred_city"]
        padj = dl.benjamini_hochberg([res["no2"]["perm_p"], res["pm25"]["perm_p"]])
        for p, pa in zip(POLLUTANTS, padj):
            res[p]["p_bh"] = float(pa)
        d = boots[f"no2_{o}"] - boots[f"pm25_{o}"]
        d_point = res["no2"]["r2cv"] - res["pm25"]["r2cv"]
        d_lo, d_hi = ci(d)
        d_p = float(np.mean(d[np.isfinite(d)] <= 0))
        verdict = reading(res["no2"]["r2cv"], res["no2"]["p_bh"], res["no2"]["rs"],
                          res["pm25"]["r2cv"], res["pm25"]["p_bh"], res["pm25"]["rs"],
                          d_lo, d_hi)
        for p in POLLUTANTS:
            r = res[p]
            rows.append({
                "panel": name, "outcome": o, "pollutant": p,
                "n_stations": n_st, "n_groups": len(units),
                "r2cv": r["r2cv"], "r2cv_lo": r["lo"], "r2cv_hi": r["hi"],
                "rmse_city": r["rmse_city"], "rmse_landuse": r["rmse_landuse"],
                "gap_pct": r["gap_pct"], "perm_p": r["perm_p"], "p_bh": r["p_bh"],
                "b_log_dist": r["coefs"]["log_dist_major_road"][0],
                "t_log_dist": r["coefs"]["log_dist_major_road"][1],
                "b_density": r["coefs"]["road_density_500m"][0],
                "t_density": r["coefs"]["road_density_500m"][1],
                "right_sign": r["rs"], "split_half_r": rel[f"{p}_{o}"],
                "mantel_r": r["mantel_r"], "mantel_p": r["mantel_p"],
                "D": d_point, "D_lo": d_lo, "D_hi": d_hi, "D_p_one_sided": d_p,
                "reading": verdict,
            })
        print(f"  {o:7s} R2cv NO2 {res['no2']['r2cv']:+.3f} "
              f"[{res['no2']['lo']:+.3f}, {res['no2']['hi']:+.3f}] p_bh {res['no2']['p_bh']:.3f}"
              f" | PM2.5 {res['pm25']['r2cv']:+.3f} [{res['pm25']['lo']:+.3f}, "
              f"{res['pm25']['hi']:+.3f}] p_bh {res['pm25']['p_bh']:.3f}"
              f" | D {d_point:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}] -> {verdict}")

    sp = None
    if spearman:
        sp_rows = []
        for o in OUTCOMES:
            for p in POLLUTANTS:
                for f in PREDICTORS:
                    stat, pv, n = dl.perm_test(units[f], units[f"{p}_{o}"], False)
                    sp_rows.append({"family": "per-feature (pre-registered)",
                                    "outcome": o, "pollutant": p, "feature": f,
                                    "rho": stat, "p": pv, "n": n})
        sp = pd.DataFrame(sp_rows)
        sp["p_bh"] = dl.benjamini_hochberg(sp["p"].to_numpy())
    return pd.DataFrame(rows), units, sp


def exploratory(units, dl):
    """EXPLORATORY, not pre-registered: the four other features one at a time,
    and an all-six-feature model."""
    u = units.copy()
    for c in ("dist_major_road_m", "dist_coast_m", "dist_industrial_m"):
        u[f"log_{c}"] = np.log10(u[c] + 10.0)
    six = ["log_dist_major_road_m", "road_density_500m", "road_density_1km",
           "log_dist_coast_m", "log_dist_industrial_m", "building_density_500m"]
    rows = []
    for p in POLLUTANTS:
        y = u[f"{p}_anomaly"].to_numpy(float)
        r2, pv = perm_p(u, y, 5000, dl.SEED, predictors=six)
        rows.append({"family": "exploratory: six-feature model",
                     "outcome": "anomaly", "pollutant": p, "feature": "all six",
                     "rho": np.nan, "r2cv": r2, "p": pv, "n": len(u)})
        for f in ("road_density_1km", "dist_coast_m", "dist_industrial_m",
                  "building_density_500m"):
            stat, pf, n = dl.perm_test(u[f], u[f"{p}_anomaly"], False, n_perm=5000)
            rows.append({"family": "exploratory: other features",
                         "outcome": "anomaly", "pollutant": p, "feature": f,
                         "rho": stat, "r2cv": np.nan, "p": pf, "n": n})
    return pd.DataFrame(rows)


def figure(t, units, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid_c, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colors = {"pm25": "#2a78d6", "no2": "#eb6834"}
    names = {"pm25": "PM2.5", "no2": "NO2"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    fig.patch.set_facecolor(surface)

    ax = axes[0]
    sub = t[t["outcome"] == "anomaly"]
    panels = list(dict.fromkeys(sub["panel"]))
    x = np.arange(len(panels))
    for p, off in (("pm25", -0.1), ("no2", 0.1)):
        r = sub[sub["pollutant"] == p].set_index("panel").loc[panels]
        y = r["r2cv"].to_numpy(float)
        ax.errorbar(x + off, y, yerr=[y - r["r2cv_lo"], r["r2cv_hi"] - y],
                    fmt="o", ms=8, color=colors[p], elinewidth=2, capsize=0,
                    mec=surface, mew=1.5, label=names[p], zorder=3)
    ax.axhline(0, color=muted, lw=1)
    short = {"primary": "primary", "S1 no flat sensors": "S1\nno flat",
             "S2 no odd-ratio monitors": "S2\nno odd ratio",
             "S3 03 window": "S3\n03 window"}
    ax.set_xticks(x, [short.get(s, s) for s in panels], fontsize=8)
    lo = float(sub["r2cv_lo"].min())
    ax.set_ylim(lo - 0.15, max(0.3, float(sub["r2cv_hi"].max()) + 0.05))
    ax.set_title("(a) Land-use skill over the city mean", loc="left",
                 fontsize=11, color=ink)
    ax.set_ylabel("R²cv, leave-one-group-out (0 = city mean)", color=muted)
    ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=2)

    for ax, p, tag in ((axes[1], "no2", "(b)"), (axes[2], "pm25", "(c)")):
        obs = units[f"{p}_anomaly"]
        pred = units[f"{p}_anomaly_pred_landuse"]
        lim = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
        pad = 0.08 * (lim[1] - lim[0])
        lim = [lim[0] - pad, lim[1] + pad]
        ax.plot(lim, lim, color=muted, lw=1, zorder=1)
        ax.axhline(0, color=grid_c, lw=1)
        ax.scatter(pred, obs, s=46, color=colors[p], edgecolor=surface,
                   lw=1.2, zorder=3)
        r = t[(t["panel"] == "primary") & (t["outcome"] == "anomaly")
              & (t["pollutant"] == p)].iloc[0]
        ax.text(0.03, 0.97, f"R²cv {r['r2cv']:+.2f}, perm p {r['perm_p']:.2f}",
                transform=ax.transAxes, va="top", fontsize=9, color=ink)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        unit = "labelled ppb" if p == "no2" else "µg/m³"
        ax.set_title(f"{tag} {names[p]}: road model vs observed", loc="left",
                     fontsize=11, color=ink)
        ax.set_xlabel(f"held-out prediction ({unit})", color=muted)
        ax.set_ylabel(f"group mean anomaly vs city ({unit})", color=muted)

    for ax in axes:
        ax.grid(True, color=grid_c, lw=0.8)
        ax.set_facecolor(surface)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid_c)
        ax.tick_params(colors=muted, labelsize=9)
    fig.text(0.01, 0.01,
             "Predictors: log10 distance to major road, road density within "
             "500 m. Bars: 95% paired group bootstrap. Matched NO2/PM2.5 "
             "station-hours. (b), (c): primary panel, one point per cv_group.",
             fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    nc = _load("aq_no2_control", "scripts/12_no2_positive_control.py")
    at = _load("aq_averaging_times", "scripts/11_averaging_times.py")
    dl = _load("aq_diurnal_landuse", "scripts/03_diurnal_landuse.py")
    panels, feat, mod = build_panels(nc, at)

    ist = panels["no2"].index.tz_convert(dl.TZ)
    common = np.asarray(ist >= pd.Timestamp(dl.COMMON_START, tz=dl.TZ))
    variants = [
        ("primary", dict(spearman=True)),
        ("S1 no flat sensors", dict(drop=FLAT)),
        ("S2 no odd-ratio monitors", dict(drop=ODD_RATIO)),
        ("S3 03 window", dict(row_mask=common)),
    ]
    tabs, sp, primary_units = [], None, None
    for name, kw in variants:
        t, units, s = analyse(name, panels, feat, mod, dl, **kw)
        tabs.append(t)
        if name == "primary":
            sp, primary_units = s, units
    t = pd.concat(tabs, ignore_index=True)
    ex = exploratory(primary_units, dl)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT_CSV, index=False, float_format="%.5g")
    primary_units.to_csv(OUT_GROUPS, float_format="%.5g")
    pd.concat([sp, ex], ignore_index=True).to_csv(OUT_SPEARMAN, index=False,
                                                  float_format="%.5g")
    figure(t, primary_units, OUT_FIG)

    with pd.option_context("display.width", 260, "display.max_columns", 40,
                           "display.float_format", "{:.3f}".format):
        print()
        print(t.drop(columns=["reading"]).to_string(index=False))
        print("\nper-feature Spearman (pre-registered, BH across 8):")
        print(sp.to_string(index=False))
        print("\nexploratory:")
        print(ex.to_string(index=False))
        print("\nprimary groups (anomaly):")
        print(primary_units[["n_stations", "stations", "dist_major_road_m",
                             "road_density_500m", "no2_anomaly", "pm25_anomaly"]]
              .to_string())
    print(f"\nwrote {OUT_CSV.relative_to(ROOT)}, {OUT_GROUPS.relative_to(ROOT)}, "
          f"{OUT_SPEARMAN.relative_to(ROOT)}, {OUT_FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
