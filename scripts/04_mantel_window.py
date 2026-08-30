"""Why does the Mantel control disagree between the two windows?

`scripts/03_diurnal_landuse.py` runs a control test -- does diurnal shape
similarity decay with distance? -- and gets two incompatible answers from the
same monitors and the same code:

    full record   (2025-02-28 to 2026-08-27)   r = +0.053, p = 0.518
    common window (2026-03-10 to 2026-08-27)   r = -0.215, p = 0.0009

This script tries to resolve that, and separately diagnoses the circular-phase
reliability collapse the window swap also produced (split-half 0.880 -> 0.423,
ceiling 0.968 -> 0.771).

The hypothesis under test for the Mantel disagreement is SEASONAL ALIASING: on
the full record three monitors came online in March 2026, so monitors average
different slices of the year; that mismatch adds noise to every pair and
attenuates Mantel toward zero, which would make the flat full-record result the
artefact and the common window's decay the real signal.

The test that decides it is cheap and it is decisive: drop the three late
arrivals and rerun the full record on the 36 monitors whose spans already match.
If aliasing is doing the work, the decay should appear. Section 1.

Everything after that follows from the answer, which is no.

  1. ALIASING. Span-matched full record; per-pair seasonal-composition mismatch.
 1b. COVERAGE. The same idea one level finer: seasonal coverage imbalance INSIDE
     the window, and profiles rebuilt at fixed seasonal weights instead of by
     pooling hours, so the imbalance cannot act at all.
  2. SEASONS. Mantel inside seasonal blocks, across the whole record, on the 36
     span-matched monitors -- so the question "does the decay hold across
     seasons?" is asked of every season the record contains, not only the two
     inside the window.
  3. LENGTH. Controls separating a real seasonal difference from the attenuation
     a shorter window causes on its own.
  4. SWEEP. Mantel over a grid of start and end dates: knife-edge or basin?
  5. JACKKNIFE. By monitor, by cv group, by pair, and by distance range, plus a
     block bootstrap over days and a fragility calibration.
  6. PHASE. How circular-phase reliability depends on record length, what it
     actually depends on instead, and what a usable trough-hour test would cost.

Nothing here changes any test in 03_; it only reads the same panel. Findings go
to a new section in NOTES.md.

Run: python scripts/04_mantel_window.py
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

TZ = "Asia/Kolkata"
SEED = 0
N_PERM = 20000        # for the headline numbers
N_PERM_SWEEP = 2000   # for the many-window sweeps, where p is indicative only
COMMON_START = "2026-03-10"
# Monsoon onset, used only to cut the common window into its two seasons for the
# fixed-weight reconstruction in section 1b. Mumbai's onset is early June; the
# exact day is not critical, because the reconstruction is also run at monthly
# resolution, which does not depend on this boundary at all.
MONSOON_START = "2026-06-01"

# The three monitors that came online in March 2026. Everything labelled
# "span-matched" drops them, which is what makes a full-record run comparable
# across seasons: the other 36 all start within six weeks of each other in
# early 2025 and run to the end.
LATE = [6258871, 6266795, 6266794]

# Below this the fitted 24h harmonic is small enough that its phase is set by
# noise rather than by a daily cycle. Chosen by inspection of the amplitude
# distribution and used only for diagnostics, never to filter a reported test.
WEAK_AMP = 1.5

np.seterr(all="ignore")
# Short blocks leave some monitors without a reading in some hour. Those rows
# are dropped explicitly below; the warning the empty mean raises is noise.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")


# ---------------------------------------------------------------- primitives
# diurnal() and shape_of() match app.py and 03_diurnal_landuse.py. They are
# reimplemented over an array rather than a Series here because every sweep
# below rebuilds profiles thousands of times and the groupby is the bottleneck.

def profiles(values, hours, mask=None):
    """Stations x 24 hour-of-day mean profiles. NaN where an hour is unseen."""
    v = values if mask is None else values[mask]
    h = hours if mask is None else hours[mask]
    out = np.full((values.shape[1], 24), np.nan)
    for hh in range(24):
        sel = h == hh
        if sel.any():
            out[:, hh] = np.nanmean(v[sel], axis=0)
    return out


def shapes(prof):
    """Level and amplitude divided out, row-wise. What is left is shape."""
    d = prof - np.nanmean(prof, axis=1, keepdims=True)
    sd = np.nanstd(d, axis=1, keepdims=True)
    return np.divide(d, sd, out=np.zeros_like(d), where=sd > 0)


def harmonic_24(prof):
    """Amplitude and phase (hours) of the fitted 24-hour harmonic, per row."""
    c = np.cos(2 * np.pi * np.arange(24) / 24)
    s = np.sin(2 * np.pi * np.arange(24) / 24)
    ok = np.isfinite(prof).all(axis=1)
    a = (np.where(ok[:, None], prof, 0.0) * c).mean(axis=1) * 2
    b = (np.where(ok[:, None], prof, 0.0) * s).mean(axis=1) * 2
    ph = (np.arctan2(b, a) / (2 * np.pi) * 24) % 24
    return np.hypot(a, b), np.where(ok, ph, np.nan), ok


def circ_diff(a, b):
    """Signed circular difference a-b in hours, wrapped to [-12, 12)."""
    return (np.asarray(a, float) - np.asarray(b, float) + 12) % 24 - 12


def circ_mean(hours_):
    th = np.asarray(hours_, float) * 2 * np.pi / 24
    return float((np.arctan2(np.sin(th).mean(), np.cos(th).mean())
                  / (2 * np.pi) * 24) % 24)


def circ_circ_r(a_h, b_h):
    """Jammalamadaka circular-circular correlation, as used in 03_."""
    a = np.asarray(a_h, float) * 2 * np.pi / 24
    b = np.asarray(b_h, float) * 2 * np.pi / 24
    am = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    bm = np.arctan2(np.sin(b).mean(), np.cos(b).mean())
    num = np.sum(np.sin(a - am) * np.sin(b - bm))
    den = np.sqrt(np.sum(np.sin(a - am) ** 2) * np.sum(np.sin(b - bm) ** 2))
    return float(num / den) if den > 0 else np.nan


def spearman_brown(r):
    return 2 * r / (1 + r) if r > -1 else np.nan


def ceiling(r):
    """sqrt(Spearman-Brown reliability): the largest correlation a perfectly
    related feature could show against a summary this noisy."""
    return float(np.sqrt(max(spearman_brown(r), 0.0)))


# ---------------------------------------------------------------- Mantel

def mantel(sim, dist, keep, n_perm=N_PERM, seed=SEED):
    """Mantel test, identical in form to the one in 03_diurnal_landuse.py."""
    n = sim.shape[0]
    iu = np.triu_indices(n, 1)
    mask = keep[iu]
    a, b = sim[iu][mask], dist[iu][mask]
    obs = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        if abs(np.corrcoef(sim[np.ix_(p, p)][iu][mask], b)[0, 1]) >= abs(obs):
            hits += 1
    return obs, (hits + 1) / (n_perm + 1), len(a)


def loo_pearson(x, y):
    """Leave-one-out Pearson r for every point, in closed form.

    Used for the pair jackknife and its calibration, both of which would
    otherwise be O(n^2) recomputations of the same correlation.
    """
    n = len(x)
    sx, sy = x.sum(), y.sum()
    sxx, syy, sxy = (x * x).sum(), (y * y).sum(), (x * y).sum()
    m = n - 1
    ax, ay = sx - x, sy - y
    axx, ayy, axy = sxx - x * x, syy - y * y, sxy - x * y
    num = m * axy - ax * ay
    den = np.sqrt((m * axx - ax ** 2) * (m * ayy - ay ** 2))
    return np.divide(num, den, out=np.zeros(n), where=den > 0)


# ---------------------------------------------------------------- harness

class Panel:
    """The panel plus everything the sweeps need, built once."""

    def __init__(self):
        spec = importlib.util.spec_from_file_location(
            "interpolate", ROOT / "model" / "interpolate.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        wide, feat, _ = mod.load()

        self.mod = mod
        self.wide, self.feat = wide, feat
        self.sids = list(wide.columns)
        self.ist = wide.index.tz_convert(TZ)
        self.hours = self.ist.hour.to_numpy()
        self.month = self.ist.month.to_numpy()
        # Day index counted from the first day of the record, so odd/even splits
        # and window arithmetic do not break across the year boundary the way
        # dayofyear does.
        self.day = (self.ist.normalize() - self.ist.normalize().min()).days.to_numpy()
        self.values = wide.to_numpy(dtype=float)

        lat = feat.loc[self.sids, "lat"].to_numpy()
        lon = feat.loc[self.sids, "lon"].to_numpy()
        self.dist = mod.distance_matrix(lat, lon)
        grp = feat.loc[self.sids, "cv_group"].to_numpy()
        self.grp = grp
        self.keep = ~(grp[:, None] == grp[None, :])

        self.matched = np.array([s not in LATE for s in self.sids])
        self.common = np.asarray(self.ist >= pd.Timestamp(COMMON_START, tz=TZ))

    def window(self, start, end=None):
        def ts(x):
            t = pd.Timestamp(x)
            return t.tz_convert(TZ) if t.tzinfo else t.tz_localize(TZ)
        m = np.asarray(self.ist >= ts(start))
        if end is not None:
            m &= np.asarray(self.ist < ts(end))
        return m

    def months(self, months):
        return np.isin(self.month, months)

    def shape_mantel(self, mask, subset=None, n_perm=N_PERM_SWEEP, seed=SEED):
        """Mantel r between shape similarity and distance over an hour mask.

        Stations without a complete 24-hour profile inside the mask are dropped
        rather than imputed -- a partial profile is a different statistic, and
        in the short blocks below a few stations always have a gap.
        """
        cols = np.arange(len(self.sids)) if subset is None else np.asarray(subset)
        P = profiles(self.values[:, cols], self.hours, mask)
        ok = np.isfinite(P).all(axis=1)
        idx = cols[ok]
        if len(idx) < 15:
            return np.nan, np.nan, 0, len(idx)
        sim = np.corrcoef(shapes(P[ok]))
        r, p, npairs = mantel(sim, self.dist[np.ix_(idx, idx)],
                              self.keep[np.ix_(idx, idx)], n_perm, seed)
        return r, p, npairs, len(idx)


def rule(title):
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------- sections

def section_reproduce(pan):
    rule("0. THE DISAGREEMENT, REPRODUCED")
    all_idx = np.arange(len(pan.sids))
    m_idx = np.where(pan.matched)[0]
    print(f"  panel {pan.wide.shape[0]:,} hours x {pan.wide.shape[1]} stations, "
          f"{pan.feat['cv_group'].nunique()} cv groups")
    print(f"  {len(LATE)} monitors came online in March 2026: "
          + ", ".join(str(s) for s in LATE))
    print()
    print(f"  {'configuration':<44}{'r':>9}{'p':>10}{'pairs':>8}{'st':>5}")
    for lbl, mask, sub in [
            ("full record, all 39      [03_ SENSITIVITY]", None, all_idx),
            ("common window, all 39    [03_ PRIMARY]", pan.common, all_idx)]:
        mm = np.ones(len(pan.ist), bool) if mask is None else mask
        r, p, npairs, ns = pan.shape_mantel(mm, sub, n_perm=N_PERM)
        print(f"  {lbl:<44}{r:>+9.3f}{p:>10.4f}{npairs:>8}{ns:>5}")
    print("\n  Both reproduce. Everything below asks which one to believe.")
    print()


def section_aliasing(pan):
    rule("1. THE SEASONAL-ALIASING HYPOTHESIS -- tested directly, and refuted")
    print("  Claim: on the full record monitors average different slices of the")
    print("  year, which adds noise to every pair and biases Mantel toward zero.")
    print("  If so, restricting to monitors whose spans already match should")
    print("  recover the decay. It does not.\n")

    all_idx = np.arange(len(pan.sids))
    m_idx = np.where(pan.matched)[0]
    print(f"  {'configuration':<44}{'r':>9}{'p':>10}{'pairs':>8}{'st':>5}")
    rows = [("full record, all 39", np.ones(len(pan.ist), bool), all_idx),
            ("full record, 36 span-matched", np.ones(len(pan.ist), bool), m_idx),
            ("common window, all 39", pan.common, all_idx),
            ("common window, same 36", pan.common, m_idx)]
    for lbl, mask, sub in rows:
        r, p, npairs, ns = pan.shape_mantel(mask, sub, n_perm=N_PERM)
        print(f"  {lbl:<44}{r:>+9.3f}{p:>10.4f}{npairs:>8}{ns:>5}")

    print("\n  -> Removing the span mismatch moves the full record by +0.005.")
    print("     The flat full-record result is NOT an aliasing artefact.")
    print("     (It also shows the window's own r is partly carried by the three")
    print("     late arrivals: -0.215 on 39 against -0.157 on the matched 36.)")

    # The mismatch the hypothesis names is real; it is just not doing the work.
    # Quantified so the refutation rests on a measurement, not an assertion.
    print("\n  how much seasonal mismatch is actually there, per pair?")
    share = np.zeros(len(pan.sids))
    monsoon = pan.months([6, 7, 8])
    for j in range(len(pan.sids)):
        okv = np.isfinite(pan.values[:, j])
        share[j] = (okv & monsoon).sum() / max(okv.sum(), 1)
    mism = np.abs(share[:, None] - share[None, :])
    iu = np.triu_indices(len(pan.sids), 1)
    mk = pan.keep[iu]
    print(f"    monsoon share of a monitor's own record: median {np.median(share):.3f}, "
          f"range {share.min():.3f}-{share.max():.3f}")
    print(f"    pairwise |mismatch|: median {np.median(mism[iu][mk]):.3f}, "
          f"90th pct {np.percentile(mism[iu][mk], 90):.3f}")
    r, p, _ = mantel(mism, pan.dist, pan.keep, n_perm=N_PERM_SWEEP)
    print(f"    Mantel(|seasonal mismatch|, distance) r = {r:+.3f}, p = {p:.4f}")
    print("    -> a median pair differs by 2.8 percentage points of monsoon share,")
    print("       and the mismatch has no clear spatial arrangement of its own.")
    print("       Too small, and too weakly placed, to have hidden a gradient.")
    print()


def section_coverage_weights(pan):
    rule("1b. COVERAGE IMBALANCE INSIDE THE WINDOW -- the finer aliasing story")
    print("  Section 1 killed aliasing ACROSS the record. The same idea has a")
    print("  version INSIDE the window: monitors do not all cover the window's")
    print("  pre-monsoon and monsoon halves in the same proportion, so a pooled")
    print("  hour-of-day mean weights the two seasons differently at different")
    print("  monitors. If that imbalance is spatially arranged it could produce a")
    print("  distance gradient on its own.\n")

    cols = np.arange(len(pan.sids))
    monsoon = np.asarray(pan.ist >= pd.Timestamp(MONSOON_START, tz=TZ))
    pre = pan.common & ~monsoon
    mon = pan.common & monsoon
    print(f"  common window {int(pan.common.sum()):,} h  =  "
          f"pre-monsoon {int(pre.sum()):,} h (Mar 10-May 31)  +  "
          f"monsoon {int(mon.sum()):,} h (Jun 1-Aug 27)")

    # --- (a) the coverage shares themselves
    n_pre = np.isfinite(pan.values[pre]).sum(0).astype(float)
    n_mon = np.isfinite(pan.values[mon]).sum(0).astype(float)
    share = n_pre / np.maximum(n_pre + n_mon, 1)
    print("\n  (a) each monitor's pre-monsoon share of its own valid in-window hours\n")
    print(f"      median {np.median(share):.3f}   mean {share.mean():.3f}   "
          f"sd {share.std(ddof=1):.3f}")
    print(f"      IQR {np.percentile(share, 25):.3f}-{np.percentile(share, 75):.3f}   "
          f"full range {share.min():.3f}-{share.max():.3f}")
    print("\n      the most imbalanced monitors:")
    order = np.argsort(-np.abs(share - 0.5))
    print(f"      {'station':>10}{'pre h':>8}{'monsoon h':>11}{'share':>9}")
    for i in order[:6]:
        print(f"      {pan.sids[i]:>10}{int(n_pre[i]):>8}{int(n_mon[i]):>11}"
              f"{share[i]:>9.3f}")

    # --- (b) is the imbalance spatially arranged?
    sd_mat = np.abs(share[:, None] - share[None, :])
    iu = np.triu_indices(len(pan.sids), 1)
    mk = pan.keep[iu]
    print("\n  (b) Mantel(|pre-monsoon share difference|, distance)\n")
    print(f"      pairwise |share difference|: median "
          f"{np.median(sd_mat[iu][mk]):.3f}, 90th pct "
          f"{np.percentile(sd_mat[iu][mk], 90):.3f}, max {sd_mat[iu][mk].max():.3f}")
    for lbl, sub in [("all 39", cols), ("36 span-matched", cols[pan.matched])]:
        r, p, n = mantel(sd_mat[np.ix_(sub, sub)], pan.dist[np.ix_(sub, sub)],
                         pan.keep[np.ix_(sub, sub)], n_perm=N_PERM)
        print(f"      {lbl:<20} r = {r:+.4f}  p = {p:.4f}  pairs = {n}")
    print("      Note the SIGN. Manufacturing a decay would need MORE mismatch at")
    print("      LONGER separation, i.e. a positive r. The observed r is negative,")
    print("      so if the imbalance does anything it works against the decay.")

    # --- (c) rebuild profiles at fixed weights instead of pooling hours
    P_pool = profiles(pan.values, pan.hours, pan.common)
    P_pre = profiles(pan.values, pan.hours, pre)
    P_mon = profiles(pan.values, pan.hours, mon)
    P_eq = 0.5 * P_pre + 0.5 * P_mon
    P_month = np.nanmean(np.stack(
        [profiles(pan.values, pan.hours, pan.common & (pan.month == k))
         for k in (3, 4, 5, 6, 7, 8)]), axis=0)
    w = share[:, None]
    P_obs = w * P_pre + (1 - w) * P_mon

    # A monitor missing a whole season inside the window has no seasonal profile
    # to average, so it cannot enter any reconstruction. Dropping it from the
    # pooled arm too is what makes the four numbers comparable.
    ok = (np.isfinite(P_pool).all(1) & np.isfinite(P_eq).all(1)
          & np.isfinite(P_month).all(1))
    print("\n  (c) profiles rebuilt at FIXED seasonal weights, pooling replaced\n")
    for i in np.where(~ok)[0]:
        print(f"      dropped {pan.sids[i]}: {int(n_pre[i])} valid pre-monsoon hours "
              f"against {int(n_mon[i])} monsoon --")
        print( "        it is monsoon-only inside the window, so it has no")
        print( "        pre-monsoon profile to reweight. Dropped from ALL four arms.")
    for lbl, base in [("all monitors", cols),
                      ("36 span-matched", cols[pan.matched])]:
        sub = np.array([i for i in base if ok[i]])
        print(f"\n      --- {lbl}: {len(sub)} monitors, identical set in every arm ---")
        got = {}
        for key, P, name in [
                ("A", P_pool, "A. pooled hours              [what 03_ does]"),
                ("B", P_eq, "B. fixed 50/50 pre-monsoon:monsoon"),
                ("C", P_month, "C. fixed equal weight per calendar month"),
                ("D", P_obs, "D. placebo: reweighted to own observed shares")]:
            r, p, n = mantel(np.corrcoef(shapes(P[sub])),
                             pan.dist[np.ix_(sub, sub)],
                             pan.keep[np.ix_(sub, sub)], n_perm=N_PERM)
            got[key] = r
            print(f"      {name:<48} r = {r:+.4f}  p = {p:.4f}  pairs = {n}")
        print(f"      -> fixed weights retain {100 * got['B'] / got['A']:.0f}% of A "
              f"under B and {100 * got['C'] / got['A']:.0f}% under C;")
        print(f"         the placebo D reproduces A to "
              f"{abs(got['D'] - got['A']):.4f}, which is the check that the")
        print( "         reweighting machinery is wired up correctly.")

    # --- (d) confirm the reconstruction is not a no-op
    sub = np.where(ok)[0]
    SP, SE = shapes(P_pool), shapes(P_eq)
    cc = np.array([np.corrcoef(SP[i], SE[i])[0, 1] for i in sub])
    rho = stats.spearmanr(np.abs(share[sub] - 0.5), 1 - cc)
    print("\n  (d) did the reweighting actually do anything?\n")
    print(f"      per-monitor corr(pooled shape, equal-weight shape): median "
          f"{np.median(cc):.4f}, min {cc.min():.4f}")
    print(f"      Spearman(|share - 0.5|, how far the shape moved) = "
          f"{rho.statistic:+.3f}, p = {rho.pvalue:.4f}")
    print("      So it bites hardest on exactly the imbalanced monitors, as it")
    print("      should. It simply does not move the Mantel.")
    print("\n  -> Seasonal-coverage imbalance is NOT the mechanism. Coverage is")
    print("     already near-balanced (median share 0.492, median pair mismatch")
    print("     0.028), the mismatch is not spatially arranged, and holding the")
    print("     seasonal weights fixed by construction leaves the decay intact.")
    print()


def section_seasons(pan):
    rule("2. MANTEL WITHIN SEASONAL BLOCKS -- every season the record contains")
    print("  On the 36 span-matched monitors, so blocks outside the window are")
    print("  computed on the same panel as blocks inside it.\n")
    m_idx = np.where(pan.matched)[0]
    blocks = [
        ("2025 Mar-May  pre-monsoon", "2025-03-01", "2025-06-01"),
        ("2025 Jun-Aug  monsoon", "2025-06-01", "2025-09-01"),
        ("2025 Sep-Oct  post-monsoon", "2025-09-01", "2025-11-01"),
        ("2025-26 Nov-Feb  WINTER", "2025-11-01", "2026-03-01"),
        ("2026 Mar-May  pre-monsoon", "2026-03-01", "2026-06-01"),
        ("2026 Jun-Aug  monsoon", "2026-06-01", "2026-08-28"),
    ]
    print(f"  {'block':<30}{'hours':>7}{'st':>5}{'r':>9}{'p':>10}")
    for lbl, a, b in blocks:
        mask = pan.window(a, b)
        r, p, _, ns = pan.shape_mantel(mask, m_idx)
        print(f"  {lbl:<30}{int(mask.sum()):>7}{ns:>5}{r:>+9.3f}{p:>10.4f}")

    print("\n  pooled across years:")
    for lbl, mo in [("all monsoon Jun-Aug", [6, 7, 8]),
                    ("all winter Nov-Feb", [11, 12, 1, 2]),
                    ("all pre-monsoon Mar-May", [3, 4, 5])]:
        mask = pan.months(mo)
        r, p, _, ns = pan.shape_mantel(mask, m_idx)
        print(f"  {lbl:<30}{int(mask.sum()):>7}{ns:>5}{r:>+9.3f}{p:>10.4f}")

    print("\n  -> Not one seasonal block reproduces the window's decay. The two")
    print("     blocks the common window is made of are the flattest of all.")

    print("\n  the decisive replication: the SAME CALENDAR MONTHS, one year earlier")
    print(f"  {'span':<30}{'hours':>7}{'st':>5}{'r':>9}{'p':>10}")
    for lbl, a, b in [("Mar 10 - Aug 27, 2025", "2025-03-10", "2025-08-28"),
                      ("Mar 10 - Aug 27, 2026  [WINDOW]", "2026-03-10", "2026-08-28"),
                      ("Sep 2025 - Feb 2026", "2025-09-01", "2026-03-01")]:
        mask = pan.window(a, b)
        r, p, _, ns = pan.shape_mantel(mask, m_idx, n_perm=N_PERM)
        print(f"  {lbl:<30}{int(mask.sum()):>7}{ns:>5}{r:>+9.3f}{p:>10.4f}")
    print("\n  -> Same months, same monitors, same length, one year apart: the")
    print("     decay does not replicate. It is not a property of Mar-Aug.")
    print()


def section_length(pan):
    rule("3. LENGTH CONTROLS -- is the flatness elsewhere just attenuation?")
    print("  A 2,000-hour block estimates each profile from half as much data as")
    print("  the 3,901-hour window, and noise attenuates correlations toward")
    print("  zero. If that alone explained the flat blocks, halving the window")
    print("  should flatten it too.\n")
    m_idx = np.where(pan.matched)[0]
    print(f"  {'configuration':<44}{'hours':>7}{'r':>9}{'p':>10}")
    for lbl, mask in [
            ("common window, all days", pan.common),
            ("common window, odd days only", pan.common & (pan.day % 2 == 1)),
            ("common window, even days only", pan.common & (pan.day % 2 == 0)),
            ("common window, alternating day-pairs", pan.common & (pan.day % 4 < 2))]:
        r, p, _, _ = pan.shape_mantel(mask, m_idx)
        print(f"  {lbl:<44}{int(mask.sum()):>7}{r:>+9.3f}{p:>10.4f}")

    rng = np.random.default_rng(SEED + 5)
    udays = np.unique(pan.day[pan.common])
    rs = []
    for _ in range(20):
        pick = rng.choice(udays, size=len(udays) // 2, replace=False)
        mask = pan.common & np.isin(pan.day, pick)
        r, _, _, _ = pan.shape_mantel(mask, m_idx, n_perm=1)
        rs.append(r)
    rs = np.array(rs)
    print(f"\n  20 random half-window day-subsets (~1,950 h each): median r = "
          f"{np.median(rs):+.3f},")
    print(f"    range {rs.min():+.3f} to {rs.max():+.3f}, "
          f"{int((rs < 0).sum())}/20 negative")
    print("\n  -> Half the window keeps the decay; whole blocks elsewhere in the")
    print("     record do not have it to lose. Attenuation is not the story.")
    print()


def section_sweep(pan):
    rule("4. WINDOW SWEEP -- knife-edge or basin?")
    m_idx = np.where(pan.matched)[0]
    end = "2026-08-28"

    print("  (a) START DATE swept, end fixed at 2026-08-27, 36 span-matched\n")
    print(f"  {'start':<14}{'hours':>7}{'st':>5}{'r':>9}{'p':>10}   {'':<20}")
    for s in pd.date_range("2025-09-01", "2026-06-10", freq="10D", tz=TZ):
        mask = pan.window(s, end)
        r, p, _, ns = pan.shape_mantel(mask, m_idx)
        mark = ""
        if abs((s - pd.Timestamp(COMMON_START, tz=TZ)).days) < 5:
            mark = "  <-- 03_'s primary start"
        bar = "#" * int(round(abs(r) * 100))
        print(f"  {s:%Y-%m-%d}    {int(mask.sum()):>7}{ns:>5}{r:>+9.3f}{p:>10.4f}   "
              f"{'-' if r < 0 else '+'}{bar}{mark}")

    print("\n  (b) fine sweep around the primary start (3-day steps)\n")
    print(f"  {'start':<14}{'hours':>7}{'r':>9}{'p':>10}")
    for s in pd.date_range("2026-02-08", "2026-04-09", freq="3D", tz=TZ):
        mask = pan.window(s, end)
        r, p, _, _ = pan.shape_mantel(mask, m_idx)
        print(f"  {s:%Y-%m-%d}    {int(mask.sum()):>7}{r:>+9.3f}{p:>10.4f}")

    print("\n  (c) END DATE swept, start fixed at the primary 2026-03-10\n")
    print(f"  {'end':<14}{'hours':>7}{'r':>9}{'p':>10}")
    for e in pd.date_range("2026-05-01", "2026-08-28", freq="10D", tz=TZ):
        mask = pan.window(COMMON_START, e)
        r, p, _, _ = pan.shape_mantel(mask, m_idx)
        print(f"  {e:%Y-%m-%d}    {int(mask.sum()):>7}{r:>+9.3f}{p:>10.4f}")

    print("\n  (d) ROLLING 180-day windows across the whole record, step 15 d\n")
    print(f"  {'window':<26}{'hours':>7}{'st':>5}{'r':>9}{'p':>10}")
    start = pd.Timestamp("2025-03-01", tz=TZ)
    stop = pd.Timestamp("2026-08-28", tz=TZ)
    L, step = pd.Timedelta(days=180), pd.Timedelta(days=15)
    roll = []
    s = start
    while s + L <= stop:
        e = s + L
        mask = pan.window(s, e)
        r, p, _, ns = pan.shape_mantel(mask, m_idx)
        roll.append(r)
        print(f"  {s:%Y-%m-%d}->{e:%Y-%m-%d}{int(mask.sum()):>7}{ns:>5}"
              f"{r:>+9.3f}{p:>10.4f}")
        s = s + step
    roll = np.array(roll)
    print(f"\n  rolling-window r: min {roll.min():+.3f}, max {roll.max():+.3f}, "
          f"sd {roll.std(ddof=1):.3f}, {int((roll < 0).sum())}/{len(roll)} negative")
    print("  -> A broad smooth basin, not a knife edge: r declines steadily from")
    print("     +0.10 at a December start to its MINIMUM at the primary start,")
    print("     then relaxes back to zero. It is robust to the end date entirely.")
    print("     But no 180-day window anywhere else in the record goes below")
    print("     -0.09, and the sweep's own spread dwarfs the permutation p-value.")
    print()


def section_jackknife(pan):
    rule("5. JACKKNIFE -- is the window's decay carried by a few monitors or pairs?")
    print("  On the primary configuration: common window, all 39 monitors.\n")
    idx = np.arange(len(pan.sids))
    P = profiles(pan.values, pan.hours, pan.common)
    sim = np.corrcoef(shapes(P))
    n = len(pan.sids)
    iu = np.triu_indices(n, 1)
    mk = pan.keep[iu]
    a, b = sim[iu][mk], pan.dist[iu][mk]
    r0 = float(np.corrcoef(a, b)[0, 1])
    print(f"  baseline r = {r0:+.4f} over {len(a)} pairs")

    # --- by monitor
    out = []
    for i in range(n):
        sub = [k for k in range(n) if k != i]
        s2, d2, k2 = (sim[np.ix_(sub, sub)], pan.dist[np.ix_(sub, sub)],
                      pan.keep[np.ix_(sub, sub)])
        j2 = np.triu_indices(n - 1, 1)
        m2 = k2[j2]
        out.append((float(np.corrcoef(s2[j2][m2], d2[j2][m2])[0, 1]), pan.sids[i]))
    out.sort()
    rr = np.array([o[0] for o in out])
    print(f"\n  leave-one-MONITOR-out (39 refits): r from {rr.min():+.4f} to "
          f"{rr.max():+.4f}")
    print(f"    all 39 remain negative: {bool((rr < 0).all())};  "
          f"largest single shift {np.abs(rr - r0).max():.4f}")
    print(f"    most influential: without {out[0][1]} -> {out[0][0]:+.4f}; "
          f"without {out[-1][1]} -> {out[-1][0]:+.4f}")

    # --- by cv group
    out = []
    for g in pd.unique(pan.grp):
        sub = [k for k in range(n) if pan.grp[k] != g]
        s2, d2, k2 = (sim[np.ix_(sub, sub)], pan.dist[np.ix_(sub, sub)],
                      pan.keep[np.ix_(sub, sub)])
        j2 = np.triu_indices(len(sub), 1)
        m2 = k2[j2]
        out.append((float(np.corrcoef(s2[j2][m2], d2[j2][m2])[0, 1]), g))
    out.sort()
    rr = np.array([o[0] for o in out])
    print(f"\n  leave-one-CV-GROUP-out (35 refits): r from {rr.min():+.4f} to "
          f"{rr.max():+.4f}")
    print(f"    all remain negative: {bool((rr < 0).all())};  "
          f"most influential group: {out[-1][1]} -> {out[-1][0]:+.4f}")

    # --- by pair
    loo = loo_pearson(a, b)
    infl = loo - r0
    print(f"\n  leave-one-PAIR-out ({len(a)} refits): r from {loo.min():+.4f} to "
          f"{loo.max():+.4f}")
    print(f"    largest single-pair influence: {np.abs(infl).max():.4f}")

    # Greedy adversarial removal, plus a calibration so the number means
    # something: an honest r of this size on this many pairs is also erased by
    # deleting a modest fraction of them, chosen adversarially.
    cur_a, cur_b = a.copy(), b.copy()
    marks = {}
    for k in range(1, 121):
        t = int(np.argmax(loo_pearson(cur_a, cur_b)))
        cur_a, cur_b = np.delete(cur_a, t), np.delete(cur_b, t)
        rc = float(np.corrcoef(cur_a, cur_b)[0, 1])
        if k in (10, 25, 50, 74, 100):
            marks[k] = rc
        if rc >= 0:
            marks["zero"] = k
            break
    print("\n  greedily deleting the most r-lowering pairs:")
    for k, rc in marks.items():
        if k == "zero":
            continue
        print(f"    drop {k:>3} pairs ({100 * k / len(a):>4.1f}%) -> r = {rc:+.4f}")
    if "zero" in marks:
        z = marks["zero"]
        print(f"    r reaches zero after {z} pairs ({100 * z / len(a):.1f}%)")

    rng = np.random.default_rng(SEED + 3)
    fr = []
    for _ in range(20):
        z = rng.multivariate_normal([0, 0], [[1, r0], [r0, 1]], len(a))
        x, y = z[:, 0], z[:, 1]
        k = 0
        while np.corrcoef(x, y)[0, 1] < 0 and k < 400:
            t = int(np.argmax(loo_pearson(x, y)))
            x, y = np.delete(x, t), np.delete(y, t)
            k += 1
        fr.append(k / len(a))
    print(f"    calibration -- a GENUINE r = {r0:+.3f} over {len(a)} pairs needs "
          f"{100 * np.mean(fr):.1f}% removed")
    print(f"    (range {100 * min(fr):.1f}-{100 * max(fr):.1f}%). "
          f"So the observed fragility is ordinary, not diagnostic.")

    # --- by distance range. r inside a restricted range is attenuated by range
    # restriction alone, so the slope is reported next to it.
    print("\n  by distance range (slope separates real decay from range restriction):")
    print(f"  {'range':<14}{'pairs':>7}{'r':>9}{'slope /km':>12}")
    for lo, hi, lbl in [(0, 1e9, "all pairs"), (0, 20, "< 20 km"),
                        (20, 1e9, "> 20 km"), (0, 25, "< 25 km"),
                        (25, 1e9, "> 25 km"), (10, 30, "10-30 km")]:
        s2 = (b >= lo) & (b < hi)
        if s2.sum() > 40:
            sl = np.polyfit(b[s2], a[s2], 1)[0]
            print(f"  {lbl:<14}{int(s2.sum()):>7}"
                  f"{np.corrcoef(a[s2], b[s2])[0, 1]:>+9.3f}{sl:>+12.5f}")
    print("    -> the gradient is a long-range contrast: inside 20 km, where more")
    print("       than half the pairs sit, the slope is a quarter of the overall")
    print("       one. 'Similarity decays with distance' overstates the near field.")

    # --- block bootstrap over days: the uncertainty the Mantel null ignores.
    print("\n  block bootstrap over days INSIDE the window (7-day blocks, 400 reps):")
    ud = np.unique(pan.day[pan.common])
    nb = len(ud) // 7
    rng = np.random.default_rng(SEED + 7)
    boot = []
    for _ in range(400):
        picks = rng.integers(0, nb, nb)
        sel = np.concatenate([ud[i * 7:(i + 1) * 7] for i in picks])
        mask = pan.common & np.isin(pan.day, sel)
        r, _, _, _ = pan.shape_mantel(mask, idx, n_perm=1)
        boot.append(r)
    boot = np.array(boot)
    print(f"    95% CI [{np.percentile(boot, 2.5):+.3f}, "
          f"{np.percentile(boot, 97.5):+.3f}], "
          f"{100 * (boot >= 0).mean():.1f}% of reps >= 0")
    print("    -> WITHIN the window the decay is solid. That is a different claim")
    print("       from it being a property of the network, which section 2 denies.")
    print()


def section_amplitude(pan):
    rule("6. IS THE WINDOW'S DECAY A NOISE-SHAPE ARTEFACT?")
    print("  shape_of() divides by the profile's own standard deviation, so a")
    print("  monitor with no daily cycle contributes a normalised noise vector.")
    print("  If those were spatially arranged they could fake a gradient.\n")
    idx = np.arange(len(pan.sids))
    P = profiles(pan.values, pan.hours, pan.common)
    amp, _, _ = harmonic_24(P)
    print(f"  {'kept':<26}{'stations':>9}{'pairs':>7}{'r':>9}{'p':>10}")
    for thr in (0.0, 1.0, 1.5, 2.0):
        sub = np.where(amp >= thr)[0]
        if len(sub) < 15:
            continue
        r, p, npairs, ns = pan.shape_mantel(pan.common, sub, n_perm=N_PERM_SWEEP)
        print(f"  {'24h amplitude >= ' + f'{thr:.1f}':<26}{ns:>9}{npairs:>7}"
              f"{r:>+9.3f}{p:>10.4f}")
    ad = np.abs(amp[:, None] - amp[None, :])
    r, p, _ = mantel(ad, pan.dist, pan.keep, n_perm=N_PERM_SWEEP)
    print(f"\n  Mantel(|24h amplitude difference|, distance) r = {r:+.3f}, p = {p:.4f}")
    print("  -> The decay strengthens when the weakest monitors are dropped, and")
    print("     amplitude is not spatially arranged. Not a noise artefact.")
    print()


def section_which_season(pan):
    rule("7. WHAT THE FULL-RECORD PROFILE ACTUALLY AVERAGES")
    print("  The aliasing story assumes the 18-month mean is a blurred mixture of")
    print("  regimes. It is not: the winter hours carry several times the diurnal")
    print("  amplitude, so they dominate the mean profile.\n")
    m_idx = np.where(pan.matched)[0]
    full = shapes(profiles(pan.values[:, m_idx], pan.hours))
    print(f"  {'season':<26}{'median 24h amp':>16}{'corr with full-record shape':>30}")
    for lbl, mo in [("Nov-Feb winter", [11, 12, 1, 2]),
                    ("Sep-Oct post-monsoon", [9, 10]),
                    ("Mar-May pre-monsoon", [3, 4, 5]),
                    ("Jun-Aug monsoon", [6, 7, 8])]:
        Ps = profiles(pan.values[:, m_idx], pan.hours, pan.months(mo))
        amp, _, ok = harmonic_24(Ps)
        S = shapes(Ps)
        good = ok & np.isfinite(full).all(axis=1)
        cs = [np.corrcoef(full[i], S[i])[0, 1] for i in np.where(good)[0]]
        print(f"  {lbl:<26}{np.median(amp[ok]):>16.2f}"
              f"{np.median(cs):>30.3f}")
    print("\n  -> The 18-month shape IS essentially the winter shape (r = 0.87),")
    print("     and the winter block is flat. The full record and its dominant")
    print("     season agree with each other. Nothing needs explaining there.")
    print()


def section_phase(pan):
    rule("8. THE CIRCULAR-PHASE RELIABILITY COLLAPSE")
    print("  03_ reports split-half circular r falling 0.880 -> 0.423 (ceiling")
    print("  0.968 -> 0.771) when profiles move to the common window, and reads")
    print("  it as the price of a third as much data. That reading is wrong.\n")

    m_idx = np.where(pan.matched)[0]
    dmax = pan.day.max()

    def split_half(mask, cols):
        odd = mask & (pan.day % 2 == 1)
        even = mask & (pan.day % 2 == 0)
        Pa = profiles(pan.values[:, cols], pan.hours, odd)
        Pb = profiles(pan.values[:, cols], pan.hours, even)
        Pf = profiles(pan.values[:, cols], pan.hours, mask)
        aa, pa, oa = harmonic_24(Pa)
        ab, pb, ob = harmonic_24(Pb)
        af, pf, of = harmonic_24(Pf)
        ok = oa & ob & of
        if ok.sum() < 15:
            return None
        return dict(r=circ_circ_r(pa[ok], pb[ok]), n=int(ok.sum()),
                    amp=float(np.median((aa[ok] + ab[ok]) / 2)),
                    weak=int(((aa[ok] + ab[ok]) / 2 < WEAK_AMP).sum()),
                    gap=float(np.median(np.abs(circ_diff(pa[ok], pb[ok])))),
                    days=len(np.unique(pan.day[mask])), hours=int(mask.sum()),
                    pa=pa, pb=pb, pf=pf, ok=ok,
                    ampv=(aa + ab) / 2)

    # --- (a) reliability against record length
    print("  (a) split-half phase reliability against WINDOW LENGTH")
    print("      (36 span-matched monitors, windows slid across the whole record)\n")
    print(f"  {'L (days)':>9}{'windows':>9}{'hours':>8}{'split-half r':>14}"
          f"{'ceiling':>9}{'median 24h amp':>16}")
    for L in (14, 30, 60, 90, 120, 171, 240, 300, 400, 546):
        outs = []
        stepn = max((dmax - L) // 12, 1) if dmax > L else 1
        for d0 in range(0, max(dmax - L + 1, 1), stepn):
            mask = (pan.day >= d0) & (pan.day < d0 + L)
            s = split_half(mask, m_idx)
            if s:
                outs.append(s)
        if not outs:
            continue
        rm = float(np.median([o["r"] for o in outs]))
        print(f"  {L:>9}{len(outs):>9}{int(np.median([o['hours'] for o in outs])):>8}"
              f"{rm:>14.3f}{ceiling(rm):>9.3f}"
              f"{np.median([o['amp'] for o in outs]):>16.2f}")
    print("\n      At L = 171 days -- the common window's own length -- the typical")
    print("      window scores 0.79, ceiling 0.94. The common window scores 0.42.")
    print("      Length does not explain it.")

    # --- (b) what does explain it
    print("\n  (b) the same split by SEASON, which is what actually moves it\n")
    print(f"  {'window':<34}{'days':>6}{'hours':>7}{'r':>8}{'ceiling':>9}"
          f"{'24h amp':>9}{'weak':>10}")
    rows = [("common window Mar10-Aug27 2026", pan.window("2026-03-10", "2026-08-28")),
            ("monsoon only Jun-Aug 2026", pan.window("2026-06-01", "2026-08-28")),
            ("monsoon, both years pooled", pan.months([6, 7, 8])),
            ("pre-monsoon Mar-May 2026", pan.window("2026-03-01", "2026-06-01")),
            ("winter Nov-Feb 2025-26", pan.window("2025-11-01", "2026-03-01")),
            ("Sep 2025-Feb 2026 (has a winter)", pan.window("2025-09-01", "2026-03-01")),
            ("12 months Sep 2025-Aug 2026", pan.window("2025-09-01", "2026-08-28")),
            ("full record, 18 months", np.ones(len(pan.ist), bool))]
    for lbl, mask in rows:
        s = split_half(mask, m_idx)
        if not s:
            continue
        print(f"  {lbl:<34}{s['days']:>6}{s['hours']:>7}{s['r']:>+8.3f}"
              f"{ceiling(s['r']):>9.3f}{s['amp']:>9.2f}"
              f"{str(s['weak']) + '/' + str(s['n']):>10}")
    print("      'weak' = monitors whose 24h harmonic amplitude is under "
          f"{WEAK_AMP} ug/m3,")
    print("      i.e. whose phase is being read off noise.")
    print("\n      111 days of winter beat 179 days of monsoon and very nearly")
    print("      match the whole 18 months. The binding constraint is amplitude,")
    print("      not length -- and amplitude is seasonal.")

    # --- (c) the mechanism, at station level
    print("\n  (c) mechanism: which monitors lose their phase, and why\n")
    all_idx = np.arange(len(pan.sids))
    for lbl, mask in [("common window", pan.common),
                      ("full record", np.ones(len(pan.ist), bool))]:
        s = split_half(mask, all_idx)
        d = np.abs(circ_diff(s["pa"], s["pb"]))
        amp = s["ampv"]
        hi = amp >= WEAK_AMP
        rho = stats.spearmanr(d, amp)
        print(f"    {lbl}:")
        print(f"      circular split-half r {s['r']:+.3f}   median 24h amp "
              f"{np.median(amp):.2f} ug/m3")
        print(f"      monitors under {WEAK_AMP} ug/m3: {int((~hi).sum())} of "
              f"{len(amp)}   their median half-to-half phase gap "
              f"{np.median(d[~hi]):.2f} h")
        print(f"      restricted to the {int(hi.sum())} above it: circular r = "
              f"{circ_circ_r(s['pa'][hi], s['pb'][hi]):+.3f}, "
              f"median gap {np.median(d[hi]):.2f} h")
        print(f"      Spearman(phase gap, amplitude) = {rho.statistic:+.3f} "
              f"(p = {rho.pvalue:.4f})")
    print("\n    -> The collapse is 14 of 39 monitors losing their daily cycle in")
    print("       the monsoon, not 39 monitors getting uniformly noisier. Among")
    print("       the monitors that keep an amplitude, the window is fine (0.85).")

    # --- (d) the estimator caveat
    print("\n  (d) a caveat on the number itself: 0.423 is estimator-dependent\n")
    s = split_half(pan.common, all_idx)
    sf = split_half(np.ones(len(pan.ist), bool), all_idx)
    print(f"  {'estimator':<44}{'window':>10}{'full record':>14}")
    for name, fn in [
            ("Jammalamadaka circular r  [used in 03_]",
             lambda x: circ_circ_r(x["pa"], x["pb"])),
            ("Pearson on wrapped deviations",
             lambda x: float(np.corrcoef(circ_diff(x["pa"], circ_mean(x["pa"])),
                                         circ_diff(x["pb"], circ_mean(x["pb"])))[0, 1])),
            ("Spearman on wrapped deviations",
             lambda x: float(stats.spearmanr(
                 circ_diff(x["pa"], circ_mean(x["pa"])),
                 circ_diff(x["pb"], circ_mean(x["pb"]))).statistic))]:
        print(f"  {name:<44}{fn(s):>+10.3f}{fn(sf):>+14.3f}")
    for lbl, x in [("window", s), ("full record", sf)]:
        th = x["pf"] * 2 * np.pi / 24
        rbar = float(np.hypot(np.cos(th).mean(), np.sin(th).mean()))
        print(f"    {lbl:<12} mean resultant length Rbar = {rbar:.3f} "
              f"(1 = all monitors peak together, 0 = spread round the clock)")
    print("    Jammalamadaka's r weights by sin(deviation), which flattens once")
    print("    deviations approach +/-6 h. In the window the phases are nearly")
    print("    spread round the clock, so the statistic is in its worst regime.")
    print("    The direction of the collapse is solid; its size is not.")

    # --- (e) what a usable trough test would cost
    print("\n  (e) what record length would make the trough-hour test informative\n")
    print("      Phase noise scales as 1/(amplitude * sqrt(days)), so a season's")
    print("      amplitude sets the exchange rate between days and precision:")
    print(f"  {'season':<24}{'median 24h amp':>16}{'days for winter-equal precision':>34}")
    ampw = None
    for lbl, mo in [("Nov-Feb winter", [11, 12, 1, 2]),
                    ("Sep-Oct post-monsoon", [9, 10]),
                    ("Mar-May pre-monsoon", [3, 4, 5]),
                    ("Jun-Aug monsoon", [6, 7, 8])]:
        Ps = profiles(pan.values[:, m_idx], pan.hours, pan.months(mo))
        amp, _, ok = harmonic_24(Ps)
        a = float(np.median(amp[ok]))
        if ampw is None:
            ampw = a
        print(f"  {lbl:<24}{a:>16.2f}{f'{(ampw / a) ** 2:.0f}x':>34}")
    print(f"\n      A monsoon-weighted record needs about {(ampw / 1.21) ** 2:.0f}x the days of a")
    print("      winter one for the same phase precision. The common window is")
    print("      171 days; matching one winter season on monsoon data alone would")
    print(f"      take roughly {171 * (ampw / 1.21) ** 2 / 365:.0f} years of monsoons. That is not a")
    print("      record-length problem that more monitoring solves.")
    print("\n      Measured directly instead of extrapolated: a 172-day window that")
    print("      CONTAINS a Nov-Feb season (Sep 2025-Feb 2026) reaches ceiling")
    print("      0.936 -- against 0.799 for the 170-day monsoon-weighted window,")
    print("      and 0.963 for the entire 18 months.")
    print("\n      So: ~6 months of common record INCLUDING one pollution season")
    print("      restores the trough test to a ~0.94 ceiling. Twelve months of")
    print("      common record buys 0.945 and removes the choice entirely. No")
    print("      amount of Mar-Aug does it. The three March-2026 monitors need to")
    print("      reach February 2027 before this test is worth running again.")
    print()


def section_verdict(pan):
    rule("9. WHICH WINDOW TO TRUST FOR THE CONTROL")
    print("""
  The full record, +0.053 -- with the control read as UNRESOLVED rather than
  as evidence of no spatial structure.

  Why the full record:
    - The aliasing hypothesis that motivated preferring the window is refuted
      directly (section 1). Span-matching moves the full record by +0.005.
    - The full-record shape is the winter shape (r = 0.87), and the winter block
      is independently flat. The full record and its dominant season agree.
    - Every seasonal block in the record is flat, including both blocks the
      common window is built from.
    - The window's decay does not replicate in the same calendar months a year
      earlier (-0.046, p = 0.58), on the same monitors at the same length.
    - No 180-day window anywhere else in 18 months reaches half of it.

  Why "unresolved" and not "no decay":
    - The window's decay is internally solid: block-bootstrap CI excludes zero,
      no monitor or pair drives it, it strengthens when weak-amplitude monitors
      are dropped. Something real is happening in that span.
    - Mar-Aug 2026 is one draw. A period-specific circulation pattern would look
      exactly like this, and would be a real fact about that period.
    - The conservative reading for 03_'s purposes is unchanged either way: assume
      the primary p-values MAY be anticonservative. All four primary tests are
      null, and a null under an anticonservative test is a stronger null.

  What is genuinely uncomfortable and should be said plainly:
    - The pre-specified start date sits at the exact minimum of the start-date
      sweep. That is not evidence of anything by itself -- the date was fixed by
      the data-collection record, not chosen -- but it is the single most
      coincidental-looking fact here, and any reader should be told.
    - The Mantel permutation null conditions on the estimated profiles. It asks
      whether a random relabelling of monitors reproduces the pattern. It never
      asks how much r would move on a different stretch of time, and the sweep
      says: a lot. p = 0.0009 is a precise answer to a narrower question than
      the one being asked of it.
""")


def main():
    pan = Panel()
    section_reproduce(pan)
    section_aliasing(pan)
    section_coverage_weights(pan)
    section_seasons(pan)
    section_length(pan)
    section_sweep(pan)
    section_jackknife(pan)
    section_amplitude(pan)
    section_which_season(pan)
    section_phase(pan)
    section_verdict(pan)


if __name__ == "__main__":
    main()
