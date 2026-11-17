"""Test 2 of reports/preregistration_2026-11.md: do low-baseline monitors
swing harder?

Hypothesis as posted: a monitor with a lower average PM2.5 shows a larger
percentage day-night swing.

The whole design is about one artefact. Percentage swing is

    (max - min) / mean

so the baseline is in the denominator. Noise in the mean alone drives a
negative correlation between mean and percentage swing with no physical cause
at all, and a naive test is guaranteed to "find" the effect.

The pre-registered primary is therefore a SPLIT-HALF: each unit's mean comes
from odd days and its percentage swing from even days, so the two statistics
carry independent noise and the shared-denominator artefact is gone. The naive
version and an absolute-swing version are reported next to it as diagnostics
of the artefact, not as results.

Units are the 35 cv_groups, not the 39 monitors, for the reason given in
scripts/03_diurnal_landuse.py: monitors under ~1 km apart are not independent.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent


def load_diurnal_module():
    spec = importlib.util.spec_from_file_location(
        "diurnal_landuse", ROOT / "scripts" / "03_diurnal_landuse.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pct_swing(profile):
    """(max - min) / mean of a 24-hour profile, in percent."""
    m = np.nanmean(profile)
    if not np.isfinite(m) or m <= 0:
        return np.nan
    return float(100 * (np.nanmax(profile) - np.nanmin(profile)) / m)


def abs_swing(profile):
    return float(np.nanmax(profile) - np.nanmin(profile))


def station_stats(mod, wide, hours, days, mask):
    """Per station: means and swings, whole-window and split by day parity."""
    odd = mask & (days % 2 == 1)
    even = mask & (days % 2 == 0)
    prof_all = mod.profiles_for(wide, hours, mask)
    prof_odd = mod.profiles_for(wide, hours, odd)
    prof_even = mod.profiles_for(wide, hours, even)

    rows = []
    for sid in wide.columns:
        v_all = prof_all.loc[sid].to_numpy()
        v_odd = prof_odd.loc[sid].to_numpy()
        v_even = prof_even.loc[sid].to_numpy()
        rows.append({
            "station_id": sid,
            "mean_all": float(np.nanmean(v_all)),
            "pct_swing_all": pct_swing(v_all),
            "abs_swing_all": abs_swing(v_all),
            "mean_odd": float(np.nanmean(v_odd)),
            "pct_swing_even": pct_swing(v_even),
            "abs_swing_even": abs_swing(v_even),
        })
    return pd.DataFrame(rows).set_index("station_id")


def to_units(st, feat):
    d = st.join(feat[["cv_group"]])
    return d.groupby("cv_group").mean(numeric_only=True)


def report(mod, label, x, y, predicted=-1, note=""):
    rho, p, n = mod.perm_test(x, y, circular=False)
    verdict = ("supports" if (p < 0.05 and np.sign(rho) == predicted)
               else "SIGNIFICANT, WRONG DIRECTION" if p < 0.05 else "null")
    print(f"  {label}")
    print(f"    rho = {rho:+.3f}   p = {p:.4f}   n = {n} units   -> {verdict}")
    if note:
        print(f"    {note}")
    return rho, p, n


def run_window(mod, wide, feat, hours, days, mask, label):
    print("-" * 72)
    print(f"[{label}]")
    print("-" * 72)
    st = station_stats(mod, wide, hours, days, mask)
    u = to_units(st, feat)
    ok = u[["mean_odd", "pct_swing_even", "mean_all", "pct_swing_all",
            "abs_swing_even"]].notna().all(axis=1)
    u = u[ok]
    print(f"  {len(u)} units with both halves usable")
    print(f"  daily mean across units: {u['mean_all'].min():.1f} to "
          f"{u['mean_all'].max():.1f} ug/m3 "
          f"(median {u['mean_all'].median():.1f})")
    print(f"  percentage swing:        {u['pct_swing_all'].min():.0f}% to "
          f"{u['pct_swing_all'].max():.0f}% "
          f"(median {u['pct_swing_all'].median():.0f}%)")
    print()

    print("  PRIMARY (pre-registered): odd-day mean vs even-day % swing")
    primary = report(mod, "    independent noise in the two statistics",
                     u["mean_odd"], u["pct_swing_even"])
    print()
    print("  DIAGNOSTICS, not results:")
    naive = report(mod, "    naive: same-data mean vs same-data % swing",
                   u["mean_all"], u["pct_swing_all"],
                   note="the mean is in the denominator here, so this one "
                        "cannot come out any other way")
    absolute = report(mod, "    absolute swing (ug/m3), no shared denominator",
                      u["mean_odd"], u["abs_swing_even"], predicted=+1,
                      note="dirtier places swing more in absolute terms is a "
                           "different claim, and the expected direction is +")
    print()
    return {"units": u, "primary": primary, "naive": naive,
            "absolute": absolute}


def main():
    mod = load_diurnal_module()
    interp = mod.load_engine()
    wide, feat, _ = interp.load()
    ist = wide.index.tz_convert(mod.TZ)
    hours = ist.hour.to_numpy()
    days = ist.dayofyear.to_numpy()
    common = np.asarray(ist >= pd.Timestamp(mod.COMMON_START, tz=mod.TZ))

    print("=" * 72)
    print("TEST 2 -- DO LOW-BASELINE MONITORS SWING HARDER?")
    print("=" * 72)
    print("Prediction: lower average PM2.5 -> larger percentage swing, so the")
    print("pre-registered direction is a NEGATIVE correlation.")
    print()

    res = run_window(mod, wide, feat, hours, days, common,
                     "PRIMARY -- common window 2026-03-10 onward")
    print("=" * 72)
    print("SENSITIVITY -- full record (pre-registered as sensitivity only)")
    print("=" * 72)
    run_window(mod, wide, feat, hours, days,
               np.ones(len(ist), dtype=bool), "full record")

    naive_rho, naive_p, _ = res["naive"]
    prim_rho, prim_p, _ = res["primary"]
    print("=" * 72)
    print("READING")
    print("=" * 72)
    if naive_p < 0.05 and prim_p >= 0.05:
        print("  The naive version is significant and the split-half is not.")
        print("  Per the pre-registration that is the finding: what looked")
        print("  like a relationship was the shared denominator.")
    elif prim_p < 0.05 and np.sign(prim_rho) < 0:
        print("  The split-half survives, so the effect is not the artefact.")
    else:
        print("  Both null, or the split-half points the other way; see above.")
    print(f"  naive rho {naive_rho:+.3f} (p {naive_p:.4f}) vs split-half "
          f"{prim_rho:+.3f} (p {prim_p:.4f})")


if __name__ == "__main__":
    main()
