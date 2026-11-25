# Pre-registration — satellite PM2.5 against the monitor network

Written **before any monitor value for the comparison period was fetched and
before any satellite PM2.5 value was looked at**, and committed on its own, so
the order can be checked. What has been computed so far is availability only:

- **Satellite** (`data/interim/satellite_mumbai_2024.nc`, cropped by
  `scripts/14_crop_satellite_mumbai.py`): 366 days, a 46 × 31 grid at
  0.01°, and 1,152 land cells. **Every land cell is valid on every day**, so
  the field is gap-free and model-filled, not a raw retrieval. Counts of
  finite cells were checked. No value was read.
- **Monitors in the satellite box:** 31 of the 39 panel monitors, in 28
  `cv_group`s. The 8 outside it (Kalyan, Kalamboli, Taloja, Ulhasnagar ×2,
  Bhiwandi, Ambernath, Dombivli) all lie east of 73.05° E and are out of the
  comparison. For one monitor, Siddharth Nagar-Worli (6959), the nearest grid
  cell is a sea cell, so the cell-matching rule below has to handle it.
- **2024 monitor data: none available from OpenAQ.** At every one of the 39
  panel locations, the PM2.5 sensor fed from CPCB ends between Jul and Oct
  2022 and its replacement starts on or after 2025-02-18. The only PM2.5
  sensor in the bounding box that covers 2024 is location 8039 (the bare
  "Mumbai" AirNow site, excluded in `data/clean.py`). One site cannot
  support any between-monitor test.

**The data window is therefore not yet fixed.** One of these will be used,
and the choice will be recorded in an addendum (below) **before any test is
run**:

1. 2024 hourly PM2.5 for the same stations, obtained outside OpenAQ (for
   example the CPCB data portal), against the 2024 satellite file; or
2. satellite files for the existing panel's span (2025-02 onward), cropped by
   the same script, against the shipped `pm25_clean.parquet`.

The hypotheses, metrics, thresholds and decision rules below do not depend
on which is chosen, and they are fixed by this file.

Seed 0. α = 0.05. 1,000 bootstrap resamples, 999 permutations.

---

## Data and preprocessing (fixed for either source)

- **Monitor cleaning:** `clean()` from `data/clean.py`, unchanged (QA flags,
  nulls, range, hour snapping, duplicate averaging, saturation, lone spikes,
  stuck runs). Under option 2 the shipped `pm25_clean.parquet` is already this.
- **Daily means:** IST calendar days. A station-day is kept only if it has
  **≥ 18 cleaned hours**, and its value is the mean of those hours.
  Implemented in `data/fetch_2024.py::daily_means`, output
  `data/interim/monitors_<period>_daily.parquet`.
- **Satellite day:** the file date is taken as the IST calendar day. Check
  (reported, does not change the primary): the Pearson r of the two citywide
  daily series at lags −1, 0, +1 days. If lag 0 is not the highest, this is
  stated next to every result.
- **Cell matching:** each monitor is matched to the **nearest land cell**
  (finite on every day), by haversine distance from the monitor to the cell
  centre. Each match distance is reported. For 6959 this is the nearest land
  cell, not the nearest cell.
- **Unit of analysis: the `cv_group`.** Monitors under ~1 km apart share a
  group, as in every earlier analysis. A group-day value is the mean over
  member monitors with a valid day, and the satellite value for that
  group-day is the mean of **those same monitors'** matched cells.
- **Matched panel (primary for every hypothesis):** a group-day enters only
  if the monitor side has it, and both datasets are then scored on exactly
  those group-days. **A day is kept if ≥ 10 groups are present.**
- **Minimum for a confirmatory run:** ≥ 15 groups that appear on ≥ 60 kept
  days each, and ≥ 120 kept days. Below that, everything is run and reported
  but labelled exploratory, and no hypothesis is scored.

**Notation.** x_gd is the value for group g on day d (either dataset).
m_d = mean of x_gd over the groups present on day d (the **citywide mean**).
a_gd = x_gd − m_d is the **anomaly**.

**Intervals:** cluster bootstrap over **calendar months**, 1,000 resamples,
seed 0, 95% percentile intervals. **The same month resamples are used for
both datasets**, so every satellite − monitor difference has a paired
interval.

---

## H1 — the satellite field has meaningful spatial structure between monitors

*After removing each day's citywide mean, the satellite values at the monitor
locations differ between places in a way that is stable and not trivially small.*

Two metrics, computed on the satellite anomalies a_gd over the matched panel:

- **Stability, b2:** for every pair of kept days at least 7 days apart with
  ≥ 10 groups in common, the Pearson r across the common groups between the
  two days' anomaly patterns. b2 is the mean over pairs (all pairs, or
  50,000 drawn with seed 0 if there are more). This is the b2 used in
  `scripts/11_averaging_times.py`, at the 24 h window.
- **Amplitude, A:** for each kept day, the SD across groups of a_gd. A is
  the median over days, in µg/m³.

| result | rule |
|---|---|
| **Supported** | b2 95% interval lower bound **> 0.30** and A **≥ 3 µg/m³** |
| **Counts against** | b2 interval upper bound **< 0.30**, or A **< 3 µg/m³** |
| Inconclusive | anything else |

The 0.30 bar sits above the monitors' own 24 h b2 on the 2025–26 panel
(0.21, interval 0.15–0.30, `reports/averaging_times.csv`). "Meaningful" then
means more stable than anything the monitors have shown. 3 µg/m³ is roughly a
tenth of a typical daily citywide level, below which a stable pattern would
be too small to matter for exposure.

**Reported with H1 but not scored:** the same two metrics for the monitors
on the matched panel. The distance pattern for the satellite, near − far of
1 − γ on standardised anomalies (pairs < 5 km minus pairs ≥ 15 km,
cross-group only), with a Mantel-style permutation p. Also b2 and A over
all 1,152 land cells and all satellite days, as a sensitivity that does not
depend on where the monitors are.

**Prediction:** supported, with b2 ≥ 0.6. A gap-filled product built from
largely static predictors (land use, emissions inventories, terrain) should
repeat the same spatial pattern day after day. If so, high stability is a
property of the model and not evidence that the pattern is real. H3b
(below) is the test of whether it is real.

---

## H2 — day-to-day variation is dominated by the citywide signal, in both datasets

*The same day-wise decomposition is applied to both datasets on the same
group-days.*

**Citywide share C** = SS_between / SS_total, where
SS_total = Σ_gd (x_gd − x̄)², SS_between = Σ_d n_d (m_d − x̄)², x̄ is the
mean over all matched group-days, and n_d is the number of groups on day d.
SS_total = SS_between + SS_within exactly, and 1 − C is the spatial
(within-day) share. It is computed in µg/m³, separately for monitors and
satellite.

| result | rule |
|---|---|
| **Supported** | C 95% interval lower bound **> 0.50** for **both** datasets |
| **Counts against** | C point estimate **≤ 0.50** for **either** dataset |
| Inconclusive | anything else |

**Pre-specified secondary: first differences.** Levels share the seasonal
cycle (monsoon lows, winter highs), which on its own can push C up. So the
same decomposition is repeated on Δx_gd = x_gd − x_g,d−1 (consecutive kept
days only), with the same 0.50 rule. If levels pass and differences fail, the
reading is "dominated by the citywide signal through the seasonal cycle, not
from one day to the next", and it is reported that way.

**Also reported:** the paired difference C_sat − C_mon with its interval. C on
log values. And per group, the leave-one-out r² of x_gd against the mean of
the other groups that day (median and IQR across groups).

**Prediction:** supported on levels, with C_mon ≈ 0.75–0.90 and C_sat ≈
0.85–0.97, and C_sat − C_mon above 0 (the satellite is smoother than the
monitors). On first differences, C_mon is lower (≈ 0.5–0.7) and may land in
"inconclusive".

---

## H3 — satellite cells at monitor locations track those monitors' daily PM2.5

**Metric:** for each group with ≥ 60 matched days, the Pearson r between the
monitor daily series and the satellite daily series for that group.
**Summary:** the median r across groups, with a month-bootstrap interval.

| result | rule |
|---|---|
| **Supported** | median r **≥ 0.70** and its interval lower bound **≥ 0.60** |
| **Counts against** | median r **< 0.60** |
| Weak | anything else |

**Expected:** median r ≈ 0.80–0.90, driven mostly by the shared seasonal
cycle. Also reported per group: r, RMSE, mean bias (satellite − monitor) and
match distance. Also the pooled r over all matched group-days.

**H3b, pre-specified secondary: do the anomalies agree?** This tests whether
the satellite's spatial pattern (H1) is the monitors' spatial pattern. It has
two parts:
- the pooled Pearson r between satellite a_gd and monitor a_gd over all
  matched group-days, with a month-bootstrap interval;
- the Spearman ρ across groups between each group's mean satellite anomaly
  and its mean monitor anomaly (which places run dirty on average), with a
  permutation p over group labels (999 permutations).

H3b is reported in full but is not part of the H3 decision. **Prediction:**
pooled anomaly r ≈ 0.0–0.3. The monitors' own day-scale anomaly pattern is
barely stable (b2 ≈ 0.2), so there is little for any field to match from one
day to the next. The mean-anomaly ρ may be moderate (≈ 0.3–0.6) if the
satellite and the monitors agree on which areas run dirty on average.

**Caveat that travels with H3.** Satellite PM2.5 products of this kind are
usually trained against ground monitors. If these stations, for this period,
were in the product's training data, H3 measures in-sample fit, not
independent skill, and a pass is described as **consistency**, not
validation. The product's documentation is checked for this before the run,
and what it says (or that it says nothing) goes in the addendum. Under
option 2, if the product's training period ends before the window, this
caveat does not apply.

---

## Reporting rules

- Every number is reported with the groups, monitors and days behind it.
- Each hypothesis is scored exactly by its table and not rounded toward a
  neighbouring outcome. Nulls get the same detail as positives.
- Anything not specified here is labelled exploratory.
- Existing results and files are not changed. New work:
  `data/fetch_2024.py` (or a loader for whichever source is chosen),
  `scripts/15_satellite_vs_monitors.py`, `reports/satellite_vs_monitors.md`.

---

## Addendum (to be completed before any test is run)

To be committed on its own after the data are in hand and before
`scripts/15_satellite_vs_monitors.py` is run. Availability only:

- data source and window chosen, and why;
- monitors and groups with usable days, and days per monitor;
- kept days under the ≥ 10-groups rule, and whether the confirmatory minimum
  is met;
- match distance per monitor;
- what the satellite product's documentation says about its training stations
  and period.
