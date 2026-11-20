# Pre-registration — does spatial structure appear at longer averaging times?

Written **before the analysis was run** and committed on its own, so the order
can be checked. The only thing computed so far is how many station-windows
pass the coverage rule below. That count uses data availability and no PM2.5
values.

**Question.** The main result is that hourly PM2.5, once the citywide mean is
removed, has almost no spatial structure at this network's spacing: the
semivariance sits near 1.0 from ~2 km out, and IDW beats the city average by
under 1%. An outside reviewer pointed out that this might depend on the
averaging time. Hourly anomalies carry a lot of short-lived noise (a passing
truck, a local fire, a sensor wobble), and stable differences between places
could be buried under it at 1 h but show up in daily, weekly or monthly
averages. If so, "no spatial structure" is a statement about hourly data
only, and it has to be worded that way.

Seed 0. α = 0.05. Nothing is corrected across window lengths. The windows are
nested views of one dataset, not independent tests, and the reading below
rests on the overall pattern across windows plus two contrasts fixed in
advance, not on any single p-value.

---

## Data, units and windows

- **Data:** `data/processed/pm25_clean.parquet` as shipped, the same 39
  monitors and 376,131 hourly readings, with no further screening.
  Timestamps are hourly in UTC, so they fall on the half hour in IST
  (UTC+5:30). Windows are built on the IST clock, with each IST hour labelled
  by its :30 stamp.
- **Unit:** the **35 existing `cv_group`s** from `station_features.parquet`.
  Monitors under ~1 km apart share a group and are not independent views of
  a spatial pattern. Two monitors 70 m apart reading alike would inflate
  every reliability and correlation number. Where a metric is per-station
  (the distance pattern and the CV skill), that is stated.

**Windows** (calendar-aligned in IST; fixed-length windows are anchored at the
first IST midnight of the record, 2025-02-28 00:30):

| label | length | nominal windows | pass coverage (75%) | median groups / window |
|---|---|---:|---:|---:|
| 1h | 1 h | 13,087 | 12,216 | 29 |
| 3h | 3 h | 4,363 | 3,964 | 28 |
| 8h | 8 h | 1,636 | 1,524 | 28 |
| 24h | IST calendar day | 546 | 503 | 27 |
| 2d | 48 h | 273 | 249 | 27 |
| 3d | 72 h | 182 | 168 | 27 |
| 1w | 168 h | 78 | 72 | 27 |
| 2w | 336 h | 39 | 37 | 26 |
| month | calendar month | 19 | 17 | 26 |
| season | IMD seasons: DJF, MAM, JJAS, ON | 7 | 5 | 27 |
| all | whole record | 1 | 1 | **24** |

At 50% coverage the counts rise slightly (e.g. 1w 76, month 18, season 6,
whole record 34 stations in 31 groups).

### Coverage rule

- A **station-window is kept only if the station has valid readings for at
  least 75% of the window's nominal hours**, i.e. ⌈0.75 × hours⌉: all 3 of a
  3 h window, 6 of 8 h, 18 of 24 h, and so on. Partial windows at the ends of
  the record are judged against their full nominal length, so a stub fails.
- A **window is kept only if ≥ 10 stations pass**, the same floor the episode
  round used for a rankable hour.
- The 75% rule costs the whole-record window 15 of 39 stations (24 pass). The
  rule is kept anyway. Loosening it only for long windows would make the long
  windows less comparable with the short ones, which is the comparison this
  study is about.
- **Sensitivity:** everything is re-run at 50% coverage (34 stations pass for
  the whole record) and reported next to the primary. If the two disagree on
  a reading, that is reported. The primary is not swapped for the
  sensitivity.

### Anomalies

For each kept window *w*: station mean *x*ₛ,w = mean of its valid hours;
citywide mean *c*w = mean of *x*ₛ,w over kept stations; standardised
anomaly *z*ₛ,w = (*x*ₛ,w − *c*w) / sdw, with sdw the across-station
standard deviation in that window. This is the same mean-and-spread removal
the main variogram uses, now applied to window means instead of hours.
Standardising keeps the dirty winter windows, where everything is more
spread out, from dominating the pooled numbers. A group's anomaly is the mean
of its member stations' *z*.

---

## (a) Spatial share

A one-way random-effects decomposition of group anomalies *z*g,w over windows:

- **Raw share** = variance across groups of each group's mean *z* ÷ total
  variance of *z*. This is the quantity as first specified, and it is
  reported. **It is biased upward as windows get fewer**: with *n* windows of
  pure noise it comes out near 1/(*n*+1), and at one window it is exactly 1.
  So it cannot be read on its own.
- **Noise-corrected share (primary for (a))** = ANOVA intraclass correlation,
  (MSB − MSW) / (MSB + (n₀ − 1)·MSW), floored at 0 for display (the unfloored
  value is kept in the CSV). It estimates the fraction of a single window's
  anomaly variance that is stable between groups, and it is near 0 for pure
  noise whatever the window count.
- **Sensitivity:** the same decomposition on unstandardised anomalies in
  µg/m³ (*x* − *c*).
- Undefined for `all` (one window, so no within-group variance).

## (b) Reliability — the noise check

**A change from the first specification, and why.** Taken literally, "split
each station's windows into odd and even, average each half, correlate across
stations" would not test the hypothesis. Each half pools roughly half of the
whole 18-month record however long the windows are. So the statistic measures
whether the *long-run mean* anomaly is stable, and it should come out about
the same at 1 h as at 2 weeks. For short windows it is also inflated, because
odd and even hours of the same day share the same weather. On a synthetic
field of stable pattern plus independent hourly noise it would be flat, not
rising. The test asked for in `tests/test_averaging_times.py` (reliability
rising with averaging time) would fail by construction.

So both are computed, with distinct names:

- **(b1) Split-half reliability of the long-run pattern**, as first
  specified: windows numbered in time order, odd vs even; each group's mean
  *z* over each half; Pearson r across groups. This answers "is there a
  stable difference between places *at all*", and it is reported at every
  window. Predicted: roughly flat across windows up to a month, highest at
  the shortest windows (the interleaving inflation). Undefined for `all`.
- **(b2) Single-window reliability (primary)**: the across-group Pearson
  correlation between the anomaly patterns of **two different windows of the
  same length**, averaged over window pairs. This is the expert's question:
  how much of one window's spatial pattern is the same in another window, and
  how much is noise. Pure noise gives ~0. A stable pattern plus noise gives a
  value that rises with window length as the noise averages out.
  - Pairs must be **separated by ≥ 7 days** (end of one to start of the
    other), so that one weather spell, a multi-day plume or persistence does
    not count as stable structure. Adjacent months and adjacent seasons are
    therefore excluded.
  - A pair needs ≥ 10 groups present in both windows.
  - If more than 50,000 eligible pairs exist, 50,000 are drawn at random
    (seed 0); otherwise all are used.
  - Reported: mean and median pair correlation, number of windows, number of
    pairs, median groups per pair.
  - **Reference curve, not a test:** if hourly noise were independent,
    reliability at *h* hours would follow Spearman–Brown from the 1 h value,
    r(h) = h·r₁ / (1 + (h − 1)·r₁). Real noise is autocorrelated, so the
    observed curve should rise more slowly. The gap between the two is
    reported.
  - Undefined for `all`. For `season`, fewer than 15 pairs exist, so it is
    descriptive only (see thin-data rule).

## (c) Distance pattern

Per **station pair** (distance is a property of stations, not groups), pooled
over windows: semivariance γ = ½·mean over shared windows of (*z*ᵢ − *z*ⱼ)²,
reported as **spatial correlation 1 − γ** (≈ 0 for no structure, matching
the main table's semivariance ≈ 1).

- Distance bins: 0–2, 2–5, 5–10, 10–15, 15–20, 20–30, ≥ 30 km. Pairs in the
  same `cv_group` (all < 1 km) are reported as their own row and **excluded
  from the summary contrast**, because cross-validation never gets to use
  them.
- A pair counts only if it shares ≥ 3 windows (≥ 1 for `all`).
- **Summary contrast:** mean (1 − γ) for cross-group pairs < 5 km minus that
  for pairs ≥ 15 km. Tested with a Mantel-style permutation, 999 permutations
  of station labels over the distance matrix (seed 0).
- Number of pairs per bin and number of windows reported everywhere.
- Check: at 1 h this should reproduce the main variogram (nearest-bin
  semivariance ~0.96, ~1.05–1.09 beyond). If it does not, the code is wrong
  and is fixed before anything else is read.

## (d) Prediction skill

Grouped spatial CV on window means, with **the same 35 leave-one-`cv_group`-out
folds** as the shipped model:

- For held-out group *g* and window *w*: **city average** = mean of *x*ₛ,w
  over kept stations outside *g*. **IDW** = `idw_estimate` from
  `model/interpolate.py` on the same stations, with the power chosen inside
  the fold by `select_idw_power` on the window-mean panel. Power is re-tuned
  for each window length, so a long window is not handicapped by an hourly
  power.
- Metric: RMSE of each per group, in µg/m³; **gap = IDW RMSE − city-average
  RMSE** (negative = IDW wins), also reported as % of the city-average RMSE.
  Mean gap across groups; Wilcoxon signed-rank across the 35 per-group gaps.
- Check: at 1 h this should land within ~0.05 µg/m³ of the shipped CV
  (IDW 21.117, city average 21.223). The ≥ 10-station window rule is the only
  intended difference.
- Station-hours, station-windows and groups behind each number are reported.

---

## Intervals and the thin-data rule

- **Intervals:** cluster bootstrap over **calendar months** (each window
  assigned to the month it starts in), 1,000 resamples, seed 0, 95%
  percentile intervals, for (a), (b1), (b2), (c) and the (d) gap. Months
  rather than windows, because a week's anomaly pattern is not independent
  of the next week's. For `month` this is a bootstrap over the 17 windows
  themselves. For `season` and `all`, no interval is computed.
- **Thin-data rule, fixed now:**

| windows kept | label | how it is read |
|---|---|---|
| ≥ 30 | interpretable | intervals and contrasts read normally |
| 8–29 | thin | reported with interval, read with caution, never decisive on its own |
| < 8 | **too thin** | descriptive only, no interval, no verdict |

  By the counts above: 1h–2w interpretable (2w has 37), month thin (17),
  season and all too thin (5 and 1). Every table row carries its label. The
  `all` row in (d) still rests on 24 stations across ≤ 24 groups and gets a
  Wilcoxon p, but it is one snapshot of the record, and that is said next to
  it.

---

## How the results will be read

Two primary contrasts, both **1 h vs 1 week**. 1 week is the longest window
with comfortable n (72 windows over ~17 months), and the contrasts are fixed
here so the endpoint cannot be picked after looking:

- **C1, noise:** b2(1w) − b2(1h), month-cluster bootstrap interval.
- **C2, skill:** IDW gap as % of city-average RMSE at 1w minus the same at
  1h, month-cluster bootstrap interval.

Three possible readings, decided by the following rules:

1. **Noise all the way down.** C1's interval includes 0, or b2 stays < 0.2
   at every interpretable window. Stable spatial differences do not emerge
   with averaging at this network, and the main result holds at every
   averaging time tested.
2. **Stable but local.** C1 > 0 (interval excludes 0) and b2(1w) ≥ 0.3,
   **but** the (c) near-minus-far contrast at 1w has permutation p ≥ 0.05
   **and** C2's interval includes 0 or is positive. Places do differ
   stably, but neighbouring monitors are not more alike than distant ones,
   and IDW gains nothing from it. Each monitor's stable offset belongs to its
   own surroundings and is not a smooth field between monitors 1–40 km apart.
   The README's "no spatial structure" would be reworded to "no spatially
   *smooth* structure at this spacing", with the averaging-time result next
   to it.
3. **Stable and smooth.** C1 > 0, b2(1w) ≥ 0.3, the (c) contrast at 1w is
   positive with p < 0.05, **and** C2 < 0 with its interval excluding 0. The
   main result depends on averaging time. The README's hourly claim would be
   restricted explicitly to hourly data, and the longer-window skill reported
   as a finding.

Anything that fits none of these (e.g. C1 > 0 with b2 between 0.2 and 0.3,
or (c) significant while C2 is null) is reported as **mixed**, saying exactly
which criteria were met and which were not. It is not rounded to the
nearest reading.

**Predicted outcome**, stated so it can be wrong: reading 2. The diurnal
work already found stable, station-specific behaviour that land use within
1 km did not explain and that IDW flattens, and the hourly variogram is
nearly pure nugget. Both point to local offsets rather than a smooth field.

**What the design cannot answer.** With 35 groups, a smooth field whose
correlation length is under ~2 km is invisible: only same-group pairs and a
handful of cross-group pairs are that close. A null in (c) is therefore "no
smooth structure at 2–40 km", not "no smooth structure". And the 18-month
record holds five complete seasons (MAM, JJAS and ON 2025, DJF 2025–26,
MAM 2026; JJAS 2026 is cut off in August and fails coverage),
so nothing at the seasonal or whole-record scale generalises past this
record.

---

## Reporting rules

- Every number carries the windows, groups/stations and (where relevant)
  pairs behind it, and its thin-data label.
- Nulls are reported in the same detail as positives.
- Both coverage levels are reported. The primary is 75%.
- Anything not specified above is labelled exploratory and never promoted
  into the reading.
- Existing results and files are not changed. The new work goes in
  `scripts/11_averaging_times.py`, `reports/averaging_times.md`,
  `reports/figures/averaging_time_structure.png`,
  `tests/test_averaging_times.py` and a new README section.
