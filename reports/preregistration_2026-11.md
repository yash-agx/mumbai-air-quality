# Pre-registration — breeze, baseline sensitivity, monitor table, time-block CV

Written **before any of these analyses were run**, and committed on its own so
the commit order is checkable. Prompted by feedback on a public post. It fixes
what each test is, what would support or reject it, and how many independent
monitors it actually rests on — the same discipline as the land-use tests in
`scripts/03_diurnal_landuse.py` and its power section.

Everything below is decided here and not revisited: analysis window, statistic,
direction of the predicted effect, decision rule, and the confounds that make a
result unreadable. Deviations, if any are forced, get written into the results
section as deviations.

Seed 0 throughout. Significance at α = 0.05, Benjamini–Hochberg within each
numbered test's primary family only, never across tests.

---

## What the units are, and why n is not 39

39 monitors, but **35 `cv_group` units**. Four pairs sit under ~1 km apart and
are one unit each; the variogram work already flagged that bin as unexploitable
and the spatial CV already holds those pairs out together. Every correlation
below is computed across the 35 units, not the 39 stations.

**35 is the ceiling, not the effective n.** Two further deductions apply:

- Diurnal shape similarity **decays with distance in this window** (Mantel
  −0.215; `scripts/04_mantel_window.py` showed the decay does not generalise
  beyond it). Spatially autocorrelated units are worth less than independent
  ones, so every p-value here is possibly anticonservative. The conservative
  reading is to treat marginal results as null.
- The coastal contrast rests on far fewer: **6 units within 2 km of the coast**
  and 10 beyond 8 km. Median `dist_coast_m` is 4.2 km.

### Power, simulated at these n (α = 0.05)

| Test shape | n | detectable effect | power |
|---|---|---|---|
| Spearman | 35 | ρ = 0.30 / 0.40 / 0.50 | 0.37 / 0.62 / 0.84 |
| Mann–Whitney, 6 coastal vs 10 inland | 16 | d = 1.0 / 1.5 / 2.0 SD | 0.38 / 0.72 / 0.92 |
| Mann–Whitney, median split | 17 vs 18 | d = 0.5 / 0.75 / 1.0 SD | 0.29 / 0.54 / 0.80 |
| Wilcoxon signed-rank, paired | 35 | d = 0.3 / 0.4 / 0.5 SD | 0.40 / 0.60 / 0.80 |

**A null here rules out a large effect, not a real one.** Anything under
ρ ≈ 0.4, or under about 1.5 SD between coastal and inland, is beyond what this
network can resolve, and saying so is part of the result.

---

## Window

The **common window, 2026-03-10 to 2026-08-27**, as in
`scripts/03_diurnal_landuse.py`: the first local day on which every monitor
reports. Before it, monitors that came online later average a different slice
of the year, which is a seasonal difference wearing a spatial costume.

Its cost is stated up front and applies to every result below: the window is
**March–August, monsoon-weighted, and holds none of Mumbai's November–February
season**. The monsoon's prevailing flow is westerly — onshore — so offshore
hours in this window are likely to be both scarce and seasonally clustered.
If the offshore sample is too small the breeze test cannot be run at all, and
that is a reportable outcome, not a reason to widen the window.

Full-record (19-month) versions of tests 1 and 2 are **sensitivity analyses
only**, reported alongside and never promoted over the window result.

---

## Test 1 — sea breeze / land breeze

**Hypothesis.** A daytime sea breeze carries pollution inland and a night-time
land breeze carries it back over the coast, so coastal monitors peak at night
and inland monitors in the afternoon.

**This is not an independent confirmation, and must not be reported as one.**
Two prior encounters with this data constrain what a result can mean:

1. `dist_coast_m × trough_hour` was one of four **pre-specified** land-use
   tests and came back **null**.
2. An **exploratory** variant (`dist_coast_m` against a trough hour from a
   24h+12h harmonic fit) gave R = 0.535, p = 0.005, was never promoted because
   the estimator was chosen after the pre-specified one came back null, and
   failed its own robustness checks. That note's condition for a proper test
   was to "use monitors this analysis did not touch". **There are only 39
   monitors and this test reuses all of them.** So 1a below is a re-test on
   used data; only 1b (wind) brings genuinely new information.

### 1a. Peak hour against distance to coast

- Statistic: Spearman ρ between `dist_coast_m` and peak hour, across 35 units.
  Peak hour is circular, so the test is the circular–linear correlation already
  used in `scripts/03`, with a permutation p-value (10,000 permutations).
- **Supports:** inland monitors peak significantly later in the day
  (afternoon) than coastal ones — a positive circular–linear association,
  p < 0.05, in the predicted direction.
- **Rejects / null:** p ≥ 0.05, or a significant association in the opposite
  direction. Given power, I will state the largest effect the null excludes.
- Secondary, reported but not primary: Mann–Whitney on peak hour, 6 coastal
  (<2 km) vs 10 inland (>8 km).

### 1b. Wind-conditional concentration

- Wind: **Open-Meteo Historical Weather API** (ERA5 reanalysis),
  `archive-api.open-meteo.com`, hourly `wind_speed_10m` and
  `wind_direction_10m` at 19.08 N, 72.88 E, same window, UTC, converted to IST.
  Free for non-commercial use; cited in the README and in the script header.
- Classification, fixed now. Mumbai's coast runs roughly N–S with the sea to
  the **west**. Meteorological direction is the direction wind comes *from*.
  - **onshore (sea breeze):** 225°–315°
  - **offshore (land breeze):** 45°–135°
  - everything else: unclassified, dropped.
  - Hours with wind speed < 1 m/s are dropped: direction is meaningless there.
- **The confound that decides this test.** Offshore hours are mostly night and
  onshore hours mostly afternoon, so a naive offshore-vs-onshore contrast would
  re-measure the diurnal cycle and call it wind. The primary analysis is
  therefore run on **hour-of-day-residualised** concentrations: from each
  monitor's reading subtract that monitor's own mean for that hour of day, so
  what is tested is deviation from its normal behaviour at that time of day.
  The unresidualised version is reported too, explicitly labelled as
  confounded.
- Statistic: per unit, mean residual during offshore hours minus mean residual
  during onshore hours, in units of that monitor's own SD. Primary test is
  Spearman ρ between `dist_coast_m` and that difference, across 35 units.
- **Supports:** ρ significantly **negative** — coastal monitors relatively
  dirtier when the wind blows offshore — p < 0.05.
- **Rejects / null:** p ≥ 0.05, or significantly positive.
- **Hour-level p-values will not be computed.** Hours are autocorrelated and
  every monitor shares one city-wide wind series, so the weather provides
  **n = 1** independent forcing, not thousands. Inference is at the monitor
  level only. This is the single most important honesty constraint on test 1.
- Reported regardless: how many hours fall in each class, and their month and
  hour-of-day distribution, so a reader can see what the contrast is made of.

### 1c. Powai

Powai is inland and peaks at 9 am — the breeze story predicts an afternoon
peak there. Reported as a named case either way, with its wind-conditional
numbers, and with the alternative explanation stated (a morning traffic peak
and an inland boundary layer that both work on a schedule the breeze does not
set). No test is attached; a single monitor is n = 1 and this is description.

---

## Test 2 — do low-baseline monitors swing harder?

**Hypothesis.** Monitors with a lower average PM2.5 show a larger *percentage*
day–night swing.

**The artefact that would fake this result.** Percentage swing is
(max − min) / mean: the baseline appears in the denominator. Noise in the mean
alone produces a negative correlation between mean and percentage swing, with
no physical cause whatsoever. A naive test here is guaranteed to "work".

- **Primary, split-half, fixed now:** compute each unit's **mean on odd days**
  and its **percentage swing on even days**. The two statistics then carry
  independent noise and the spurious correlation is removed. Spearman ρ across
  35 units, between odd-day mean and even-day percentage swing.
- **Supports:** ρ significantly negative, p < 0.05.
- **Rejects / null:** p ≥ 0.05, or positive.
- Reported alongside, as diagnostics of the artefact rather than as results:
  the naive same-data version, and the correlation using **absolute** swing
  (max − min, µg/m³), which has no shared denominator.
- If the naive version is significant and the split-half version is not, the
  conclusion is that the effect was the artefact. That is the expected failure
  mode and it gets reported as the finding.

---

## Test 3 — monitor table (descriptive, no inference)

A README table: every monitor, its area, agency, coordinates rounded to 3
decimals, distance to the nearest major road, and distance to the coast, from
`station_features.parquet`. No hypothesis, no p-value.

It carries one required sentence, which is the point of the table: **each curve
in this project describes one instrument at one spot, not a neighbourhood.** A
monitor 50 m from a highway and one 800 m away are both labelled with the same
area name, and the diurnal chart's area labels are shorthand for the monitor,
not for where anyone lives.

---

## Test 4 — time-block cross-validation

**Question.** The shipped CV holds out whole `cv_group`s (spatial). Does the
gap between the models and the city-average baseline change when whole **time
blocks** are held out as well?

- Design: leave-one-`cv_group`-and-one-month-out. Evaluation cells are
  (group, month) pairs. To bound runtime, **4 months are drawn at random with
  seed 0** from the 19 in the record, crossed with all 35 groups: **140 cells**,
  fixed before running. If a single fit proves too slow for this to finish,
  the fallback — fixed now — is 3 months, not a change of model or metric.
- The comparison is **paired within cell**. For each cell, both schemes are
  evaluated on exactly the same held-out rows, differing only in whether the
  training set also excludes that month:
  - *spatial only:* train on other groups, all months
  - *spatial + time:* train on other groups, excluding that month
- Models: `regional_mean` (the city-average baseline), `idw`, `kriging`,
  `trees`, `trees+idw`, as shipped.
- Metric: RMSE per cell, and the **gap** = model RMSE − `regional_mean` RMSE.
- Statistic: Wilcoxon signed-rank on the per-cell change in gap, with the 35
  groups as the independent units (months within a group are not independent,
  so cells are averaged to one value per group before testing).
- **Supports a real degradation:** the gap narrows significantly under the
  time-block scheme — the models' advantage over the baseline was partly
  borrowed from having seen the same hours in training.
- **Supports the current claim:** no significant change, i.e. the shipped
  spatial-only number was not flattered by temporal leakage.
- **Stated in advance, so it is not mistaken for a finding:** `idw`,
  `kriging` and `regional_mean` predict from *concurrent* readings at other
  stations, so holding out a month cannot affect them except through fitted
  hyperparameters (the IDW power, the variogram). The tree models are the ones
  with something to leak. If the gap moves, it should move for the trees.

---

## Reporting rules

- Every test above is reported, including nulls, with effect size, p-value,
  n, and the largest effect the n could not have detected.
- No test is dropped, renamed, or re-specified after seeing its result.
- Anything not listed here that gets run is labelled **exploratory** and is
  never promoted to a finding.
