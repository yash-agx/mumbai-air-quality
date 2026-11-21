# Pre-registration: does land use explain NO2 better than PM2.5?

Written **before the analysis was run** and committed on its own, so the order
can be checked. No relationship between NO2 and any land-use feature has been
computed or looked at. What has been computed so far uses no land use, or no
concentrations at all:

- the per-monitor NOx/NO2 ratio, used only to fix the list of 10 monitors
  for sensitivity S2 below. It uses concentrations but no land use.
- the spread and cross-correlation of the two road features over the 35
  groups. Features only.
- a power simulation on **synthetic** outcomes at the real feature values.
  No concentrations.

**Question.** The NO2 positive control ([`no2_positive_control.md`](no2_positive_control.md))
failed: NO2 showed no more distance structure or IDW skill than PM2.5. An
expert pointed out that land-use regression (LUR) works much better for NO2
than for PM2.5, because NO2 falls off within tens to hundreds of metres of a
road. Monitors 1–40 km apart cannot see that through spatial correlation. A
model that reads each monitor's *own* road environment can. So if NO2's
long-run differences between monitors do not follow its road environment
better than PM2.5's do, that points at the NO2 data, not at the physics.

**Same methods as the PM2.5 land-use tests.** Everything that can be reused
from `scripts/03_diurnal_landuse.py` (the tests that
[`preregistration_2026-11.md`](preregistration_2026-11.md) takes as its model)
is reused unchanged, so NO2 and PM2.5 are tested the same way:

- the six OSM features in `data/processed/station_features.parquet`, from
  `scripts/02_features.py`, not recomputed;
- the 35 `cv_group` units, built with 03's `to_groups` convention: a group's
  outcome and features are the means of its member monitors';
- the permutation machinery: 20,000 permutations, seed 0, the +1
  correction, and 03's `perm_test` for the per-feature Spearman tests;
- Benjamini–Hochberg within a family only, never across families;
- 03's two controls reported next to every test: a split-half reliability
  ceiling, and a Mantel check for spatial autocorrelation;
- 03's power reasoning, re-simulated here for the new statistics.

**One deliberate difference from 03:** the data window. 03 used the common
window (2026-03-10 to 2026-08-27). This test uses the full matched panel from
the NO2 control, because that is the panel the NO2 result is about. The
common window is run as sensitivity S3.

**What has already been seen, so this is not blind on the PM2.5 side.** 03
printed building density against each monitor's mean PM2.5 level in the
common window. The shipped tree model used all six features for hourly PM2.5
and did worse than the city average. Neither used the matched panel, the
anomaly outcome or this two-feature model, and neither touched NO2. The NO2
control did show that NO2 has a stable between-monitor pattern (b2 at 1 h of
0.248) and that four near-flat sensors produce much of it.

Seed 0. α = 0.05. Code to be written: `scripts/13_no2_landuse.py`. Outputs:
`reports/no2_landuse.md`, `reports/no2_landuse.csv`,
`reports/figures/no2_landuse.png`.

---

## Data

- **Panel:** the primary matched panel from the NO2 control, unchanged. It
  takes NO2 from `data/raw/no2_raw.parquet` with that control's
  pre-registered cleaning (`clean_no2` in `scripts/12_no2_positive_control.py`,
  spike screen on, zeros kept) and the shipped `pm25_clean.parquet`. A
  station-hour is used only when it has both a cleaned PM2.5 and a cleaned
  NO2 reading (364,177 station-hours, 39 monitors, 35 groups). Both
  pollutants are scored on exactly the same rows.
- **Units:** NO2 in its labelled unit, PM2.5 in µg/m³. Every skill measure
  below is a ratio, so the two can be compared without converting units.

## Outcomes

Each is computed per monitor, then averaged to its `cv_group`, for each
pollutant separately.

- **Level:** the monitor's mean over its matched hours.
- **Anomaly (primary):** the monitor's mean of (*x*ₛ,ₜ − *c*ₜ) over its
  matched hours. *c*ₜ is the mean of all monitors on the matched panel at hour
  *t*. Only hours with ≥ 10 monitors count (the 1 h window rule used before).
  The level carries *when* a monitor reported as well as *where* it is: a
  monitor that missed the winter looks cleaner. The anomaly removes that by
  comparing each monitor with the city in the same hour, which is why it is
  primary. Both outcomes get every test, but the reading rules use the
  anomaly.
- A monitor needs ≥ 2,000 matched hours to enter. The run reports how many
  qualify. No threshold is changed afterwards.

## Predictors (fixed now, on physical grounds)

Exactly two, both from `scripts/02_features.py`:

1. **log₁₀(`dist_major_road_m`)**: distance to the nearest
   motorway/trunk/primary/secondary road. Near-road NO2 falls off roughly
   exponentially over the first few hundred metres, so it is linear in log
   distance more than in distance. Across the 35 groups the distance runs
   4 m to 1.08 km (median 112 m), which is the range where the gradient is
   steepest. **Expected sign: negative.**
2. **`road_density_500m`**: km of drivable road per km² within 500 m, a
   proxy for local traffic emissions. **Expected sign: positive.**

They measure different things: Spearman ρ = −0.03 between them across groups.
The other four features (`road_density_1km`, `dist_coast_m`,
`dist_industrial_m`, `building_density_500m`) are **exploratory only**. They
are not road-traffic proxies, or (1 km density) they overlap with predictor 2.

**Model:** ordinary least squares, outcome ~ intercept + the two predictors,
across groups. No interactions, no selection, no regularisation.

## Validation

- **Leave-one-`cv_group`-out**, 35 folds. Fit on 34 groups and predict the
  held-out one.
- **Baseline ("city mean"):** the intercept-only model in the same folds,
  i.e. the mean of the 34 training groups. For the anomaly this is close to 0,
  the city average itself.
- **Skill:** R²cv = 1 − SSE(land use) / SSE(baseline), over the 35 held-out
  predictions. It can be negative: land use then predicts worse than the city
  mean. RMSE is reported for both models in the pollutant's unit, along with
  the gap as % of the baseline RMSE.
- **Permutation p:** shuffle the outcome across the 35 groups with features
  fixed and recompute R²cv, 20,000 times (seed 0). One-sided:
  p = (#{null ≥ observed} + 1) / 20,001.
- Full-sample OLS coefficients are reported with their signs (needed for the
  reading rules), along with the leave-one-out residual for every group.

## Primary contrast

**D = R²cv(NO2) − R²cv(PM2.5), anomaly outcome.**

- **Paired bootstrap over groups.** Draw 35 groups with replacement (2,000
  resamples, seed 0). The same draw is used for both pollutants. Rerun the
  whole leave-one-group-out procedure in each resample, with every copy of a
  drawn group held out together, so no group predicts itself. 95% percentile
  interval, and the share of resamples with D ≤ 0 as a one-sided p.
- The same contrast on the level outcome is reported as secondary.

## Families and multiple testing

- **Primary:** D on the anomaly. One test, no correction.
- **Family A (anomaly):** R²cv permutation p for NO2 and for PM2.5, BH across
  the two.
- **Family B (level):** the same two, BH across the two, plus D on the level.
- **Per-feature, as in 03:** 03's `perm_test` Spearman (two-sided) for each
  of the 2 predictors × 2 pollutants × 2 outcomes, BH across all 8. These are
  supporting only and do not enter the reading.
- Everything else is exploratory, labelled so, and never promoted. That
  includes all six features, the four non-primary features, the
  station-level (pseudo-replicated) version, standardised anomalies and any
  tree model.

## Controls reported next to every test (from 03)

- **Split-half reliability ceiling.** Each outcome recomputed on odd and on
  even IST days, Pearson r across groups. An outcome that does not repeat
  between halves cannot be predicted by anything. A null against a low
  ceiling says little, and the report says so.
- **Mantel check.** Correlation between |difference in outcome| and distance
  across group pairs, with the label-permutation Mantel null from 03 (2,000
  permutations). If the outcome is spatially autocorrelated, the features can
  pick up geography for free, and the permutation p-values may be
  anticonservative. In that case marginal results are read as null.

## Sensitivities

Each is re-run in full: both pollutants, both outcomes, skill, permutation
and contrast. The same monitors are dropped for both pollutants.

- **S1, near-flat NO2 sensors removed:** Powai (6956), Kurla (6945), Mulund
  West (12040), Mahape (6943). These are the four with hourly CV under half the
  network median in the NO2 control. 35 monitors, **31 groups**.
- **S2, odd NO2/NOx ratio removed.** These are the 10 monitors whose median
  NOx × 1000 ÷ NO2, over hours where both are reported, is ≥ 1.5. The other
  29 sit near 0.5–1.4. Nothing falls between 1.41 (Worli 3409323) and 1.70
  (Powai). The 10: Powai (6956), Borivali East (6965), Sion (6967), Mahape
  (6943), Airport T2 (6948), Kurla (6945), Mulund West (12040), Khadakpada
  (7752), Bandra Kurla Complex (3409328), Deonar (3409329). Medians run
  1.70–4.22, and 8 of the 10 are the legacy-ID monitors, which matches the
  NO2 control's description of this group. The rule was written down only
  now. The ten were identified after the NO2 control ran, from the ratio
  alone, without any land use. 29 monitors, **27 groups**. All four
  near-flat sensors from S1 are inside this list.
- **S3, the 03 window:** the matched panel restricted to the common window,
  2026-03-10 to 2026-08-27, which is where every PM2.5 land-use result in 03
  comes from.

**A sensitivity never replaces the primary.** If one changes the reading,
both readings are reported. S1 and S2 drop monitors chosen after earlier NO2
results were seen, so a different reading under them is *suggestive* of a
data problem, not proof.

## Reading rules

Notation: R²N, R²P = anomaly R²cv for NO2 and PM2.5. pN, pP = their BH-adjusted
permutation p (family A). D and its 95% interval as above. "Right sign" means
the full-sample fit for that pollutant gives at least one predictor the
expected sign and gives neither predictor a wrong-signed coefficient with
|t| > 2.

| reading | condition | meaning |
|---|---|---|
| **L1: land use explains NO2 more** | D interval lower bound > 0, and pN < 0.05, R²N > 0, right sign | The expert's premise holds here. NO2's between-monitor differences follow roads and PM2.5's do not, so the NO2 control's failure was about spatial correlation at 1–40 km, not about NO2 having no local structure. |
| **L2: NO2 signal, contrast unresolved** | pN < 0.05, R²N > 0, right sign, but D interval includes 0 | NO2 follows roads, but this network cannot show that it does so more than PM2.5. |
| **L3: no detectable land-use signal** | pN ≥ 0.05 and pP ≥ 0.05 | Neither pollutant's long-run pattern is predicted by roads at a size 35 groups can see. Consistent with an NO2 data problem but does not show one, because power is low (below). |
| **L4: PM2.5 more than NO2** | D interval upper bound < 0, or pP < 0.05 with pN ≥ 0.05 | The opposite of what LUR experience predicts. Read as evidence that the NO2 data, not the physics, hides NO2's local structure. |
| **L5: signal, but not the road signal** | pN < 0.05 (or pP < 0.05) without right sign | Skill exists but runs against the physics. Reported as unexplained, not as support for anything. |

If none of the rows fits exactly (e.g. pP < 0.05 and pN < 0.05 with D
spanning 0), the result is described cell by cell without forcing a label.

## Prediction (written so it can be wrong)

**Primary: L3.** R²N < 0.15, pN ≥ 0.05, and the D interval includes 0. NO2
failed every spatial measure in the control, four sensors are near-flat, and
some monitors' NO2 scale is in doubt. All of that adds between-monitor noise
of exactly the kind a long-run mean picks up.

**S2:** removing the 10 odd-ratio monitors raises R²N by at least 0.10 over
the primary, and moves R²P by less than that.

The primary prediction is wrong if the reading is L1, L2, L4 or L5. The S2
prediction is wrong if R²N rises by less than 0.10, or if R²P rises by at
least as much.

## Power, stated honestly

Simulated on synthetic outcomes at the real 35-group feature values: true
signal split equally across the two predictors in their expected directions,
Gaussian noise, the exact LOGO-R²cv statistic.

| true R² | median R²cv observed | power, skill test (p < 0.05) |
|---:|---:|---:|
| 0.0 | −0.07 | 0.05 (size) |
| 0.1 | +0.03 | 0.38 |
| 0.2 | +0.13 | 0.69 |
| 0.3 | +0.25 | 0.89 |
| 0.4 | +0.37 | 0.99 |

**Contrast (D interval above 0)**, with PM2.5's true R² = 0 and independent
noise: power 0.09, 0.26, 0.40, 0.70, 0.91 at NO2 true R² of 0.2, 0.3, 0.4,
0.5, 0.6. Real paired data share hours and monitors, which should help a
little, but the order of magnitude stands.

For the per-feature Spearman tests, 03's own table applies: power 0.37 / 0.62 /
0.84 at ρ = 0.30 / 0.40 / 0.50.

**So 35 groups can only detect a strong effect.** One pollutant's road
signal is detected reliably only at a true R² around 0.25–0.3 or more. The NO2-over-PM2.5 contrast needs
NO2 to explain about half of the between-group variance while PM2.5 explains
none. That is at the top of what European NO2 LUR studies report, with far
denser networks and site-specific predictors. **A null contrast is the
expected outcome even if the expert is right**, and it will be reported as
"not resolvable at this n", not as "land use does not explain NO2". With 31
and 27 groups, S1 and S2 have less power still.

## Reporting rules

- Every number above is reported, including nulls, with n (monitors and
  groups) and the relevant power line.
- Nothing is dropped, renamed or re-specified after the result is seen. A
  forced deviation (e.g. a monitor under the hour threshold) is written into
  the results as a deviation.
- Anything not listed here is labelled **exploratory**.
