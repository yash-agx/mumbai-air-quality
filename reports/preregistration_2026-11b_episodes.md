# Pre-registration — do the models beat the baseline during episodes?

Written **before the analysis was run**, committed on its own so the order is
checkable. Second pre-registration of this round; the first is
`preregistration_2026-11.md`.

**Question.** Across all hours, the spatial models barely beat the city-average
baseline: leave-one-group-out CV puts IDW about 0.16 µg/m³ of RMSE ahead of it,
under 1% of a 21.4 µg/m³ baseline. Does that change during extreme pollution —
the hours anyone actually cares about? A method that is useless on a calm
Tuesday and useful during a Diwali night or a December inversion would be worth
knowing about, and the all-hours average would hide it.

**Prediction being tested.** During episodes, the spatial field is less uniform
than usual: a city-wide average cannot represent a plume over one side of the
city, while a distance-weighted estimate can. So the models' advantage over the
baseline should be **larger during episodes than overall**.

Seed 0. α = 0.05. No correction across the three episode families — they are
reported as three separate questions, not one family, and none is promoted over
the others.

---

## Predictions come from the existing grouped spatial hold-out

The same leave-one-`cv_group`-out scheme as the shipped CV: 35 folds, each
model trained on the other groups and scored on monitors it never saw, with the
IDW power and the variogram fitted inside the fold. Nothing about the episode
definitions touches the training.

Predictions are saved per station-hour for all five rows — `regional_mean`
(the baseline), `idw`, `kriging`, `trees`, `trees+idw` — and the episode
subsets are taken afterwards. Episodes never influence what the models saw.

Not time-blocked. The previous round already measured that axis: holding out
whole months moved IDW's gap by 0.020 µg/m³ (p = 0.068), so the spatial
hold-out is the right scheme here and the time axis is a known small effect.

---

## The unit of analysis is the episode, not the hour

123 of the 12,216 usable hours are top-1% hours, but they are not 123
independent facts: they arrive in **46 contiguous runs**, and the runs
themselves cluster into a handful of months. One synoptic weather state covers
the whole city for a day or more, so an hour inside an episode tells you
little that the hour before it did not.

**An episode is a contiguous run of qualifying hours, merging runs separated by
6 hours or less.** All uncertainty is computed over episodes:

- **Primary uncertainty:** cluster bootstrap resampling whole episodes with
  replacement, 2,000 resamples, seed 0, reported as a 95% percentile interval
  on the gap.
- **Secondary:** Wilcoxon signed-rank across per-episode gaps, where n allows.
- **Sensitivity for (a):** the same test with the calendar **month** as the
  cluster, which is the more conservative unit if synoptic states persist
  longer than the 6-hour bridge.

### Effective n, and where it is fatal

| family | hours | episodes | months | notes |
|---|---:|---:|---:|---|
| top 1% (≥ 85 µg/m³) | 123 | 46 | 8 | |
| top 5% (≥ 70 µg/m³) | 611 | 91 | 9 | |
| Diwali 2025 nights | ~114 | **1** | 1 | one festival, one year |
| winter spells (primary) | — | **6** | 2 | all in **one** winter |

Simulated Wilcoxon power (α = 0.05), per-episode gap:

| n episodes | d = 0.3 | d = 0.5 | d = 0.8 | d = 1.0 |
|---:|---:|---:|---:|---:|
| 4 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.06 | 0.12 | 0.23 | 0.35 |
| 8 | 0.10 | 0.20 | 0.43 | 0.60 |
| 46 | 0.51 | 0.91 | 1.00 | 1.00 |
| 91 | 0.79 | 1.00 | 1.00 | 1.00 |

**At n = 4 the power is 0.00 at every effect size — not rounding, arithmetic.**
The smallest two-sided p a signed-rank test can return at n = 4 is 0.125, so a
four-episode test cannot produce a significant result however large the effect
is. This is why the winter definition below is the one that yields six spells,
and why the winter result is reported as an interval rather than a verdict.

**Diwali is n = 1.** No p-value will be computed for it. It is described, and
the description is the deliverable.

---

## (a) The most polluted hours citywide

- Rank hours by the **mean of observed PM2.5 across reporting monitors**,
  requiring ≥ 10 monitors reporting for an hour to be rankable (12,216 of
  12,294 hours qualify).
- **Top 1%** (≥ 85 µg/m³) and **top 5%** (≥ 70 µg/m³), the thresholds being
  whatever those percentiles turn out to be on that series.
- Note what this subset is: it is defined by the *baseline's own input*. Hours
  are selected because the city-wide average is high, which is the quantity the
  baseline predicts with. That makes it a fair test of "does the model beat the
  baseline when it matters" and **not** a test of "does the model detect
  episodes" — which this design cannot answer and will not claim to.

**Supports:** the gap (model RMSE − baseline RMSE) is negative and its 95%
bootstrap interval excludes zero, for IDW, and is **more negative than the
all-hours gap** with the difference's interval also excluding zero.

**Rejects / null:** the interval includes zero, or the gap is positive
(the models do worse than the baseline during episodes).

## (b) Diwali 2025 nights

- Lakshmi Puja fell on **20 October 2025**; the festival's firework nights span
  the days around it. The window is fixed as **18–22 October 2025 inclusive,
  18:00–02:00 IST**, which covers the date and its one-day regional ambiguity
  on both sides.
- Reported: baseline and model RMSE/MAE over the window, per-night figures, and
  how the window's citywide levels compare with the rest of the record.
- **n = 1 episode. No test, no p-value, no claim of significance.** If the
  numbers look dramatic they are still one festival in one city in one year.

## (c) Winter inversion spells

- **Primary definition, fixed here:** within December–January, a run of ≥ 48
  consecutive hours in which the **24-hour rolling citywide mean is ≥ 50
  µg/m³**. On this record that yields **6 spells**.
- **Sensitivity:** the same with a ≥ 60 µg/m³ threshold (4 spells) — reported,
  but with the n = 4 arithmetic above stated next to it.
- The thresholds were chosen knowing the distribution of citywide
  concentrations (median 23, p90 60, Dec–Jan rolling-24h max 75 µg/m³) and
  **not** knowing any model error. The independence that matters — the
  threshold is uncorrelated with which model wins — holds.
- **The honest n is one winter.** The record covers December 2025 and January
  2026 only. Six spells drawn from a single season are six correlated samples
  of one winter's weather, not six independent winters. Whatever the interval
  says, the result does not generalise to "winters".

**Supports:** gap negative with a bootstrap interval excluding zero, in both
threshold variants.

**Rejects / null:** interval includes zero. Given n, this is the expected
outcome even if the effect is real, and the report will say so.

---

## Reporting rules

- Every family above is reported with: RMSE and MAE for all five rows, the gap
  to the baseline, a 95% bootstrap interval over episodes, the number of
  episodes, hours and station-hours, and the all-hours comparison.
- Nulls are reported in the same detail as anything else.
- The largest effect each n could not have detected is stated alongside every
  null.
- Anything not specified here is labelled exploratory and is never promoted.
