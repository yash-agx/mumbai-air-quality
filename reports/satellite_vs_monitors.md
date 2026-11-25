# Satellite PM2.5 against the monitor network, 2024

Pre-registered in
[`docs/preregistration_satellite.md`](../docs/preregistration_satellite.md),
with the data fixed in its Addendum 1 before any test was run. Code:
`scripts/15_satellite_vs_monitors.py`, which reuses the b2 day-pair
selection, distance pattern and Mantel test of
`scripts/11_averaging_times.py`. Tests: `tests/test_satellite_vs_monitors.py`.
Full numbers: [`satellite_vs_monitors.csv`](satellite_vs_monitors.csv) (every
metric), [`satellite_vs_monitors_groups.csv`](satellite_vs_monitors_groups.csv)
(per group) and
[`satellite_vs_monitors_distance.csv`](satellite_vs_monitors_distance.csv)
(distance bins).

**Data.**
- Monitors: CPCB CCR hourly PM2.5 for 2024, cleaned with `data/clean.py`
  and reduced to IST daily means (≥ 18 hours).
- Satellite: the IIT Delhi daily 0.01° field, with each monitor matched to
  its nearest land cell (121–632 m).
- Matched panel: **21 monitors in 19 `cv_group`s, 366 days, 5,882
  group-days.** The confirmatory minimum is met, so every hypothesis is
  scored.
- Day alignment: the citywide series correlate at r = 0.937 / **0.986** /
  0.940 at lags −1 / 0 / +1, so lag 0 is best and the day convention holds.

Brackets are 95% month-cluster bootstrap intervals (1,000 resamples, seed 0),
with the same resamples for both datasets.

## Scores

| | rule (pre-registered) | result | score |
|---|---|---|---|
| **H1** satellite has meaningful between-monitor structure | b2 lower bound > 0.30 and A ≥ 3 µg/m³ | b2 0.290 [0.222, 0.428]; A 3.24 µg/m³ | **inconclusive** |
| **H2** citywide signal dominates both datasets | C lower bound > 0.50 for both | monitors 0.730 [0.621, 0.781]; satellite 0.952 [0.920, 0.962] | **supported** |
| H2b (secondary) the same on first differences | same rule | monitors 0.539 [0.479, 0.591]; satellite 0.867 [0.837, 0.885] | inconclusive |
| **H3** satellite tracks each monitor's daily PM2.5 | median r ≥ 0.70 and lower bound ≥ 0.60 | median r 0.919 [0.886, 0.949], 19 groups | **supported, as consistency** |

The rules were applied exactly as written. H1 misses "supported" because its
b2 interval reaches below 0.30, and it misses "counts against" because the
interval also reaches above 0.30 and A clears 3 µg/m³. H3 is read as
**consistency, not validation**. Whether these monitors' 2024 records were
used to train the satellite product is unresolved (Addendum 1), so H3 cannot
show independent skill.

## H1: spatial structure in the satellite field

Anomalies are each day's values minus that day's citywide mean, at the
monitor locations, on the matched panel. b2 is the mean across-group
correlation between two days' anomaly patterns, over 49,607 day pairs at
least 7 days apart.

| | satellite | monitors (reference) |
|---|---:|---:|
| b2 | **0.290** [0.222, 0.428] | 0.213 [0.180, 0.304] |
| A, median daily SD of anomalies (µg/m³) | **3.24** [2.46, 4.99] | 10.26 [7.25, 14.06] |
| near − far of 1 − γ (28 near / 72 far pairs) | 0.402, Mantel p = 0.003 | 0.010, p = 0.436 (exploratory) |

- **Satellite vs monitors:** the satellite pattern is more stable than the
  monitors' (paired difference +0.077 [+0.008, +0.165]). It is about a third
  of their amplitude.
- **Distance:** unlike the monitors, the satellite pattern is clearly smooth
  in distance. Nearby monitor locations are more alike than distant ones.
- **Sensitivity:** over all 1,152 land cells and 366 days, b2 = 0.403
  (50,000 pairs) and A = 3.95 µg/m³.

**Prediction (b2 ≥ 0.6): wrong.** The satellite field does not repeat one
static spatial pattern day after day. Its between-place pattern is only
modestly more stable than the monitors', and small in µg/m³.

## H2: how much of the variation is citywide

C is the share of variance in group-day values that is explained by each
day's citywide mean. Both datasets are scored on the same 5,882 group-days.

| | monitors | satellite | satellite − monitors |
|---|---:|---:|---:|
| C, levels (µg/m³) | **0.730** [0.621, 0.781] | **0.952** [0.920, 0.962] | +0.222 [+0.182, +0.306] |
| C, log values | 0.739 [0.612, 0.787] | 0.959 [0.931, 0.965] | — |
| C, first differences (5,608 rows) | 0.539 [0.479, 0.591] | 0.867 [0.837, 0.885] | +0.329 [+0.288, +0.369] |
| leave-one-out r², median (IQR across groups) | 0.821 (0.697–0.881) | 0.973 (0.967–0.982) | — |

**H2 is supported:** the citywide signal is the larger part of the variation
in both datasets.

**First differences are inconclusive for the monitors.** Day-to-day changes
at a monitor are about half citywide (0.539), and the interval reaches below
0.50. The pre-registered rule "levels pass and differences fail" does not
apply, since the differences are inconclusive rather than failing. So the
statement is: **levels are dominated by the citywide signal. Day-to-day
changes at a monitor are only about half citywide, which this test cannot
place above or below a half.** The satellite is citywide-dominated on both
measures.

**Prediction:** C_mon 0.75–0.90 came in slightly low (0.730), and C_sat
0.85–0.97 was right. C_sat > C_mon was right: the satellite is much smoother
than the monitors. First-difference C_mon 0.5–0.7, possibly inconclusive, was
right.

## H3: does each cell track its monitor?

Per group, the Pearson r between the monitor's daily series and its
satellite cells over matched days:

- median **0.919** [0.886, 0.949];
- range 0.761 (Worli group) to 0.975 (Ghatkopar);
- pooled over all group-days, r = 0.894 [0.837, 0.920].

Per-group RMSE is 8.5–16.5 µg/m³, and mean bias (satellite − monitor) is
−8.5 to +6.4 µg/m³. **Prediction (0.80–0.90): slightly exceeded.**

### H3b (secondary): do the anomalies agree?

| | result | prediction |
|---|---:|---|
| pooled r, satellite vs monitor anomalies | **0.631** [0.512, 0.713] | 0.0–0.3, **wrong** |
| Spearman ρ across groups, mean anomalies | **0.696**, p = 0.001 one-sided (0.002 two-sided), 19 groups | 0.3–0.6, slightly exceeded |

The satellite and the monitors agree on which places run high relative to
the city. **They also agree on day-specific departures:** with each group's
mean anomaly removed, the pooled r is still 0.608 (*exploratory, not
pre-registered*). That was not expected. The monitors' own anomaly pattern
barely repeats from one day to the next (b2 0.21), yet the satellite follows
a large part of each day's departures.

Two explanations fit, and these data cannot tell them apart:
1. both datasets see real, short-lived local variation;
2. the satellite field is trained on, or calibrated to, these monitors' 2024
   readings, so it reproduces their day-specific departures.

The training question is open, so the second explanation cannot be ruled
out. **H3b is therefore not evidence that the satellite resolves real
local structure.**

The satellite's anomalies are compressed relative to the monitors' (mean
group anomaly −5.1 to +3.5 µg/m³ against −10.2 to +10.2). In the
per-group table, this shows as negative bias at sites that run high and
positive bias at sites that run low. That is the pattern expected of a
smoothed field (exploratory).

## What this means

1. **Both datasets agree that Mumbai's daily PM2.5 is mostly a citywide
   signal.** The satellite says so more strongly than the monitors: about
   95% citywide against about 73%. How much of the monitors' extra
   between-place variance is real local signal, and how much is noise that
   the smoother field does not carry, cannot be settled here, because the
   H3b agreement may be circular.
2. **The satellite has distance-smooth structure that the monitors lack**
   (Mantel p 0.003 against 0.44). Its stability against the pre-registered
   0.30 bar is undecided (H1 inconclusive), and its amplitude is about 3
   µg/m³, a third of the monitors'.
3. **The satellite tracks the monitors closely, levels and anomalies
   alike.** Until the product's training data is known, that is consistency,
   not independent confirmation. It should not be used to argue that the
   satellite resolves the local structure the monitor network cannot see.

## Reporting notes

- Nothing above was changed after the results were seen. Every threshold is
  as committed in `0fd7bd9` and `c99b4d5`.
- A, the amplitude, uses the across-group SD with ddof = 0, following
  `scripts/11_averaging_times.py`. The pre-registration did not specify
  ddof.
- The distance pattern is at station level (21 monitors, cross-group pairs
  only), as in script 11. The monitors' distance pattern was not listed in
  the pre-registration and is exploratory. The same goes for the
  group-means-removed anomaly r.
- The H3b permutation p is one-sided (agreement), matching the Mantel
  convention, and the two-sided value is given alongside.
- The two same-group monitor pairs (Borivali East MPCB/IITM, Worli
  MPCB/Siddharth Nagar) have very dissimilar daily anomalies (pooled
  1 − γ = −1.26). That rests on two pairs only and is exploratory.
