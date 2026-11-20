# Pre-registration — NO2 as a positive control

Written **before any NO2 value was looked at** and committed on its own, so the
order can be checked. What has been computed so far is availability only:
which sensors exist, their unit labels and date ranges, and how many hours
each reports (a non-null, unflagged reading). No NO2 or NOx concentration,
summary statistic or distribution has been printed or inspected.

**Question.** Every PM2.5 result in this project says the same thing: once the
citywide mean is removed, there is little spatial structure at this network's
spacing. Hourly semivariance is ~1 from ~2 km out. IDW beats the city average
by 0.5% of RMSE. b2 reliability is 0.14 at 1 h and 0.26 at 1 week. An expert
pointed out that this is what PM2.5 is expected to do, because it is
dominated by regional, long-lived particles. NO2 and NOx come mainly from
local combustion (traffic, industry) and live for hours, not days, so they
should vary much more between places. If the same method, run on the same
monitors and hours, finds clearly more structure in NO2, then the method can
see structure when it is there, and the PM2.5 null is a property of PM2.5 and
not a blind spot of the method.

Seed 0. α = 0.05. 1,000 bootstrap resamples, 999 permutations.

---

## Data

- **Source:** OpenAQ v3, the same API, locations and period as the PM2.5
  history. Pulled by `data/fetch_no2.py` (reusing `data/fetch.py`'s request
  and paging code) into `data/raw/no2_raw.parquet`, with a per-sensor cache
  in `data/raw/_parts_no2/`. Both are under `data/raw/`, which is
  gitignored, as the PM2.5 and wind raw pulls are. The fetch script is
  committed with the analysis, not with this file.
- **Window:** the cleaned PM2.5 panel's span, 2025-02-27 21:00 to
  2026-08-27 01:00 UTC (13,085 hours).

### Availability (counts only)

**All 39 monitors report NO2, and all 39 report NOx.** Each has exactly one
NO2 sensor and one NOx sensor active in the window. Thirteen monitors also
have an older NO2 sensor labelled µg/m³, but every one of those ended
before the window and returns no hours, so none enters the data. **Every
in-window sensor is labelled ppb.**

Reported NO2 hours per monitor, inside the window (before cleaning):

| | NO2 | NOx |
|---|---:|---:|
| monitors reporting | 39 | 39 |
| median hours per monitor | 11,024 (84.2%) | 11,119 (85.0%) |
| range | 3,576 (27.3%) – 11,796 (90.1%) | 3,584 – 11,781 |
| monitors under 50% | 4 | 4 |
| station-hours, total | 384,289 | 393,986 |
| station-hours also carrying a cleaned PM2.5 reading | 364,177 | 370,795 |

The four under 50% are the same four short-record monitors that are short in
PM2.5 (Pimpleshwar Mandir, Kalu Nagar, Vithalwadi, Chinchpada; 27–30%). The
shortest PM2.5 record kept in every earlier analysis is 3,296 hours (25%).

Windows on the **matched panel** (station-hours with both PM2.5 and NO2), under
the averaging-time coverage rule (75% of a window's hours, ≥ 10 monitors):

| window | windows kept | median monitors / window | median groups / window |
|---|---:|---:|---:|
| 1 h | 12,216 | 30 | 28 |
| 24 h | 503 | 28 | 26 |
| 1 week | 72 | 27 | 25 |

The PM2.5-only analysis kept 12,216 / 503 / 72 windows with 29 / 27 / 27
median groups, so matching costs about one group per window and no windows.

### Station and coverage rule

- **Stations:** all 39 monitors, in the **same 35 `cv_group`s**. No monitor
  is excluded for low NO2 coverage, for the same reason none was excluded for
  PM2.5: the per-window coverage rule already decides when a monitor counts,
  and dropping the four short records would make the panel unlike the
  PM2.5 one.
- **Per-window coverage:** identical to `scripts/11_averaging_times.py`. A
  station-window counts if the monitor has ≥ 75% of the window's nominal
  hours. A window counts if ≥ 10 monitors pass. At 1 h this means the hour
  is used if ≥ 10 monitors report.
- **Matched panel (primary):** a station-hour is used **for both pollutants
  only if both have a cleaned reading**. Coverage is then applied to that
  joint mask, so PM2.5 and NO2 are scored on exactly the same monitors,
  hours and windows.

---

## Cleaning NO2 (and NOx), fixed in advance

The steps of `data/clean.py`, in the same order, adapted where PM2.5's rule
depends on PM2.5's units or physics. The PM2.5 data is the shipped
`pm25_clean.parquet`, unchanged.

1. **QA-flagged rows dropped**, as for PM2.5.
2. **Nulls dropped.**
3. **Range: negative values and values above 1,000 (labelled units) are
   dropped.** PM2.5 drops ≤ 0. For NO2, exactly 0 can be a genuine
   at-detection-limit reading on a clean night, so **zeros are kept** in the
   primary and the count is reported. If zeros are more than 1% of rows, the
   primary results are re-run with zeros dropped and reported as a
   sensitivity.
4. **Units.** Every in-window sensor carries the same label (ppb), so values
   are used as labelled, with no conversion. Every structure measure here
   (standardised anomalies, correlation, semivariance, RMSE gap as % of the
   city-average RMSE) is unchanged by multiplying every monitor by the same
   constant. So whether the label is right or not cannot affect the results,
   as long as it is the same at every monitor. Levels quoted for context are
   given in ppb, with µg/m³ = 1.88 × ppb (25 °C, 1 atm) alongside.
5. **Snap to the hour**, average any duplicate station-hours.
6. **Saturation:** PM2.5's rule was one specific value (985.0) that turned up
   as the exact maximum at five unrelated monitors. The generic version is
   used: **any exact value that is the recorded maximum at ≥ 3 monitors is
   treated as a ceiling and dropped wherever it occurs.** The count and the
   value are reported.
7. **Lone spikes: a temporal rule replaces PM2.5's spatial one. This is the
   one deliberate change.** PM2.5 drops a reading > 150 µg/m³ when the other
   monitors' median that hour is < 50, on the grounds that PM2.5 episodes are
   regional. Using that logic on NO2 would delete exactly the thing under
   test: a local plume that the rest of the city does not see. It would also
   bias the NO2 result toward the PM2.5 null. So a NO2 reading is a lone
   spike if **it exceeds 4× the median of the same monitor's readings in the
   surrounding ±3 hours** (itself excluded; at least 3 of the 6 must be
   present), **and** exceeds that monitor's own 99.9th percentile. It is a
   single-hour jump that its own monitor does not sustain. The count is
   reported. **Sensitivity:** primary results re-run with no spike screen.
8. **Stuck runs:** one value repeated for ≥ 24 unbroken hours is masked,
   exactly as for PM2.5.

NOx goes through the same eight steps independently.

---

## Tests

These mirror the PM2.5 analysis, with the functions reused from
`scripts/11_averaging_times.py` and `model/interpolate.py` and no
re-implementation. The IST hourly grid, window anchoring and per-window
standardisation are the same: anomaly z = (station mean − citywide mean) /
across-station SD.

**(a) Distance pattern, 1 h primary.** Per station pair, spatial
correlation 1 − γ, with γ = ½·mean over shared hours of (zᵢ − zⱼ)². Same
distance bins (same-group row separate; 0–2, 2–5, 5–10, 10–15, 15–20, 20–30,
≥ 30 km). Pairs must share ≥ 3 windows. **Summary:** near − far = mean
(1 − γ) for cross-group pairs < 5 km minus pairs ≥ 15 km (44 and 464 pairs if
every monitor appears, as for PM2.5). Mantel-style permutation p, 999
permutations of station labels. Also reported at 24 h and 1 week.

**(b) Grouped spatial CV, 1 h primary.** Leave-one-`cv_group`-out over the
same 35 groups. City average = mean of the monitors outside the held-out
group. IDW power is re-tuned inside every fold with `select_idw_power`.
**Gap % = (IDW RMSE − city-average RMSE) / city-average RMSE × 100**;
negative means IDW wins. Wilcoxon signed-rank test across per-group gaps.
Also reported at 24 h and 1 week.

**(c) b2 reliability at 1 h, 24 h and 1 week.** Across-group Pearson r
between the anomaly patterns of two windows of the same length, at least 7
days apart, with ≥ 10 groups present in both, averaged over pairs (all of
them, or 50,000 drawn with seed 0).

**Intervals:** cluster bootstrap over **calendar months**, 1,000 resamples,
seed 0, 95% percentile intervals. **The same month resamples are used for
PM2.5 and NO2,** so each NO2 − PM2.5 difference gets a paired interval.

**Every number is reported with** monitors, groups, windows (hours), and
where relevant station pairs, b2 window pairs and CV station-windows.

### Panels

| panel | role |
|---|---|
| **PM2.5 and NO2 on the matched panel** | **primary; all decision rules use these** |
| NO2 on its own availability | secondary, shows what matching cost |
| PM2.5 on its own availability | secondary and a code check: must reproduce `reports/averaging_times.csv` (b2 1 h 0.137, 1 w 0.258; gap % 1 h −0.50, 1 w −3.02) to within 0.005 / 0.05 pp, or the code is fixed before anything is read |
| NOx matched with PM2.5 | secondary positive control, reported in full, not part of the decision rules |

---

## How the results will be read

Three families of structure, each with its own criterion, all on the matched
panel. "D" means NO2 minus PM2.5, with a paired month-bootstrap interval.

- **S, stable between-place differences (b2).** Met if **D b2(1 h) and
  D b2(1 w) both have intervals above 0, and NO2 b2(1 h) ≥ 2 × PM2.5
  b2(1 h).** The ratio floor keeps a significant but trivial difference from
  counting as "clearly more".
- **G, distance pattern (nearby monitors more alike).** Met if **NO2
  near − far at 1 h > 0 with Mantel p < 0.05, and D near − far(1 h) has an
  interval above 0.**
- **P, prediction (IDW beats the city average).** Met if **NO2 gap %(1 h)
  has an interval entirely below 0, D gap %(1 h) has an interval entirely
  below 0, and NO2 gap %(1 h) ≤ −2%**, four times PM2.5's hourly gain.

**"NO2 shows clearly more structure than PM2.5" means S is met and at least
one of G or P is met.**

| reading | rule | what it means for the PM2.5 result |
|---|---|---|
| **R1, full pass** | S, G and P all met | The method detects stable and spatially smooth structure when it exists. The PM2.5 nulls on all three measures are properties of PM2.5, not blind spots of the method. |
| **R3, partial pass** | S met, exactly one of G or P | NO2 shows clearly more structure. The limb that passed is validated. The one that did not is unvalidated for both pollutants and is said so. |
| **R2, stability only** | S met, neither G nor P | The method sees stable between-place differences, so PM2.5's low b2 is real. NO2's structure is not smooth at 1–40 km either, so this cannot show that the distance and IDW nulls for PM2.5 are not a spacing limit. Those nulls stay "no structure this network's spacing can see". This does **not** count as "clearly more structure" overall. |
| **R4, control fails** | S not met | NO2, the pollutant expected to vary most, shows no clearly more stable structure than PM2.5. Either the network or this method cannot resolve local structure at this spacing, or NO2 here is less local than expected. The PM2.5 null then cannot be called "not a blind spot", and the README says so plainly. |
| **Mixed** | G or P met, S not | Reported as mixed, naming exactly which criteria were met. |

### Prediction, stated so it can be wrong

**R2, stability only.** NO2 b2 will be clearly higher than PM2.5's: roughly
0.3–0.5 at 1 h against PM2.5's ~0.14, because a roadside monitor reads high
relative to a background one almost every hour. But NO2 from traffic falls
off over tens to hundreds of metres from a road, far below the 1–40 km
between these monitors. So the predicted distance contrast is not
significant, and IDW's gain is under 2%. In short, NO2 should validate
the stability measure but not the smoothness measures. NOx, which includes
the even shorter-lived fresh NO, is predicted to show higher b2 than NO2 and
no more distance structure.

**What this design cannot answer.** NO2 is a positive control only for
structure that NO2 actually has *at this spacing*. If NO2's spatial
correlation is gone within ~1 km, then a G or P failure says nothing about
whether the method could detect smooth structure on a denser network. It
only shows that this network cannot. A pass on S validates the b2 measure,
not the PM2.5 distance and CV nulls.

---

## Reporting rules

- Nulls are reported in the same detail as positives. The reading is applied
  as written and not rounded to the nearest one.
- Anything not specified above is labelled exploratory.
- Existing results and files are not changed. New work: `data/fetch_no2.py`,
  `scripts/12_no2_positive_control.py`, `reports/no2_positive_control.md`,
  `reports/figures/no2_vs_pm25.png`, `tests/test_no2_positive_control.py`
  and a short README section.
