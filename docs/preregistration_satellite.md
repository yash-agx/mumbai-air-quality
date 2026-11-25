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

---

## Addendum 1 — 2026-11-25 (data fixed; no test run)

Written after the monitor data were assembled and **before
`scripts/15_satellite_vs_monitors.py` exists or any test has been run.**
Everything below is availability only. Nothing above this addendum is changed.

### Source and window

**Option 1: 2024 hourly PM2.5 from the CPCB CCR portal, against the 2024
satellite file.** OpenAQ has no 2024 PM2.5 for any panel location (see the
top of this file), and CCR does hold 2024 records for most in-box stations.
The window is **IST calendar year 2024, 366 days.**

- **Files:** one manual CCR export per station, 1-hour averaging, in
  `data/raw/cpcb_2024/` (gitignored, like all raw data). The columns used are
  `Timestamp` and `PM2.5 (µg/m³)`.
- **Loader:** `data/load_cpcb_2024.py`. Each file is matched to a panel
  station by the name in its filename. Panel ids, coordinates and
  `cv_group`s are used, since the exports carry no coordinates. Rows go
  through `clean()` from `data/clean.py` unchanged, then
  `data/fetch_2024.py::daily_means` (IST days, ≥ 18 cleaned hours). The
  output is `data/interim/monitors_2024_daily.parquet`.
- **Timestamps:** CCR stamps whole IST hours, taken as the **start** of each
  hour. `clean()` floors timestamps to the UTC hour, which moves IST-hour
  readings 30 minutes earlier. The same shift applies to every station, so
  each rule in `clean()` sees the same hours. `daily_means` adds the
  30 minutes back when it assigns IST days, so each IST day holds hours
  00:00–23:00.
- **Duplicate guard:** the loader drops every file whose PM2.5 series is
  identical to another file's. The first exports for Khindipada-Bhandup
  West (CCR 5397) and Chakala-Andheri East (CCR 5399) were byte-identical.
  Both were downloaded again. The new 5397 export is a different series, and
  the new 5399 export is identical to the earlier one. So the earlier 5399
  file was genuine and the earlier 5397 file was the copy. The final files
  are all distinct.
- **Values seen:** while checking the column format, the first four hourly
  PM2.5 values of two files (Kurla, Powai; 2024-01-01 00:00–03:00) were
  displayed. No other value, summary or distribution has been looked at, for
  either dataset.

### Which stations

Of the 31 in-box panel monitors:

- **21 have usable 2024 data**, in **19 `cv_group`s**.
- **9 have no 2024 data on CCR:** CSIA T2, Mahape, Bhayandar West, CBD
  Belapur, Kasarvadavali, Kopripada-Vashi, Pimpleshwar Mandir, Sanpada and
  Upvan Fort.
- **Bandra Kurla Complex - IITM (3409328) has a CCR file with 0 PM2.5
  hours.** Its group, `bandra_east`, is represented by Kherwadi_Bandra East
  alone.

The 8 panel monitors outside the satellite box were not downloaded.

Cleaning (`clean()`, unchanged): 193,248 rows in, 156,059 out.
- −36,673 null
- −1 saturation
- −114 lone spikes
- −401 stuck-run hours
- 0 QA-flagged (CCR exports carry no flag), 0 out of range, 0 duplicate hours

### Coverage per station

Days = IST days with ≥ 18 cleaned hours, of 366. Match distance = monitor to
the centre of its nearest land cell.

| station | id | group | days | first – last | cleaned hours | match (m) |
|---|---|---|---:|---|---:|---:|
| Byculla - BMC | 3409510 | 3409510 | 349 | 01-01 – 12-31 | 8,456 | 423 |
| Kherwadi_Bandra East - MPCB | 3409486 | bandra_east | 348 | 01-02 – 12-31 | 8,501 | 582 |
| Ghatkopar - BMC | 3409514 | 3409514 | 343 | 01-01 – 12-31 | 8,299 | 423 |
| Bandra Kurla Complex - MPCB | 3409482 | 3409482 | 340 | 01-01 – 12-31 | 8,368 | 505 |
| Powai - MPCB | 6956 | 6956 | 337 | 01-01 – 12-31 | 8,396 | 589 |
| Mulund West - MPCB | 12040 | 12040 | 336 | 01-01 – 12-31 | 8,169 | 591 |
| Chembur - MPCB | 3409483 | 3409483 | 336 | 01-01 – 12-31 | 8,274 | 621 |
| Sion - MPCB | 6967 | 6967 | 334 | 01-01 – 12-31 | 8,215 | 587 |
| Mindspace-Malad West - MPCB | 3409479 | 3409479 | 333 | 01-01 – 12-31 | 8,066 | 241 |
| Khindipada-Bhandup West - IITM | 12039 | 12039 | 331 | 01-01 – 12-31 | 8,140 | 520 |
| Kandivali West - BMC | 3409512 | 3409512 | 326 | 01-01 – 12-31 | 8,019 | 495 |
| Borivali East - MPCB | 6965 | borivali_east | 326 | 01-01 – 12-30 | 8,097 | 540 |
| Worli - MPCB | 3409323 | worli | 319 | 01-01 – 12-30 | 7,963 | 499 |
| Kurla - MPCB | 6945 | 6945 | 307 | 01-01 – 12-30 | 7,948 | 430 |
| Sewri - BMC | 3409513 | 3409513 | 301 | 01-01 – 12-31 | 7,583 | 344 |
| Borivali East - IITM | 11606 | borivali_east | 299 | 01-01 – 12-31 | 7,387 | 290 |
| Malad West - IITM | 11611 | 11611 | 242 | 01-01 – 12-31 | 5,974 | 388 |
| Shivaji Nagar - BMC | 3409511 | 3409511 | 233 | 01-01 – 12-31 | 5,854 | 357 |
| Deonar - IITM | 3409329 | 3409329 | 202 | 01-23 – 12-27 | 5,175 | 321 |
| Siddharth Nagar-Worli - IITM | 6959 | worli | 193 | 01-01 – 12-31 | 5,098 | 632 |
| Chakala-Andheri East - IITM | 12024 | 12024 | 166 | 01-01 – 07-07 | 4,077 | 121 |

Total: 6,301 station-days. Siddharth Nagar-Worli's nearest grid cell is sea
(see above), so its match is the nearest land cell, 632 m away. Chakala-Andheri
East has no valid day after 2024-07-07, so it has no data for the winter
season.

### Coverage per group and the confirmatory minimum

A group-day exists if any member monitor has a valid day.

| group | monitors | days |
|---|---:|---:|
| borivali_east | 2 | 363 |
| worli | 2 | 355 |
| 3409510 | 1 | 349 |
| bandra_east | 1 | 348 |
| 3409514 | 1 | 343 |
| 3409482 | 1 | 340 |
| 6956 | 1 | 337 |
| 12040 | 1 | 336 |
| 3409483 | 1 | 336 |
| 6967 | 1 | 334 |
| 3409479 | 1 | 333 |
| 12039 | 1 | 331 |
| 3409512 | 1 | 326 |
| 6945 | 1 | 307 |
| 3409513 | 1 | 301 |
| 11611 | 1 | 242 |
| 3409511 | 1 | 233 |
| 3409329 | 1 | 202 |
| 12024 | 1 | 166 |

- **Groups per day:** median 16, range 12–19.
- **Kept days (≥ 10 groups):** **366 of 366.** That is 5,882 group-days,
  with every month complete.
- **Groups with ≥ 60 kept days:** **19** (minimum 15).
- **Kept days:** 366 (minimum 120).
- **The confirmatory minimum is met**, so H1–H3 are scored as written.

With every day kept, the month bootstrap resamples all 12 calendar months of
2024.

### Training stations of the satellite product: unresolved

The satellite files (`Rf2_PM2.5_2024-MM-DD ALOK.nc`) carry no global or
variable metadata beyond fill values. No published description of this
product, its predictors, its training stations or its training years was
found. **Whether these 21 monitors' 2024 records were used to train or
calibrate the field is therefore unknown.** Under the caveat fixed above,
an H3 pass is described as **consistency between the two datasets, not
independent validation**. The same applies to any H3b agreement. If
documentation turns up later, what it says is added as a further addendum,
and the scoring rules do not change.
