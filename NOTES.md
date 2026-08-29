# Project Notes — Mumbai Air Quality Dashboard

Deferred ideas, open questions, and decisions made. Not part of the current build.

---

## Future upgrade: multi-city training

Add stations from other Indian cities to improve land-use feature learning.

- **Pune** is the cheapest option — same state, similar climate, ~10 stations
- **Delhi / Bangalore / Chennai** for real scale (300+ stations total)

**Critical:** train on deviation from each city's mean, not raw PM2.5. Delhi runs 3–4x Mumbai. Raw values teach the model city identity, not road effects.

**What it fixes:** more examples for learning "road density → higher PM2.5". With Mumbai alone there are only ~39 stations to learn land-use relationships from, which is thin.

**What it doesn't fix:** Mumbai's sensor sparsity is unchanged. Prediction in Andheri still interpolates from Mumbai stations only — Delhi's readings say nothing about today's air in Andheri.

Do Mumbai-only end-to-end first. This is an upgrade, not a prerequisite.

---

## Headline result: spatial correlation dies below network spacing

**This is the finding for the model card.** Take each hour's readings, subtract that hour's city-wide mean and divide by its spread — what is left is the purely spatial pattern, with the shared "everyone is high tonight" signal removed. That residual field has almost no spatial structure at the distances between our stations.

Empirical semivariogram of the standardised field (semivariance ~1.0 means two stations are no more alike than two random ones):

| separation | pairs | semivariance |
|---|---|---|
| ~1.2 km | 10 | **0.96** |
| 3.7 km | 38 | 1.06 |
| 7.7 km | 99 | 1.05 |
| 12.6 km | 130 | 1.09 |
| 17.7 km | 114 | 1.08 |
| 24.4 km | 217 | 1.09 |
| 35.7 km | 133 | 1.07 |

Flat beyond ~2 km. Even at the closest separation only about a tenth of the variance is spatially structured, and past 2 km none of it is. The same shape appears in the raw, artefact-screened and log-transformed versions, so it is a property of the field rather than of the outliers.

Two caveats on the table. The nearest bin rests on only 10 pairs, and those are the near-coincident stations that CV holds out together anyway — so the one bin showing structure is the one we deliberately refuse to exploit. And the fitted exponential variogram is correspondingly unstable across folds: median nugget 0.95 with the range poorly identified (2 km to the 300 km bound). That instability *is* the result — there is no well-defined range to find because there is almost no structure to fit.

**Why it matters:** the closest station pairs we have (outside the co-located CV groups, which are held out together) are ~1 km apart, and most pairs are 5–40 km. The correlation is gone before the second-nearest station. There is very little for any interpolation method to exploit, which caps how far any model can get beyond "today the city is at X".

**What it predicts, and did:** IDW, kriging and a plain city-wide average land within 0.5% of each other on RMSE (21.12 / 21.17 / 21.22 µg/m³). Kriging cannot beat IDW because the variogram it fits is nearly pure nugget. The dashboard's surface is honestly "the city average, tilted slightly by land use" — say so on the model card rather than implying a resolved pollution field.

**Consequence for the map:** do not render the surface at a resolution that implies more spatial detail than ~2–3 km of real correlation. A smooth high-resolution raster would look far more certain than the data supports.

---

## For the README: the monitor network is sited unlike the city it measures

81% of the Mumbai bounding box is masked as outside the training range. Worth stating plainly on the README, because the reason is not what it first looks like.

**The mask is mostly emptiness, not density.** Cells fail the band overwhelmingly by falling *below* it:

| feature | cells below the training minimum |
|---|---|
| building_density_500m | 75.1% |
| road_density_500m | 43.3% |
| road_density_1km | 39.8% |

The bounding box is a rectangle over a coastal city: Arabian Sea, Sanjay Gandhi National Park, Thane creek, rural land east of Kalyan. 37% of cells contain no mapped roads *and* no mapped buildings at all. Masking those is correct and uninteresting.

**The real finding is the direction of the siting bias.** Monitors sit in places notably denser and busier than the city's typical inhabited ground:

| | stations | typical inhabited cell |
|---|---|---|
| building density (median) | 244.5 /km² | 90.4 /km² |
| road density 500 m (median) | 18.3 km/km² | 7.6 km/km² |

Stations sit at roughly 2.7× the building density of ordinary inhabited land, and the densest station lands at the 98th percentile of inhabited cells. The other half of the mask says the same thing from the other side: 51% of cells are farther from a major road than any station is, and 50% are farther from industry. Monitoring sites cluster near roads and industrial land, which is presumably deliberate — that is where the regulatory interest is.

**Consequence to state honestly.** The model is trained on busy, road-adjacent, industry-adjacent locations and asked to predict everywhere. Where it extrapolates, it is mostly extrapolating *downward* — into quieter, greener, lower-traffic ground it has never seen. Any claim about a calm residential pocket rests on stations that are systematically busier than it. Nothing in the CV score speaks to that, because every held-out station is itself one of these busy sites.

**What is not a problem:** only 1.6% of inhabited cells (23 of 1,410) are denser than the densest station. Mumbai's dense neighbourhoods are, with few exceptions, inside the training range. The gap is at the quiet end, not the crowded end.

---

## Decisions already made

**18-month window over 5-year window.** 40 live stations with ~18 months of history beats 14 stations with 5 years. Spatial density matters more than history depth for interpolation, and leave-one-station-out CV is only meaningful with enough stations. Still covers two winters, which is Mumbai's peak pollution season.

**OpenAQ over scraping CPCB directly.** CPCB's portal requires manual clicking with per-request limits. OpenAQ aggregates the same official data through an API.

**Spatial cross-validation, not random k-fold.** Random splits put readings from the same station in both train and test, which leaks spatial information and produces a fake-good score. Hold out whole stations.

**Near-coincident stations grouped for CV.** Stations under ~1km apart must be held out together, or the model predicts a "hidden" station by peeking at its neighbour across the street.

Groups:
- `bandra_east` — Kherwadi_Bandra East [3409486] + Bandra Kurla Complex [3409328], ~70m apart
- `borivali_east` — Borivali East [11606] + Borivali East [6965], ~600m apart
- Sidhi Vinayak Nagar [3409484] + Vithalwadi [6258871], 676m apart
- Siddharth Nagar-Worli [6959] + Worli [3409323], 729m apart

**Station 8039 ("Mumbai") dropped.** 2.2% coverage, no proper site name, 12,764-hour gap.

**Artefact screen, evidence-based rather than a threshold.** `VALID_RANGE`'s 1000 µg/m³ ceiling only catches the physically absurd; sensor faults sit well inside it and dominated every squared error (RMSE ~46 µg/m³ against a median reading of 23.5). Two narrower rules, each tied to something observable:

- *Saturation* — drop readings of exactly 985.0. That was the exact maximum at five unrelated stations; independent sensors do not agree on a maximum to one decimal unless it is a shared ceiling.
- *Lone spike* — drop a reading over 150 µg/m³ when the median across every other reporting station that hour is under 50. Episodes are regional; one station at 690 with the rest of the city at 23 is a sensor.

Corroboration does the work, so a genuinely high reading the city agrees with survives at any magnitude. Confirmed by the kept highs clustering in Oct–Feb with none in the monsoon.

---

## Open questions

**Do engineered land-use features actually beat kriging?** ~~Unknown~~ **Answered: no, and neither beats IDW.** Under leave-one-group-out CV, a properly tuned IDW baseline wins; kriging and gradient-boosted trees on the OSM features both lose to it, and every method is within ~1.7% of simply averaging the reporting stations. The variogram above explains why. The modest honest result is the deliverable — this is it.

Two traps found while getting there, worth remembering:
- *Tune the baseline or the comparison is worthless.* The conventional 1/d² is not IDW's best power here; with p=2 kriging appeared to win by 2.4%, and with the power chosen per fold IDW wins instead.
- *Tuning needs the group structure too.* Selecting the IDW power by predicting each training station from the others let `bandra_east`'s 70 m partner leak in. The inner split has to drop whole groups, same as the outer one.

**Settled: the extrapolation band is min–max, not p01–p99.** `predict_surface(..., band=("p01","p99"))` still switches it.

With only 39 stations, p01 sits just above the minimum, so a station holding the minimum on any one of six features falls outside its own band. Nine did — the airport, Powai, Worli, BKC, Kasarvadavali, Kalamboli and three others — meaning the mask shaded locations where we hold ground truth. min–max excludes none of them and costs 1.4 points of grid coverage (81.3% masked against 82.7%). The case for p01–p99 is robustness to one freak station, which is worth something at a few hundred stations and not much at thirty-nine.

**OPEN: 113 readings above 500 µg/m³ survive the artefact screen.** Deliberately left as-is for now; revisit if the error bars look wrong at the top of the range.

They are kept because the city median cleared 50 µg/m³ that hour, which is the rule as specified. Some are clearly genuine — Mulund West at 1000 while the city median was 211, with Upvan Fort at 989 in the same hour, a real October 2025 episode. Others clear the bar only narrowly while sitting 15–18× above the city median:

| station | value | city median that hour |
|---|---|---|
| Kopripada-Vashi | 979 | 53.8 |
| Kopripada-Vashi | 964 | 51.4 |
| Sion | 891 | 57.0 |
| Shivaji Nagar | 856 | 61.4 |

The current rule is a *floor* on the regional median. A *ratio* test — flag when a reading exceeds, say, 8× the same-hour median across other stations — would catch these while still keeping the genuine episodes, where the whole basin rises together. Worth doing if the top of the uncertainty band looks unreliable, since these sit in the highest-prediction bin where sigma is already largest (~45 µg/m³).

**Stuck sensors.** Several stations repeat one value for 24+ hours (Khadakpada 456h, Mulund West 211h). Currently masked rather than dropped. Worth checking whether masking changes results materially.

**Three stations came online March 2026.** Kalu Nagar, Vithalwadi, Chinchpada — ~25–28% coverage each. Fine for recent predictions, useless for training on 2025 data. Consider whether to include them at all.
