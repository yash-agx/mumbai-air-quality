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

## The other half: diurnal timing *is* location-specific, and IDW discards it

The variogram result above is about the hourly **level**. Take the same data and ask about **timing** instead — when in the day each place peaks and troughs — and the answer inverts. Together these two are the finding; neither is complete alone.

**Locations have genuinely different daily rhythms.** Averaging each monitor's readings by hour of day, then removing that site's level and swing so only shape remains:

| comparison | median r | IQR |
|---|---|---|
| Same monitor, independent halves of the record | **0.889** | 0.799–0.956 |
| Two different monitors | **0.504** | 0.259–0.683 |

**The noise ceiling is what makes this comparison valid**, and it is the reason for the odd/even-day split. A bare between-monitor correlation of 0.50 proves nothing on its own — profiles built from finite, gappy records are noisy, and the de-mean-and-rescale step amplifies that noise for low-amplitude sites, so 0.50 could just be measurement error. Splitting each monitor's own record by odd and even days gives two independent estimates of the *same* site's profile. Correlating those fixes how well a profile can be reproduced at all when the underlying shape is identical by construction: r = 0.889. Different monitors reach only 57% of that ceiling. Gap +0.385, Mann-Whitney p = 5.6e-20. The difference is real structure, not sampling noise.

A single common shape explains just **32.9%** of the variance across the 39 sites. Median day–night swing is 10.4 µg/m³ (range 2.6–20.5), so the shape is a substantial share of what a resident actually experiences.

**IDW cannot represent any of it.** An interpolated series is a fixed-weight average over all reporting monitors, so its diurnal shape is the citywide shape whatever cell it is computed for. Measured at each monitor's own location, against the all-monitor average curve:

| | median r with the city curve |
|---|---|
| Measured monitor | **0.758** |
| IDW estimate for the same spot | **0.997** |

The estimate reproduces the city's daily rhythm essentially exactly and the local one not at all. Same mechanism as the flat variogram — with no exploitable spatial correlation, the weights spread wide and the answer converges on the city average — but here the loss is visible as a *shape* being flattened, not just a number being close.

**So the honest statement of the finding is both halves together:** at this network's spacing the hourly level is not spatially predictable, while the diurnal timing is location-specific and reliably measurable — and the interpolation throws the second away along with the first. Anything that wants local timing needs measurement at that location; no amount of interpolation between monitors 1–40 km apart will recover it.

**Cheap to compute, which is worth recording.** One cell's full 12,294-hour interpolated history is a single matrix product against the panel (distances to monitors are fixed), about **7 ms** — so the app's time-patterns section needs no precomputation and no sampling.

---

## Does the diurnal shape track land use? Four pre-specified tests, all null

The previous section establishes that each monitor has its own daily rhythm and that
the rhythm is measured reliably. The obvious next question is *why* — whether a site's
shape follows the land around it. Answer: not detectably, at this network's size.

`scripts/03_diurnal_landuse.py` reproduces everything below. It reuses `diurnal()` and
`shape_of()` verbatim from `app.py` (lines 345, 353), so the summaries are built on
exactly the computation the dashboard reports.

### The primary analysis is the common window; the full record is a sensitivity

**Profiles are built on 2026-03-10 onward — the span over which every monitor reports —
and the full 18-month record is demoted to a sensitivity check.** Three monitors (Kalu
Nagar, Vithalwadi, Chinchpada) came online in March 2026. On the full record their
hour-of-day profiles average a different, more monsoon-weighted slice of the year than
everyone else's, so a seasonal difference enters the analysis wearing a spatial costume.
Restricting to the window over which every monitor reports removes that by construction.

**What makes the confound impossible to leave in place is that two of the four primary
estimates move under the window — one of them changes sign:**

| primary test | full record | common window | |
|---|---|---|---|
| `dist_industrial_m` × balance | −0.084 | **+0.091** | **sign flip** |
| `building_density_500m` × swing | +0.233 | **+0.027** | **collapses to zero** |
| `road_density_500m` × balance | −0.147 | −0.229 | grows |
| `dist_coast_m` × trough | 0.265 | 0.363 | grows |

`dist_industrial_m` does not merely shrink, it reverses: on the full record sites farther
from industry look *less* morning-dominant, on the common window *more*. And
`building_density_500m` × swing — the largest of the three linear estimates on the full
record, and the one with the most plausible mechanism behind it — loses about 88% of its
magnitude and lands on nothing. (It does not survive a gentler window either: a 2026-03-01
start leaves +0.089, keeping 38% of it, where the true common start keeps 12%.) An estimate
that flips direction and an estimate that evaporates once the 8,393 hours not every monitor
covers are dropped are not estimates of a stable spatial relationship. **Neither version
is significant either way, so the conclusion does not turn on this** — but it settles which
version is the honest primary, and it is the reason the swap was made.

**The caveat that has to travel with it: the window was promoted to primary *after* the
full-record run had been seen.** The confound it removes is real, and the argument for it
does not depend on any result — it would have been the right choice made first, and the
script now takes it first. But the *decision* to prefer it was made with the numbers in
view, which is the same forking-paths problem flagged for the trough statistic below. Two
things distinguish it, and they are why the swap is made rather than left as a footnote:
the window is fixed by the data-collection record, which is known independently of any
outcome, and there is only one non-arbitrary choice of it (the day the last monitor came
online), where the trough estimator was picked from a menu of defensible alternatives.
Both versions are reported in full at every step, so a reader who disagrees can take the
other one.

**The window is not free — it costs two-thirds of the record and all of the pollution
season:**

| | full record | common window (primary) |
|---|---|---|
| span | 2025-02-28 → 2026-08-27 | 2026-03-10 → 2026-08-27 |
| hours | 12,294 | 3,901 (32%) |
| months represented | all twelve | Mar–Aug 2026 only |
| Nov–Feb pollution-season hours | — | **0** |
| mean concentration | 30.8 µg/m³ | 21.4 µg/m³ (−31%) |
| median swing amplitude | 10.4 µg/m³ | 8.1 µg/m³ |
| median hours behind one profile | 10,806 | 3,434 |

So the primary analysis describes **the clean half of the year**. Any mechanism that only
bites under winter inversions is outside its reach, and every profile is built on a third
as much data — which shows up directly as lost precision in the reliability table below,
and is the single biggest thing the swap gives away.

**Fixed before any result was seen.** Four primary tests, Benjamini-Hochberg across
those four only:

| feature | summary |
|---|---|
| `road_density_500m` | morning–evening balance |
| `dist_coast_m` | trough hour |
| `building_density_500m` | day–night swing amplitude |
| `dist_industrial_m` | morning–evening balance |

Summary definitions, also fixed in advance:

- **morning–evening balance** — mean of the normalised shape over 06:00–10:00 IST minus
  its mean over 18:00–22:00. Positive means morning-dominant. Median +0.56 (range −2.14
  to +2.23).
- **peak / trough hour** — the phase of a fitted 24-hour harmonic, in continuous hours.
  A raw `argmax` over 24 integers is both discrete and circular-blind.
- **day–night swing amplitude** — max minus min of the *raw* profile, in µg/m³. Included
  deliberately: `shape_of()` divides it out, and it is the quantity most likely to respond
  to ventilation. Median **8.1 µg/m³ in the primary window** (range 3.4–22.6); 10.4 µg/m³
  on the full record, which is the figure recorded in the previous section.
- **bimodality** — the 12-hour harmonic's amplitude relative to the 24-hour one. How
  two-peaked the day is. Median 0.86 (range 0.23–4.58).

**The unit of analysis is the CV group, not the station: n = 35, not 39.** Near-coincident
monitors are not independent observations of a land-use relationship — `bandra_east`'s two
sit 70 m apart and share their feature values to three significant figures. Counting them
twice inflates n and understates every p-value. Circular summaries are averaged circularly
within a group. Station-level numbers are reported as a sensitivity and are not the result.

Tests are permutation-based (20,000 draws, seed 0): Spearman ρ for the linear summaries,
Mardia's rank-based circular–linear R for the circular ones. n is small and the distance
features are heavily skewed, so an asymptotic null would not be trustworthy.

### Control: shape similarity *does* decay with distance — where on the full record it did not

This is the one place where the swap changes a conclusion rather than a number, so it gets
stated rather than absorbed.

If diurnal shape is spatially autocorrelated, then any feature that is *itself* spatially
autocorrelated can inherit geography's significance and the p-values below are
anticonservative. Mantel test on the 737 between-monitor pairs, with the 4 near-coincident
(same `cv_group`) pairs excluded — decay shows as a **negative** correlation between shape
similarity and separation:

**Mantel r(shape similarity, distance) = −0.215, p = 0.0009.** Similarity falls with
separation, and falls monotonically across every bin:

| separation | pairs | median r(shape) |
|---|---|---|
| 1.2–8.5 km | 106 | 0.484 |
| 8.5–12.9 km | 105 | 0.384 |
| 12.9–17.1 km | 105 | 0.381 |
| 17.1–21.3 km | 105 | 0.328 |
| 21.3–25.6 km | 105 | 0.246 |
| 25.6–32.2 km | 105 | 0.236 |
| 32.2–47.6 km | 106 | 0.198 |

**On the full record the identical test gives +0.053, p = 0.518 — flat, with no bin
structure at all.** Same monitors, same statistic, same code; only the window differs.
That is a large reversal and I cannot resolve it with what is here. Two readings:

- the decay is real and the full record conceals it, because an 18-month profile averages
  two circulation regimes whose spatial gradients differ, and mixing them cancels the
  gradient out; or
- the decay is a product of the window's seasonality — Mar–Aug is monsoon-dominated, and a
  single coherent regional flow can impose a distance gradient that is about weather
  rather than about place.

One argument favours the first. The common-window profiles rest on a third of the data and
are measurably noisier (every split-half reliability below drops). Noise attenuates
correlations toward zero, so a *noisier* estimate showing a *stronger* spatial signal is
not the shape a noise artefact takes. Suggestive, not conclusive.

**What it does to the tests below.** The features are themselves spatially structured,
several of them strongly:

| feature | Mantel r vs distance | p |
|---|---|---|
| `dist_coast_m` | **+0.727** | 0.0005 |
| `building_density_500m` | +0.250 | 0.0050 |
| `dist_industrial_m` | +0.193 | 0.0080 |
| `road_density_1km` | −0.079 | 0.314 |
| `dist_major_road_m` | +0.017 | 0.855 |
| `road_density_500m` | −0.016 | 0.839 |

So for `dist_coast_m`, `building_density_500m` and `dist_industrial_m` both ingredients of
spatial confounding are now present, and their permutation p-values are anticonservative —
a free label permutation understates the true variance, so the p-values come out too small.
**For the four primary tests this happens to cut in a helpful direction, and that should be
said explicitly rather than glossed: all four are null, and a null under a test biased
towards false positives is a more robust null, not a weaker one.** Where it does bite is
the exploratory grid, where any hit on a spatially structured feature has to be discounted.
That is applied below.

It also costs the previous section part of its argument. That section read the flat
variogram and the flat shape-Mantel together as a correlation length shorter than the
network's spacing. The level half stands — it is a separate statistic, on the full record.
The shape half is window-dependent and should not be leaned on.

The near-coincident pairs still point where they did: 4 pairs at median separation 0.70 km
agree at **median r = 0.702**, against 0.484 for the closest ordinary bin and ~0.33 across
all pairs. Three of the four are cross-agency (IITM vs MPCB), so that agreement is not a
shared instrument or a shared processing pipeline. On 4 pairs it is suggestive, not
established.

### The summaries are reliable — but the window costs real precision

Split-half: every summary recomputed independently on odd and even local days, within the
primary window. The full-record column is the same split on the whole 18 months, and the
gap between them is what the shorter window costs.

| summary | split-half r | ceiling | *(full record r)* | typical half-to-half gap |
|---|---|---|---|---|
| morning–evening balance | 0.859 | 0.961 | *0.867* | 0.48 (summary units) |
| peak hour | 0.423 | 0.771 | *0.880* | 1.12 h |
| trough hour | 0.423 | 0.771 | *0.880* | 1.12 h |
| swing amplitude | 0.670 | 0.896 | *0.871* | 1.37 µg/m³ |
| bimodality | 0.520 | 0.827 | *0.690* | 0.36 |

Ceiling is √(Spearman-Brown reliability): the largest correlation a *perfectly* related
feature could show against a summary this noisy.

**This is where the swap hurts, and it is a real cost.** Morning–evening balance is
essentially untouched (0.859 vs 0.867), so both balance tests keep a ceiling above 0.96.
But the circular phase falls from 0.880 to 0.423 — ceiling 0.968 → 0.771 — because a phase
estimated from a third of the record is much noisier. The `dist_coast_m` × trough null is
therefore materially less informative than its full-record counterpart: no feature,
however perfectly related, could correlate above 0.77 with this summary. Swing drops from
0.871 to 0.670 (ceiling 0.896). The power table further down is computed for a noiseless
summary and so is optimistic for all but the two balance tests.

### The four primary tests

| feature | summary | test | statistic | p | **BH q** | n | ceiling |
|---|---|---|---|---|---|---|---|
| `dist_coast_m` | trough hour | Mardia R | 0.363 | 0.106 | 0.372 | 35 | 0.771 |
| `road_density_500m` | morning–evening balance | Spearman ρ | −0.229 | 0.186 | 0.372 | 35 | 0.961 |
| `dist_industrial_m` | morning–evening balance | Spearman ρ | +0.091 | 0.598 | 0.797 | 35 | 0.961 |
| `building_density_500m` | swing amplitude | Spearman ρ | +0.027 | 0.875 | 0.875 | 35 | 0.896 |

**0 of 4 survive BH at q < 0.05. None is significant even unadjusted.** The headline is
unchanged from the full-record version. Three of the sensitivity rows below do reach
p ≤ 0.05 on a single column — a May window start, the pseudo-replicated station-level unit,
and the swapped trough estimator — and each is discussed where it appears; none of the three
is a variant with a claim to being the analysis.

What did change is the sign pattern, and it is worth a line because it bears on how much
these point-estimates can be read at all. `road_density_500m` × balance and `dist_coast_m`
× trough both grew under the window, and both still point the way you would guess — busier
road → less morning-dominant, farther from the coast → shifted trough. The other two no
longer do: `building_density_500m` × swing is indistinguishable from zero, and
`dist_industrial_m` × balance points opposite to the full record. **Two of the four
directions are not stable enough across windows to interpret at all**, which is a stronger
statement of "null" than a small p-value on its own.

Sensitivities on the same four (each cell ρ or R / p):

| variant | n | road×balance | coast×trough | building×swing | industrial×balance |
|---|---|---|---|---|---|
| **PRIMARY: common window, CV groups** | 35 | −0.229 / 0.186 | 0.363 / 0.106 | +0.027 / 0.875 | +0.091 / 0.598 |
| full record — *the former primary* | 35 | −0.147 / 0.400 | 0.265 / 0.313 | **+0.233** / 0.179 | **−0.084** / 0.637 |
| station level (pseudo-replicated) | 39 | −0.184 / 0.261 | 0.394 / 0.050 | +0.007 / 0.966 | +0.063 / 0.703 |
| in-window coverage ≥ 50% | 33 | −0.204 / 0.249 | 0.317 / 0.208 | +0.027 / 0.881 | +0.071 / 0.691 |
| window start 2026-01-01 | 35 | −0.205 / 0.237 | 0.297 / 0.224 | +0.196 / 0.256 | −0.048 / 0.784 |
| window start 2026-03-01 | 35 | −0.255 / 0.139 | 0.331 / 0.159 | +0.089 / 0.607 | +0.073 / 0.675 |
| window start 2026-04-01 | 35 | −0.269 / 0.120 | 0.319 / 0.179 | −0.055 / 0.759 | +0.011 / 0.949 |
| window start 2026-05-01 | 35 | **−0.428 / 0.011** | 0.181 / 0.590 | +0.001 / 0.995 | +0.007 / 0.967 |
| artefact screen at 500 µg/m³ | 35 | −0.208 / 0.232 | 0.360 / 0.109 | +0.075 / 0.663 | +0.135 / 0.433 |
| amplitude relative to site level | 35 | — | — | −0.002 / 0.993 | — |
| trough from a 24h **+ 12h** fit | 35 | — | **0.535 / 0.005** | — | — |

Two rows need reading carefully rather than skimming.

**The window-start ladder is not reassuring.** `road_density_500m` × balance broadly
strengthens as the window starts later — −0.205 (Jan 1), −0.255 (Mar 1), −0.229 (the
primary Mar 10 start), −0.269 (Apr 1), −0.428 (May 1), the last of which is raw-significant
at p = 0.011. It is not monotonic — the primary start sits slightly weaker than a Mar-1
start — but the direction across the range is not in doubt. That could be the
confound clearing further, or it could be that a later start means a smaller, more
monsoon-pure sample and a less stable estimate; n is 35 throughout but each profile rests
on progressively fewer hours. Either way the estimate is window-sensitive, which cuts
against the primary as much as against the full record, and no start date later than the
last monitor's onset has any principled claim. **It is recorded, and it is not promoted.**

**The station-level row reaches p = 0.050 on coast × trough**, which is exactly what
pseudo-replication is expected to do — n rises from 35 to 39 by counting near-coincident
monitors twice, and p falls accordingly. It is the reason the CV group is the unit.

The morning/evening boundaries were a choice too, so they were varied — 05–09/17–21,
07–10/19–22, 06–09/19–22, 04–11/16–23, 07–09/20–22. Both balance tests stay null across all
five: `road_density_500m` ranges ρ = −0.13 to −0.28 (every p ≥ 0.10) and `dist_industrial_m`
+0.07 to +0.21 (every p ≥ 0.23). The other two columns are untouched by construction, which
is a useful check that the variation does what it claims. **The null is not an artefact of
where the hour windows were drawn** — though note the industrial estimate stays positive
across all five, so its sign flip against the full record is a property of the window, not of
the boundaries.

### What the nulls can and cannot rule out

This is the most important caveat and it is a hard one. At n = 35, a permutation Spearman
test:

| true ρ | power at α = 0.05 |
|---|---|
| 0.20 | 0.19 |
| 0.25 | 0.28 |
| 0.30 | 0.37 |
| 0.35 | 0.50 |
| 0.40 | 0.62 |
| 0.45 | 0.74 |
| **0.50** | **0.84** |
| 0.55 | 0.91 |

80% power arrives at about |ρ| = 0.48, *before* the BH correction, before the circular
tests, which are weaker still, and before the ceilings above — this table assumes a
noiseless summary, which only the balance summaries approach. All four primary effects sit
between 0.03 and 0.36, squarely in the range this network would miss more often than not.
(Two larger effects do turn up further down — both exploratory, neither pre-specified.)

**So the honest statement is "no relationship large enough for 35 monitors to see", not "no
relationship".** A true ρ of 0.3 between building density and swing amplitude would be a
real, publishable land-use effect, and we would have failed to detect it about two times in
three. Nothing here licenses the claim that land use is irrelevant to diurnal shape.

### Exploratory: the full 6 × 5 grid — NOT findings

Reported for completeness, per pre-specification, and not promoted. Cells are Spearman ρ
(signed) or Mardia R (unsigned, circular), with the permutation p in brackets:

| feature | balance | peak h | trough h | swing | bimodality |
|---|---|---|---|---|---|
| `dist_major_road_m` | −0.28 (0.109) | **0.47 (0.019)** | **0.47 (0.019)** | +0.10 (0.571) | +0.15 (0.384) |
| `road_density_500m` | −0.23 (0.186) | 0.27 (0.286) | 0.27 (0.286) | −0.02 (0.896) | +0.00 (0.981) |
| `road_density_1km` | **−0.51 (0.002)** | **0.51 (0.009)** | **0.51 (0.009)** | −0.05 (0.771) | +0.19 (0.271) |
| `dist_coast_m` | −0.08 (0.635) | 0.36 (0.106) | 0.36 (0.106) | +0.24 (0.166) | +0.03 (0.864) |
| `dist_industrial_m` | +0.09 (0.598) | 0.28 (0.269) | 0.28 (0.269) | −0.18 (0.311) | −0.12 (0.506) |
| `building_density_500m` | +0.24 (0.156) | **0.43 (0.039)** | **0.43 (0.039)** | +0.03 (0.875) | −0.17 (0.337) |

**7 of 30 cells reach raw p < 0.05, against 1.5 expected by chance — but 0 survive BH
across the grid**, the smallest q being 0.062 for `road_density_1km` × balance. The 7-vs-1.5
comparison overstates the case badly and should not be quoted without the correction that
follows.

Three separate reasons the grid is weaker than 7-of-30 sounds:

1. **The 30 cells are only 24 distinct tests.** Peak-hour and trough-hour are *identical by
   construction* — with a first-harmonic phase the trough sits 12 h from the peak, and
   Mardia's R is rotation-invariant, so the two columns are literally the same test. All
   three of the "new" hits are peak/trough duplicates. Counted properly the grid has **4
   distinct hits out of 24 tests, against 1.2 expected** — elevated, but a long way from
   7-vs-1.5, and BH across the grid clears none of them.
2. **The anticonservative p-values from the failed control apply here.** `building_density_500m`
   is spatially structured (Mantel +0.250), so its peak-hour hit is exactly the kind of
   result that bias manufactures, and it should be discounted.
3. **The full-record grid found 1 hit in 30.** The grid's contents are not stable across the
   window either.

The one cell that survives all three cuts is `road_density_1km` × balance, ρ = −0.51,
p = 0.002, BH q = 0.062: denser road network within 1 km → less morning-dominant. It is the
largest ρ anywhere in this analysis, it points the same way as the pre-specified
`road_density_500m` test (−0.23) at a larger radius, its summary is the most reliable of
the five (ceiling 0.961), and `road_density_1km` is one of the three features with **no**
spatial structure (Mantel −0.079, p = 0.31), so the control's failure does not touch it.
**It is still not a finding.** It was not pre-specified, it does not survive correction
across the grid it was found in, and on the full record the same cell gives ρ = −0.21,
p = 0.234 — so it is also window-dependent. It goes in the open-questions list, not the
results.

### The one exploratory result worth writing down, and why it still isn't promoted

The trough-hour statistic was pre-specified as a first-harmonic phase. Swapping it for the
continuous minimum of a 24h **+ 12h** fit — which, unlike the first-harmonic phase, is not
pinned 12 h from the peak — moves `dist_coast_m` × trough hour from R = 0.363, p = 0.106 to
**R = 0.535, p = 0.005**. Substituted into the primary set it would carry a BH q of 0.020.

Under the full record this variant was the one thing in the section that survived every
robustness check thrown at it. **Under the primary window it no longer does**, and the
checks that fail are the ones that matter:

- Leave-one-group-out R ranges 0.492–0.587. No single group drives it. *(Passes.)*
- The statistic is bimodal — 22 monitors trough pre-dawn (1.3–8.0 h), 17 in the afternoon
  (13.6–19.0 h). On the full record that split was 5 / 34 and the two clusters did not
  differ in `dist_coast_m`. **Here they do** (Mann-Whitney p = 0.035), so a large part of
  the association is the cluster split itself rather than a gradient within either cluster —
  and "which of two modes a site falls into" is a far cruder claim than "how far its trough
  shifts with distance inland". *(Fails.)*
- Dropping the 22 pre-dawn sites leaves R = 0.577 on n = 16 groups, **p = 0.072 — no longer
  significant.** On the full record the equivalent check held at p = 0.033. *(Fails.)*
- Within the afternoon arc a plain Spearman gives ρ = −0.582, p = 0.018: farther inland →
  earlier afternoon trough, the coherent sea-breeze direction. *(Passes, on 16 groups.)*
- Split-half stability of the statistic itself: circular r = 0.672 (0.780 on the full
  record), median half-to-half difference 0.68 h, and **6 of 39 sites flip cluster between
  halves** — against 2 of 39 before. The bimodal split that now carries the association is
  itself unstable in 6 of the 39 sites. *(Weakened.)*

So the picture is worse than it was, not better, despite the smaller p-value. The
association is larger and the mechanism behind it is less clean.

**And the original objection is untouched and still decisive: the estimator was changed
after the pre-specified one came back null.** That is the garden of forking paths in its
purest form, and no amount of post-hoc robustness repairs it — every one of those checks
was also run after seeing the p-value. The 24h+12h fit is not obviously the *wrong* choice,
which is precisely the problem: it is a defensible alternative that would not have been
looked at had the first one worked.

Two further reasons to hold it at arm's length. The split-half replication tests only that
the *summary* is stable, not that the *association* replicates — both halves use the same
35 CV groups and the same `dist_coast_m` values, so they were never independent evidence
about the relationship. And `dist_coast_m` is by far the most spatially structured feature
in the set (Mantel +0.727) at exactly the moment the control has stopped clearing spatial
confounding — so this is the single test in the whole section where "it is really just
geography" is hardest to dismiss, and it is now unclearable rather than merely awkward.

**This is a hypothesis for a future pre-registration, not a result.** The clean test would
be `dist_coast_m` against a 24h+12h trough hour, fixed in advance, on monitors this
analysis has not touched.

### Flags — things that make the above weaker than it looks

- **Power dominates everything.** n = 35 detects |ρ| ≈ 0.48 at 80%, and that figure assumes
  a noiseless summary. All four nulls are uninformative about small-to-moderate real
  effects. See the power table.
- **The primary window holds no pollution season.** Mar–Aug 2026 only: zero Nov–Feb hours,
  mean concentration 31% below the full record. Whatever land use does to diurnal shape
  under a winter inversion, this analysis cannot see it. That is the price of removing the
  seasonal confound and it is not a small one.
- **The primary window's profiles rest on a third as much data.** 3,434 hours behind the
  median profile against 10,806. What that costs in precision is measured directly rather
  than assumed: circular-phase reliability falls from 0.880 to 0.423, capping the coast ×
  trough test at a 0.77 ceiling.
- **The control now fails.** Shape similarity decays with distance in the primary window
  (Mantel −0.215, p = 0.0009), so the three spatially structured features have
  anticonservative p-values. Harmless for the nulls, which it can only have made harder to
  achieve; not harmless for anything in the exploratory grid.
- **The control's verdict is window-dependent, which is its own problem.** Flat on the full
  record, decaying on the window. One of those two is being produced by the seasonal
  composition of its span and there is nothing here that says which.
- **Two of four primary directions are unstable.** `dist_industrial_m` × balance flips sign
  and `building_density_500m` × swing collapses between the two windows. The point-estimates
  should not be read as weak evidence of anything, in either direction.
- **`road_density_500m` × balance depends on where the window starts**, broadly
  strengthening to −0.428 (p = 0.011) at a May start — not monotonically, since a Mar-1
  start is stronger than the primary Mar-10 one. No later start is principled, but the
  ladder means the primary estimate is not stable either.
- **Amplitude is partly a level statistic** — a dirtier site swings more in absolute µg/m³
  for the same fractional cycle — though within the primary window the link is weak and not
  significant (ρ = +0.249, p = 0.126, against +0.482 on the full record). For this test it is
  inert regardless: `building_density_500m` barely correlates with mean level (ρ = +0.143)
  and dividing amplitude by level gives the same answer (−0.002 vs +0.027).
- **Features are a static OSM snapshot** against a behavioural average. No feature varies by
  hour, which is a strange basis for explaining hour-of-day structure. Traffic *volume* by
  hour, which is what a road-density feature stands in for, is not measured.
- **Circular tests are weaker than the power table suggests.** It was computed for Spearman;
  Mardia's R spends a degree of freedom on the phase and needs a larger effect for the same
  p — on top of the 0.77 ceiling.
- **`bandra_east` retains only 2 of its 3 listed station ids** (7850 retired in 2021), so
  group collapsing is doing slightly less work than the CV group table implies.

### What would actually answer the question

More monitors, which is the same answer as everywhere else in this project — the multi-city
upgrade at the top of this file would take n from 35 into the hundreds and put |ρ| ≈ 0.15 in
reach, and would also make it possible to run the analysis within season rather than
choosing between a confounded record and a monsoon-only window. Failing that: a full year of
common record, so the window costs coverage rather than the entire pollution season; hourly
traffic counts instead of static road density; and pre-registered tests of the two
hypotheses this run turned up — the sea-breeze trough and `road_density_1km` × balance — on
monitors this analysis has not touched.

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

**BLOCKED: the live view needs the data.gov.in key authorised for dataset access.** Everything around it is built and tested; it simply cannot fetch.

`DATAGOV_API_KEY` in `.env` is a valid key — `https://api.data.gov.in/lists` returns 200 and reports 285,974 resources — but every `/resource/{id}` call returns `403 {"error": "Key not authorised"}`, including the CPCB feed `3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69` (which is live, `active=1`). Tested against four different resource ids; all 403. So it is the key's access level, not the resource id.

**To fix:** sign in at data.gov.in, open My Account, and generate or activate an API key with resource access. No code change needed — `data/live.py` reads it from `.env` and the app switches to live automatically.

**Why OpenAQ cannot stand in.** The obvious fallback is the source Phase 1 already uses, but it republishes CPCB on a lag: measured at **63.5 hours** behind for these stations (newest reading 27 Aug 02:30 UTC against a 29 Aug clock), with zero readings inside three hours. Its `/parameters/2/latest` endpoint also ignores `bbox`, `coordinates`+`radius` and every other spatial filter, returning the same global 1,000 rows regardless. Per-location calls do work — 39 of them run concurrently in ~5 s, inside the 60/min free tier — but return those same two-day-old numbers, so it would be a live-looking view of stale data. Not shipped.

**Do engineered land-use features actually beat kriging?** ~~Unknown~~ **Answered: no, and neither beats IDW.** Under leave-one-group-out CV, a properly tuned IDW baseline wins; kriging and gradient-boosted trees on the OSM features both lose to it, and every method is within ~1.7% of simply averaging the reporting stations. The variogram above explains why. The modest honest result is the deliverable — this is it.

Two traps found while getting there, worth remembering:
- *Tune the baseline or the comparison is worthless.* The conventional 1/d² is not IDW's best power here; with p=2 kriging appeared to win by 2.4%, and with the power chosen per fold IDW wins instead.
- *Tuning needs the group structure too.* Selecting the IDW power by predicting each training station from the others let `bandra_east`'s 70 m partner leak in. The inner split has to drop whole groups, same as the outer one.

**Settled: a grid cell containing a monitor is always in-range, whatever its features say.**

The mask exists to flag locations with no comparable training data. A cell with a monitor standing in it has ground truth by definition, so refusing to estimate there is incoherent — and it was happening: **11 of 39 monitors sat in masked cells**, including Bandra Kurla Complex, the airport, Powai and Andheri East. Press "Use my location" while standing at BKC and the app said it could not estimate, with a monitor 600 m away.

The cause is two things compounding. The mask is evaluated at the *cell centre*, which at 2.3 km cells can sit most of a kilometre from the monitor; and a min–max band has no tolerance at its own edge. BKC's cell was excluded for `road_density_500m` of **28.99 against a training maximum of 28.98** — a margin of 0.035%. That is a rounding-scale difference, not extrapolation.

Deliberately *not* fixed with a percentage tolerance. Any figure would be arbitrary, chosen to make this case pass, and would need re-defending every time a cell landed just outside it. The monitor rule is defensible from first principles instead.

Effect on coverage — small, and smaller at finer resolutions, because a smaller cell puts its centre closer to the monitor:

| grid | cell size | masked before | masked after | cells freed |
|---|---|---|---|---|
| 15×15 | 3.8 km | 82.7% | 77.3% | 12 |
| 25×25 | 2.3 km | 81.3% | **79.8%** | 9 |
| 40×40 | 1.4 km | 81.2% | 80.9% | 4 |
| 60×60 | 0.9 km | 81.4% | 81.2% | 6 |

39 monitors occupy 32 distinct cells at the default resolution — the near-coincident CV pairs share one. After the change, 0 of 39 monitors sit in a masked cell at any of the four resolutions. Implemented in `clear_station_cells()`, applied both when baking `grid_masks.npz` and on the un-baked fallback path so the two cannot disagree.

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

**Three stations came online March 2026.** Kalu Nagar, Vithalwadi, Chinchpada — ~25–28% coverage each. Fine for recent predictions, useless for training on 2025 data. Consider whether to include them at all. They are also what forces the diurnal–landuse analysis onto a common Mar-2026-on window, at the cost of the whole pollution season.

**OPEN: does diurnal shape decay with distance, or doesn't it?** The Mantel control in the
land-use section gives opposite answers on the two windows — flat on the full record
(r = +0.053, p = 0.518), clearly decaying on the primary common window (r = −0.215,
p = 0.0009), same monitors and same code. One of the two is an artefact of its span's
seasonal composition and nothing here says which. It matters beyond that section: the
"correlation length shorter than the network's spacing" reading of the headline result
leans on the flat version. Worth resolving with a season-matched comparison — the same
calendar months in 2025 and 2026 — before either version is quoted again.

**OPEN: the sea-breeze hypothesis, for a real pre-registration.** The land-use tests above
are all null, but one exploratory variant — `dist_coast_m` against a trough hour taken from a
24h+12h harmonic fit rather than the pre-specified first-harmonic phase — gives R = 0.535,
p = 0.005 on the primary window (ρ = −0.582 on the afternoon-trough arc: farther inland,
earlier trough). It is *not* a finding, for the reason it never was — the estimator was chosen
after the pre-specified one came back null — and on the primary window it also stopped passing
its own robustness checks: the pre-dawn/afternoon cluster split now itself carries coastal
information (p = 0.035) and dropping the pre-dawn sites leaves p = 0.072. Written up in full in
the land-use section. If it is ever tested properly, fix the 24h+12h trough in advance and use
monitors this analysis did not touch.

**OPEN: `road_density_1km` × morning–evening balance.** ρ = −0.51, p = 0.002 (BH q = 0.062
across the exploratory grid) on the primary window — the largest ρ anywhere in the
analysis, pointing the same way as the pre-specified 500 m test, on the most reliable summary,
and on one of the three features with no spatial structure, so the control's failure does not
touch it. Not pre-specified, does not survive grid-wide correction, and gives ρ = −0.21,
p = 0.234 on the full record. Same treatment as the sea-breeze one: pre-register or drop.
