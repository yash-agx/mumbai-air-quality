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

---

## Open questions

**Do engineered land-use features actually beat kriging?** Unknown, and this is the finding. Kriging only knows distance; ML can know road density and industrial proximity. But with 39 stations the gains may be modest. A modest honest result is the deliverable — not an inflated one.

**Stuck sensors.** Several stations repeat one value for 24+ hours (Khadakpada 456h, Mulund West 211h). Currently masked rather than dropped. Worth checking whether masking changes results materially.

**Three stations came online March 2026.** Kalu Nagar, Vithalwadi, Chinchpada — ~25–28% coverage each. Fine for recent predictions, useless for training on 2025 data. Consider whether to include them at all.
