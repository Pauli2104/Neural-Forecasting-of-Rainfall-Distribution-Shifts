# XAI_Rfpipe — reproducible pipeline for rainfall-distribution-shift forecasting

[![pipeline](https://github.com/Pauli2104/Neural-Forecasting-of-Rainfall-Distribution-Shifts/actions/workflows/ci.yml/badge.svg)](https://github.com/Pauli2104/Neural-Forecasting-of-Rainfall-Distribution-Shifts/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Every experiment, statistical test, table and result figure in **"Best of how
many? Candidate-count bias in environmental model benchmarking"** (submitted to
*Environmental Modelling & Software*), from the raw daily files to the numbers
in the paper.

The study asks how much can be concluded from a machine-learning study of
environmental change when the record is short: three reanalysis grid cells in
northwestern and central Argentina, 43 hydrological years each, from which annual
rainfall-distribution shift targets are built and forecast. The answer the paper
defends is a cautionary one, and this code is how it is checked rather than
asserted.

---

## Documentation

| File | For |
|---|---|
| [TUTORIAL.md](TUTORIAL.md) | four worked exercises; start here |
| [USER_GUIDE.md](USER_GUIDE.md) | inputs, outputs, every option, expected behaviour |
| [DATA.md](DATA.md) | provenance, licensing, and what the data does not support |

## Quickstart

```bash
git clone https://github.com/Pauli2104/Neural-Forecasting-of-Rainfall-Distribution-Shifts.git
cd Neural-Forecasting-of-Rainfall-Distribution-Shifts
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py --stages data,targets          # ~1 s, proves the install
python tests/check_invariants.py results              # the paper's claims, checked
python run_pipeline.py --stages all                   # full run, ~70 min on 2 cores
```

| invocation | what it does | time |
|---|---|---|
| `--stages all` | everything, from daily CSVs to `SUMMARY.txt` | ~70 min |
| `--stages all --quick` | same path, fewer seeds and resamples | ~17 min |
| `--stages stats,proxy` | reuses the `results/` already on disk | seconds |
| `--out mydir` | writes elsewhere, leaving the published run intact | — |

`--quick` trims seed counts, bootstrap resamples and permutations. It changes no
definition, so quick results are directionally comparable — but they should not
be quoted.

**Dependencies** are `numpy`, `pandas`, `scipy`, `scikit-learn`, and `matplotlib`
for the figure. PyTorch is **not** required, and `statsmodels` is not used
(SciPy ≥ 1.11 supplies Tukey HSD). `requirements-lock.txt` pins the exact
versions that produced the committed `results/`.

The repository ships with its input data (4.8 MB) and with the artefacts of a
full run, so the numbers in the paper can be inspected without running anything.
See [DATA.md](DATA.md) for provenance and attribution terms — the meteorology is
NASA POWER's and carries its own citation requirement.

---

## Layout

```
data/power/{Catamarca,Cordoba,Lamadrid}.csv     NASA POWER daily reanalysis
data/floods/{Catamarca,Cordoba,Lamadrid}.csv    documented flood catalogue
XAI_Rfpipe/
  core.py         loading, audit, hydrological years, descriptors, targets
  experiments.py  supervised framing, model zoo, metrics, uncertainty
  statistics.py   Friedman / Iman-Davenport / Nemenyi / Wilcoxon / Tukey
  sensitivity.py  lag-window, feature-set and seed sweeps
  explain.py      attributions and their agreement, faithfulness, stability
  resolvability.py  can this design resolve the claim it is about to make?
  figures.py      result figures, rebuilt from the artefacts
  proxy.py        flood-year discrimination
  postproc.py     ablation of daily post-processing components
  report.py       SUMMARY.txt and tables.tex
tests/check_invariants.py   the paper's claims, asserted against a run
audit_published.py          the audit applied to comparisons this pipeline did not run
run_pipeline.py             orchestrator; each stage writes CSV/JSON to results/
```

Every stage writes numbered artefacts (`01_…` to `09_…`) and can run alone, so a
long run is resumable and any single table can be regenerated.

---

## Stages

| stage | what it does | key outputs |
|---|---|---|
| `data` | load, audit integrity, build ETCCDI descriptors | `01_audit`, `02_descriptors` |
| `targets` | divergence targets, Proposition 1 check, trend tests | `03_targets`, `03_proposition1` |
| `models` | 14 models × 3 stations × 3 targets × seeds | `04_model_results`, `04_leakage_report` |
| `stats` | omnibus, post-hoc, contrasts, win counts, resolvability audit | `05_friedman`, `05_nemenyi_pairs`, `05_family_wins`, `05_resolvability` |
| `sensitivity` | lag / feature-set / seed sweeps | `06_factor_effects` |
| `xai` | four attribution methods, three diagnostics, and Fig. 5 | `07_attribution_agreement`, `07_faithfulness`, `Fig5_xai_heatmap` |
| `proxy` | AUROC against documented flood years | `08_discrimination` |
| `postproc` | post-processing ablation (below) | `09_postproc_*` |
| `report` | assembles everything | `SUMMARY.txt`, `tables.tex` |

---

## What the pipeline guarantees

**No leakage.** The targets are built from a moving window of `win_years`
hydrological years, so consecutive target values share `win_years − 1` years of
daily data. Splits are therefore strictly chronological with a validation buffer,
and `04_leakage_report.csv` *measures* the shared support rather than asserting
it is zero. On this configuration the measured overlap is **0.000**, with a
7-year gap between the last training year and the first test year.

**No bare point estimates.** Every metric carries a bootstrap confidence interval
and a margin of error; every model is run across seeds and summarised by
quantiles; interval forecasts are calibrated by split conformal prediction so
nominal and empirical coverage can be compared directly. On this record they
cannot: nominal 90% intervals achieve roughly half that coverage.

**Omnibus before post-hoc.** Friedman, then Iman–Davenport, and only then Nemenyi
critical differences. Pairwise Wilcoxon contrasts carry rank-biserial effect
sizes and Holm correction.

**A win count cannot be quoted bare.** `statistics.family_win_counts` counts its
denominator from the block matrix instead of taking one on trust — the design has
`stations × targets` blocks, and writing "of twelve" for a 3 × 3 design is the
arithmetic error that produced two of the three conflicting counts this project
had to reconcile. It returns the Nemenyi critical difference and the number of
models lying within one CD of the best alongside the count, and sets
`resolvable: false` when that number exceeds one. A model whose family assignment
is arguable is counted both ways through `alt_conventional`, and a name that
matches nothing is reported as skipped rather than passing silently.

**A win count is tested against the null its design implies.** When the per-block
winner on each side is the *best* of that side's candidates, the null win probability
is `K_A / (K_A + K_B)`, not one half: the maximum of a larger sample is stochastically
larger, so a family that brings more candidates wins more blocks even when every model
is exchangeable with every other. `resolvability.py` derives this, checks it against a
within-block permutation null, and reports what the design could ever have resolved -
which win counts it could call significant and how many blocks a claim would need.
On this record the correction reverses the verdict: the same 0-of-9 outcome gives
p = 0.004 against a fair coin and p = 0.22 against the candidate-aware null, and no
attainable outcome could have supported the claim at all. `audit_published.py` applies
the same audit to two published comparisons, where it strengthens one and leaves the
other untouched.

**A blank attribution cell means no attribution.** `run_xai` samples Shapley over
every feature (`shapley_top_k=None`). An earlier version sampled the top twelve,
which left 82% of the map uncomputed and indistinguishable from zero wherever it
was plotted. `figures.build_fig5` refuses to draw below 95% coverage.

**Nothing is fabricated.** Where the data cannot support an analysis the pipeline
says so and skips it rather than approximating. The clearest case is the next
section.

---

## The post-processing ablation

The record is a single deterministic reanalysis series. **There is no forecast
ensemble**, so SEAS5-style ensemble calibration cannot be run on it, and this
module does not pretend otherwise.

What it ablates is the *generative* half of that machinery — the statistical
model of daily precipitation a post-processor fits — under leave-one-year-out
cross-validation over all 129 station-years. Components: Bernoulli occurrence,
Gamma intensity, monthly conditioning, normal quantile transform with a
meta-Gaussian step, empirical quantile mapping, and the Schaake shuffle. Each is
switched off in turn (`minus_<component>`) against `full`, `climatology` and
`qmap_only`. This bears on the paper directly: the divergence targets are
functionals of the daily wet-day distribution, so anything that changes how that
distribution is represented changes the targets.

**Each component is scored on what it targets**, which is the point of the
design. CRPS is a *marginal* score and is blind to day-to-day structure, so
judging the Schaake shuffle by CRPS alone is a category error. The ablation
reports CRPS, the variogram score (proper and multivariate), lag-1
autocorrelation error, wet/dry spell errors, Brier score at the 1 mm threshold,
and rank-histogram flatness. On this record the shuffle **costs about 48% of
CRPS** and is **essential to temporal autocorrelation** — removing it degrades
lag-1 autocorrelation error by 98%. A component can earn its place and still cost on the
headline score; the ablation is built to show that rather than hide it.

Four of the six components are inert here: removing occurrence, seasonal
conditioning, the normal quantile transform or the Gamma intensity model moves
CRPS by at most 0.7%, with all Holm-corrected *p* ≥ 0.42.

---

## Where this pipeline and the manuscript differ

This is an independent reimplementation written from the definitions stated in
the paper, not ported from the exploratory notebooks. Its purpose is to establish
which findings survive reimplementation, so the differences are part of the
result and are listed rather than reconciled away.

**Three findings reproduce without qualification.** Proposition 1 holds
numerically — across 117 station-years the largest absolute difference between
EMD and W₁ is 2 × 10⁻¹⁴ with Pearson *r* = 1.0000, so the study has three
independent targets and not four, and the pipeline drops EMD from the modelling
stage for that reason. Train/test window overlap is measured and is exactly zero.
The ordering of design factors is unchanged: mean within-cell spread in test MASE
is 39.6 for the lag window, 30.1 for the feature set and 14.6 for the seed, so
the lag window moves the result 2.7× as far as reseeding does.

**Two reproduce in direction but not magnitude.** A raw descriptor still
discriminates documented flood years better than any shift target — RX1day
reaches AUROC 0.623 (*p* = 0.034) against 0.614 for D_KL (*p* = 0.041) — but both
differ from the manuscript's own run, which uses a different target construction.
Conventional baselines again dominate the neural comparators, by a wider margin
here because the architectures are untuned defaults.

**One is sharper.** Split-conformal intervals at a nominal 90% achieve 54%
empirical coverage for the conventional models and 71% for the neural ones.

**The per-block win count runs the other way.** With untuned default
architectures the best neural model loses all nine blocks and the three MLPs
occupy the bottom three mean ranks; the manuscript's run gives six of nine the
other way. Both sit inside the critical difference, which is the point:
`05_family_wins_summary.json` reports `resolvable: false`. The count is evidence
that the statistic is unstable, not a result.

**The flood catalogue holds 43 events**, not the 40 of earlier drafts: Catamarca
20, Córdoba 9, La Madrid 14.

---

## A caveat on the target definition

The divergence targets are implemented from the definitions stated in the paper —
reference window versus moving window over the wet-day amount distribution, on a
fixed 40-bin support to 60 mm — not ported line by line from the exploratory
notebooks, which do not expose them as reusable functions. `verify_prop1`
confirms the implementation behaves as Proposition 1 requires, which is a strong
check but not a proof of equivalence with the original code.

Anyone comparing against an independent implementation should cross-check one
station-year before quoting numbers across the two. `Config.ref_years`,
`win_years`, `n_bins` and `bin_max_mm` are the knobs that matter.

---

## Citing

See [CITATION.cff](CITATION.cff). Please cite both the paper and the archived
release of this code, and follow NASA POWER's own attribution requirement in
[DATA.md](DATA.md) when reusing the meteorology.

## Licence

Code: [MIT](LICENSE). Data and figures: see [DATA.md](DATA.md).
