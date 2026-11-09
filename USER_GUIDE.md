# User guide

What this software takes, what it produces, every option it accepts, and what it is
expected to do. For a walkthrough, start with [TUTORIAL.md](TUTORIAL.md) instead; for
installation, see the Quickstart in [README.md](README.md).

The package has two halves and they are independent:

1. **The resolvability audit** — `XAI_Rfpipe.resolvability` and
   `XAI_Rfpipe.robustness`. Reusable on any model-by-block comparison, including
   published ones. Needs only NumPy, pandas and SciPy. This is the paper's
   contribution.
2. **The demonstration pipeline** — `run_pipeline.py` and the remaining modules.
   Reproduces the rainfall-distribution-shift study the paper uses as its worked
   example.

Nothing in the audit depends on the pipeline. If you came here to audit your own
benchmark, you can ignore sections 2 and 3 entirely.

---

## 1. The audit

### 1.1 Inputs

The audit accepts a comparison in either of two forms.

**A score matrix**, as a pandas `DataFrame` with blocks as rows and models as columns.
"Block" is whatever unit the comparison is summarised over — basin, catchment, grid
cell, station, site, year. Cell `(i, j)` is model `j`'s score on block `i`. Scores are
read as **losses, so lower is better**; pass negated values for a skill score. Any
numeric dtype works. The matrix must have at least two columns and at least one row.
You also pass `family_a`, the list of column names forming the family of interest.

**Four integers**, when you have only a published summary: `wins`, `n_blocks`, `k_a`,
`k_b`. This is what makes the audit applicable to a comparison whose data was never
deposited, and all four are normally reported.

### 1.2 Outputs

Every function returns a plain `dict` or `DataFrame`. Nothing is written to disk unless
you are running the pipeline.

| Call | Returns |
|---|---|
| `family_win_probability(k_a, k_b)` | the reference null `k_a / (k_a + k_b)` |
| `family_win_test(wins, n_blocks, k_a, k_b)` | the candidate-aware test: `p_value`, the naive fair-coin contrast, and `verdict_changes_under_correct_null` |
| `family_win_profile_permutation(mat, family_a)` | the randomisation test by common relabelling — **the operative test** |
| `design_report(k, n_blocks, k_a, k_b)` | what the design can and cannot establish, before any data is seen |
| `detectable_win_counts(n_blocks, k_a, k_b)` | which win counts this design could ever call significant |
| `blocks_required_for_win_claim(k_a, k_b)` | the smallest `n_blocks` at which a claim is attainable |
| `pool_sensitivity(mat, family_a)` | how the verdict moves as pool membership changes |
| `run_audit(res)` | all of the above, from a long-format results table |

From `robustness`:

| Call | Returns |
|---|---|
| `within_family_correlation(mat, family)` | mean within-family score correlation — **check this first** |
| `effective_candidates(k_a, k_b, rho_a, rho_b)` | the candidate counts the correlation implies |
| `win_probability_correlated(k_a, k_b, rho_a, rho_b)` | the null under correlated candidates |
| `power_curve(n_blocks, k_a, k_b)` | power against a stated alternative |
| `blocks_for_power(k_a, k_b, alt_p)` | blocks needed to reach a target power |

### 1.3 Options

`alpha` (default `0.05`) is the significance level throughout. `n_perm` (default
`20000`) is the number of relabellings; the attainable p-value resolution is
`1 / (n_perm + 1)`, so do not reduce it below about 2000 if you intend to quote a
p-value. `seed` (default `0`) fixes the permutation draw, so results are reproducible
and two runs with the same seed are byte-identical. `ties` (default `0`) is the number
of blocks with no unique winner. `direction` on `blocks_required_for_win_claim` is
`"greater"`, `"less"` or `"two-sided"`; the paper quotes two-sided.

### 1.4 Expected behaviour

The audit is a statistical procedure, so "expected behaviour" means more than "it does
not crash". These are the properties the test suite checks, and you can verify any of
them yourself with `python tests/check_invariants.py results`:

- `family_win_probability(k, k)` returns exactly `0.5`. A balanced comparison is
  unaffected by the correction — that is the point.
- The candidate-aware test holds its nominal size across the design space: the realised
  type-I error never exceeds `alpha` by more than simulation noise.
- The fair-coin reading does not. Its error **grows with the number of blocks**, because
  a test centred on the wrong null probability is biased rather than merely noisy. At
  one-versus-eighteen with 25 blocks it rejects a true null essentially always.
- `family_win_profile_permutation` is typically **less** significant than the binomial
  when model skill persists across blocks, and that is correct: the profile null is
  more variable. If the relabelling gives you a smaller p-value than the binomial,
  check for a coding error.
- Additive dependence between blocks does not disturb the reference null, for a
  structural reason: adding a constant to a block cannot change which model attains its
  minimum.
- Strongly correlated candidates within a family **do** disturb it, and the correction
  then understates the null. Always read `within_family_correlation` before quoting a
  p-value.

One documented failure mode, kept here as a caution. Permuting model labels *within*
each block reproduces the binomial null exactly, for any score matrix whatsoever, so
agreement between the two is a tautology and not a validation. An earlier version of
this work made that mistake. `within_block_permutation_is_binomial` demonstrates it,
and `family_win_permutation` is retained only to show the equivalence. Neither should
be used as a test.

---

## 2. The demonstration pipeline

### 2.1 Inputs

Read from `--root` (default the repository root):

- `data/power/{Catamarca,Cordoba,Lamadrid}.csv` — daily NASA POWER series for three
  MERRA-2 grid cells, 1981-01-01 to 2024-07-01, POWER's own column names, `-999` as the
  fill value. Converted to NaN on load, counted in the audit, never imputed.
- `data/floods/{Catamarca,Cordoba,Lamadrid}.csv` — the documented flood catalogue,
  semicolon-separated, Spanish column names, year and month only. Both UTF-8-with-BOM
  and cp1252 accents are handled, so the shipped files are byte-identical to the working
  copies.
- `data/published/original_run_block_matrix.csv` — the transcribed block matrix of the
  earlier thirteen-model run, used by `audit_published.py`.

See [DATA.md](DATA.md) for provenance, licensing and — importantly — what the flood
catalogue does **not** support.

### 2.2 Outputs

All artefacts go to `--out` (default `results/`), numbered by stage, and every number in
the paper is read from one of these files rather than carried in prose. `00_manifest.json`
records the run; `SUMMARY.txt` is human-readable; `tables.tex` holds LaTeX fragments.
Figures are written as both PNG and PDF, with timestamps suppressed so repeated runs are
byte-identical.

### 2.3 Options

    python run_pipeline.py --stages all          # everything, ~70 min on 2 cores
    python run_pipeline.py --stages data,targets # ~1 s, proves the install
    python run_pipeline.py --stages stats,proxy  # reuses results/ already on disk

Stages, in order: `data`, `targets`, `models`, `stats`, `sensitivity`, `xai`, `proxy`,
`report`. Later stages read earlier stages' CSVs from disk, so a long run can be
resumed, and any stage can be run alone provided its predecessors' artefacts exist.

`--root` sets the input directory, `--out` the artefact directory, `--seeds` overrides
the seed count. `--quick` trims seeds, bootstrap and permutation counts for a smoke
test: it changes no definition, so quick results are directionally comparable but
**must not be quoted**.

### 2.4 Expected behaviour

A full run takes about 70 minutes on two CPU cores and needs no GPU and no
deep-learning framework. It is deterministic: the same inputs and seeds give the same
artefacts, byte for byte, figures included. It fails loudly rather than silently —
`XAI_Rfpipe.figures.build_fig5` refuses to draw below 95% attribution coverage rather
than present a map with holes in it.

---

## 3. Checking a run

    python tests/check_invariants.py results   # 29 checks against the artefacts
    python tests/test_caravan_loader.py        # 15 checks, synthetic fixture
    python audit_published.py                  # re-reads published comparisons

`check_invariants.py` tests the paper's claims against the files a run produced, so it is
how you confirm your run agrees with ours. `test_caravan_loader.py` runs against a
synthetic Caravan-schema fixture and needs no download. Both exit non-zero on failure
and are run by continuous integration on every push.

---

## 4. Reporting problems

Open an issue with the output of `python tests/check_invariants.py results`, your Python
version, and the contents of `results/00_manifest.json`. For a question about the audit
rather than the software, the relevant sections of the paper are the method and the
statistical validation.
