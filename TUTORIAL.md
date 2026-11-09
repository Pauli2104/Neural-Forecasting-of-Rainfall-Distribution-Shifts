# Tutorial

Four worked exercises, in increasing order of involvement. Every command and every
number below was run against this version of the code; you should see the same output.

If you only have five minutes, do exercise 1. It is the one that applies to someone
else's published comparison, and it needs no data at all.

    git clone https://github.com/Pauli2104/Neural-Forecasting-of-Rainfall-Distribution-Shifts.git
    cd Neural-Forecasting-of-Rainfall-Distribution-Shifts
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

---

## 1. Re-read a published comparison (2 minutes, no data needed)

A comparison summarised as "our model beat theirs at *w* of *N* sites" can be audited
from four integers, which are almost always reported. Take Arsenault et al. (2023): one
LSTM against the per-basin best of eighteen hydrological model and regionalisation
combinations, the LSTM ahead in 86% of 148 catchments.

```python
from XAI_Rfpipe import resolvability as rv

t = rv.family_win_test(wins=127, n_blocks=148, k_a=1, k_b=18)
print(t["null_p"])                               # 0.05263157894736842
print(t["expected_wins"])                        # 7.789
print(t["p_two_sided"])                          # 2.135e-138
print(t["naive_p_two_sided"])                    # 1.111e-19
print(t["verdict_changes_under_correct_null"])   # False
```

Read that carefully, because it is the opposite of what people expect a "bias
correction" to do. The LSTM is a pool of **one** facing a pool of **eighteen**, so
selection favours its opponent: under candidate exchangeability the LSTM should win only
about 7.8 of 148 catchments, not 74. It won 127. The correction makes their result
**stronger**, not weaker, and `verdict_changes_under_correct_null` is `False` because
the conclusion was never in doubt.

Now the reverse case. Kratzert et al. (2019) compare one model against one ensemble:

```python
rv.family_win_probability(1, 1)   # 0.5 exactly
```

With equal candidate counts the correction is the identity. A balanced comparison is
untouched — which is the point, and worth confirming before you trust the machinery on
an unbalanced one.

> **Caveat.** Both of these are illustrations conditional on candidate exchangeability,
> not reanalyses. Eighteen regionalisation variants of the same hydrological model share
> structure and are certainly not exchangeable competitors, and a precise p-value would
> need the authors' score matrix. See exercise 3 for what correlation does to the null.

---

## 2. Audit your own comparison (5 minutes)

Now the normal case, where you have the scores. Arrange them as a DataFrame with blocks
as rows and models as columns, **lower is better**. Here are six models over twelve
blocks of pure noise, so we know the truth: nothing distinguishes any model.

```python
import numpy as np, pandas as pd
from XAI_Rfpipe import resolvability as rv, robustness as rob

rng = np.random.default_rng(7)
mat = pd.DataFrame(rng.normal(size=(12, 6)),
                   columns=["nn1", "nn2", "base1", "base2", "base3", "base4"])
A = ["nn1", "nn2"]

wins = int(mat.idxmin(axis=1).isin(A).sum())
print(wins, "of", len(mat))                  # 2 of 12
print(rv.family_win_probability(2, 4))       # 0.3333...
```

Two wins from twelve looks like a rout. Against a fair coin it would be reported as one.
But the neural family fields two candidates against four, so its reference null is 1/3,
and 2 of 12 against an expectation of 4 is unremarkable. The operative test is the
randomisation by common relabelling:

```python
perm = rv.family_win_profile_permutation(mat, A, n_perm=5000, seed=1)
print(perm["observed_wins"], perm["null_mean"])   # 2  4.0178
print(perm["p_two_sided"])                        # 0.3912
print(perm["variance_ratio"])                     # 0.999
```

No effect, correctly. `variance_ratio` near 1 says the permutation null and the binomial
agree here, which is what should happen when candidates really are independent — in the
paper's own data, where one model dominates most blocks, that ratio is about 3 and the
p-value triples.

**Always check correlation before quoting the p-value:**

```python
print(rob.within_family_correlation(mat, A))        # -0.094
print(rob.within_family_correlation(mat, ["base1", "base2", "base3", "base4"]))   # -0.190
```

Both near zero, so the reference null stands. Had either been strongly positive, go to
exercise 3.

---

## 3. When candidates are correlated

Correlated candidates behave as fewer candidates, so the reference null moves toward
1/2 and the uncorrected ratio understates it. This is the method's main limitation and
it is better to measure it than to hope.

```python
from XAI_Rfpipe import robustness as rob

rob.win_probability_correlated(k_a=1, k_b=18, rho_b=0.0)   # ~0.053, the nominal null
rob.win_probability_correlated(k_a=1, k_b=18, rho_b=0.9)   # moves sharply toward 0.5
rob.effective_candidates(k_a=1, k_b=18, rho_b=0.9)         # the counts that implies
```

As `rho_b` approaches 1 the eighteen behave as one and the null approaches 1/2. If your
own families contain near-duplicate configurations — the same model under different
regionalisations, the same architecture at different seeds — report the correlation
alongside the p-value, or use the effective counts.

---

## 4. Before you collect anything: can the design even answer the question?

This is the exercise worth doing *first*, and the one most often skipped. The paper's
own design is fourteen models over nine blocks, three of them neural:

```python
d = rv.design_report(k=14, n_blocks=9, k_a=3, k_b=11)
print(d["family_null_p"])                    # 0.2143
print(d["min_significant_wins"])             # 5
print(d["some_win_count_significant"])       # True
print(d["blocks_for_zero_win_claim_two_sided"])   # 16
print(d["min_detectable_rank_difference"])   # 6.614  (of a 13-wide rank span)
```

Read `blocks_for_zero_win_claim_two_sided` as the sting: with three neural models
against eleven conventional ones, **even winning zero of nine blocks is not
significant**. You would need sixteen blocks before a clean sweep in the other direction
could be claimed two-sided. A study of this shape cannot conclude that the neural family
is worse, however the win count falls — and that is knowable before any model is
trained.

```python
rv.blocks_required_for_win_claim(k_a=3, k_b=11, direction="two-sided")
rob.blocks_for_power(k_a=3, k_b=11, alt_p=0.5, target=0.8)
```

The first is attainability — the smallest design in which *some* outcome could reach
significance. The second is power against a stated alternative. They are different
questions and the paper keeps them apart; a design can be attainable and still have
almost no power.

---

## 5. Reproduce the paper

```bash
python run_pipeline.py --stages data,targets   # ~1 s, proves the install
python run_pipeline.py --stages all            # ~70 min on 2 cores
python tests/check_invariants.py results       # 29 checks against the artefacts
python audit_published.py                      # the published-comparison table
```

The run is deterministic: same inputs and seeds give the same artefacts byte for byte,
figures included. `--quick` trims seeds and permutation counts for a smoke test and
changes no definition, but quick numbers should not be quoted.

If `check_invariants.py` reports 29 of 29, your run agrees with ours. If not, the
failing line names the claim that broke, and that is the thing to report.

---

## What not to do

Do not reach for a within-block permutation of model labels. It reproduces the binomial
null **exactly, for any score matrix**, so agreement between the two proves nothing
about your data. An earlier version of this work reported that agreement as a
validation; it was a tautology, and the error is documented in the paper rather than
quietly fixed. `rv.within_block_permutation_is_binomial` demonstrates the equivalence if
you want to see it fail for yourself.

Next: [USER_GUIDE.md](USER_GUIDE.md) for the complete API, options and expected
behaviour.
