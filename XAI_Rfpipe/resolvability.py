"""
XAI_Rfpipe.resolvability - can this design resolve the claim it is about to make?

Blocked model comparisons in environmental modelling are routinely summarised by
a family win count: the best model of one family beat the best model of another
in w of N sites, basins or station-target blocks. The statistic is easy to read
and easy to report, and three things about it are almost never stated.

1. ITS NULL IS NOT ONE HALF.
   When the comparison is "best of K_A against best of K_B", and the two
   families hold different numbers of candidates, the win count is biased by
   candidate count alone. Under the null that no model is systematically better
   - that within a block the k = K_A + K_B scores are exchangeable - the block
   minimum is equally likely to belong to any model, so

       P(min of A < min of B) = K_A / (K_A + K_B)                          (1)

   not 1/2. Three neural models against eleven conventional ones gives 3/14:
   the conventional family is expected to win about 78% of blocks when nothing
   distinguishes any model from any other. Testing such a count against a
   fair-coin null (the implicit default, and the explicit null of the binomial
   test in Fidal and Kjeldsen 2020, which is correct for the one-against-one
   case they treat) attributes to the architecture what belongs to the size of
   the candidate pool. Best-of-K comparisons are common in this literature -
   Arsenault et al. (2023) compare a single LSTM against the per-basin best of
   eighteen model-regionalisation combinations - so the correction is not a
   hypothetical concern. There the asymmetry runs against the authors' own
   claim and so strengthens it; reversed, it manufactures one.

2. IT MAY NOT BE RESOLVABLE AT ANY VALUE.
   With few blocks, no attainable win count is significant. If N = 5 and
   p0 = 3/14, even a clean sweep gives p = 0.27: the design cannot support a
   family claim at alpha = 0.05 whatever happens. This is knowable BEFORE the
   experiment is run, from the design alone, and is worth more to a reader than
   the count itself.

3. IT DEPENDS ON THE POOL.
   Benavoli et al. (2016) showed that mean-rank post-hoc tests are pool
   dependent: whether A differs from B can change when an unrelated C joins the
   experiment. A family win count has the same defect by a different mechanism -
   adding a candidate to a family changes that family's null win probability
   through (1), and may change which model is its per-block best. Leaving each
   model out in turn measures how much of the conclusion is the pool's doing.

This module computes all three, plus the rank-based counterpart of (2): the
minimum mean-rank difference a Friedman-Nemenyi design can detect, and the
number of blocks a target difference would need. Nothing here is specific to
rainfall; it takes a models-by-blocks matrix of any score where lower is better.

References
----------
Benavoli, A., Corani, G., Mangili, F. (2016). Should we really use post-hoc
    tests based on mean-ranks? JMLR 17(5), 1-10.
Demsar, J. (2006). Statistical comparisons of classifiers over multiple data
    sets. JMLR 7, 1-30.
Fidal, J., Kjeldsen, T. (2020). Operational comparison of rainfall-runoff
    models through hypothesis testing. J. Hydrol. Eng. 25(4), 04020005.
    doi:10.1061/(ASCE)HE.1943-5584.0001892
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .statistics import Q05


# ───────────────────────────────────────────── the null a win count needs
def family_win_probability(k_a: int, k_b: int) -> float:
    """P(best of A beats best of B) in one block under exchangeability, eq. (1).

    Exact for continuous scores. Ties have probability zero in theory and are
    counted separately in practice by `family_win_test`.
    """
    if k_a < 1 or k_b < 1:
        raise ValueError("each family needs at least one candidate")
    return k_a / (k_a + k_b)


def family_win_test(wins: int, n_blocks: int, k_a: int, k_b: int,
                    ties: int = 0, alpha: float = 0.05) -> dict:
    """Exact binomial test of a family win count against the candidate-aware null.

    The naive fair-coin result is returned alongside, not as an alternative but
    as a diagnostic: when the two disagree, the disagreement is the finding.
    """
    p0 = family_win_probability(k_a, k_b)
    eff = n_blocks - ties
    p_hi = float(stats.binomtest(wins, eff, p0, alternative="greater").pvalue)
    p_lo = float(stats.binomtest(wins, eff, p0, alternative="less").pvalue)
    p_two = float(stats.binomtest(wins, eff, p0, alternative="two-sided").pvalue)
    naive = float(stats.binomtest(wins, eff, 0.5, alternative="two-sided").pvalue)
    return dict(
        wins=int(wins), n_blocks=int(n_blocks), ties=int(ties),
        k_a=int(k_a), k_b=int(k_b),
        null_p=p0, expected_wins=round(p0 * eff, 3),
        excess_wins=round(wins - p0 * eff, 3),
        p_greater=p_hi, p_less=p_lo, p_two_sided=p_two,
        significant=bool(p_two < alpha),
        naive_p_two_sided=naive,
        naive_significant=bool(naive < alpha),
        verdict_changes_under_correct_null=bool((naive < alpha) != (p_two < alpha)))


def family_win_permutation(mat: pd.DataFrame, family_a: list[str],
                           n_perm: int = 20000, seed: int = 0,
                           alpha: float = 0.05) -> dict:
    """Permutation null for the family win count, shuffling labels within blocks.

    Model labels are permuted inside each block, which destroys any systematic
    advantage while preserving that block's own spread of scores and whatever
    dependence exists between blocks (blocks are never resampled). It needs no
    continuity assumption, handles ties as observed, and it is the check that
    the closed form (1) is behaving.
    """
    cols = list(mat.columns)
    a_idx = [cols.index(c) for c in family_a if c in cols]
    b_idx = [i for i in range(len(cols)) if i not in a_idx]
    if not a_idx or not b_idx:
        return {}
    X = mat.to_numpy(float)
    rng = np.random.default_rng(seed)

    def count(mat_vals):
        a = mat_vals[:, a_idx].min(axis=1)
        b = mat_vals[:, b_idx].min(axis=1)
        return int((a < b).sum()), int((a == b).sum())

    obs, obs_ties = count(X)
    null = np.empty(n_perm, dtype=int)
    for i in range(n_perm):
        P = np.array([row[rng.permutation(len(cols))] for row in X])
        null[i] = count(P)[0]
    p_hi = float((null >= obs).mean())
    p_lo = float((null <= obs).mean())
    return dict(observed_wins=obs, ties=obs_ties, n_perm=int(n_perm),
                null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)),
                null_q05=float(np.quantile(null, 0.05)),
                null_q95=float(np.quantile(null, 0.95)),
                p_greater=p_hi, p_less=p_lo,
                p_two_sided=float(min(1.0, 2 * min(p_hi, p_lo))),
                significant=bool(min(1.0, 2 * min(p_hi, p_lo)) < alpha),
                closed_form_null_p=family_win_probability(len(a_idx), len(b_idx)),
                closed_form_expected=round(
                    family_win_probability(len(a_idx), len(b_idx)) * len(X), 3))


# ─────────────────────────────────────── what the design could have detected
def detectable_win_counts(n_blocks: int, k_a: int, k_b: int,
                          alpha: float = 0.05) -> dict:
    """The win counts this design could call significant, before any data.

    Returns the smallest count that would be significantly high, the largest
    that would be significantly low, and - the useful part - whether either
    exists at all. When neither does, the design cannot support a family claim
    at this alpha however the experiment turns out.

    "Could this design show family A is worse?" is a directional question, so
    the headline fields use the one-sided tail; the two-sided counterparts are
    returned beside them under `*_two_sided`, because a paper that reports
    two-sided p-values for the outcome should not quietly switch convention
    when reporting what was attainable. Report whichever you use, by name.
    """
    p0 = family_win_probability(k_a, k_b)
    hi = next((w for w in range(n_blocks + 1)
               if stats.binomtest(w, n_blocks, p0, alternative="greater").pvalue < alpha),
              None)
    lo = next((w for w in range(n_blocks, -1, -1)
               if stats.binomtest(w, n_blocks, p0, alternative="less").pvalue < alpha),
              None)
    hi2 = next((w for w in range(n_blocks + 1)
                if w / n_blocks > p0 and
                stats.binomtest(w, n_blocks, p0).pvalue < alpha), None)
    lo2 = next((w for w in range(n_blocks, -1, -1)
                if w / n_blocks < p0 and
                stats.binomtest(w, n_blocks, p0).pvalue < alpha), None)
    return dict(n_blocks=int(n_blocks), null_p=p0, alpha=alpha,
                tail="one-sided",
                min_significant_wins=hi, max_significant_losses=lo,
                min_significant_wins_two_sided=hi2,
                max_significant_losses_two_sided=lo2,
                any_high_claim_possible=hi is not None,
                any_low_claim_possible=lo is not None,
                resolvable=bool(hi is not None or lo is not None))


def blocks_required_for_win_claim(k_a: int, k_b: int, alpha: float = 0.05,
                                  direction: str = "greater",   # or "less"/"two-sided"
                                  max_blocks: int = 500) -> int | None:
    """Smallest N at which a clean sweep would be significant."""
    p0 = family_win_probability(k_a, k_b)
    for n in range(1, max_blocks + 1):
        w = n if direction == "greater" else 0
        kw = {} if direction == "two-sided" else dict(alternative=direction)
        if stats.binomtest(w, n, p0, **kw).pvalue < alpha:
            return n
    return None


def min_detectable_rank_difference(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference, read as a minimum detectable effect.

    CD is normally quoted after the fact to explain why nothing separated. Read
    before the fact it is a design statement: no mean-rank difference smaller
    than this can be declared, so a design whose CD approaches the width of the
    rank scale itself (k - 1) cannot distinguish the best model from the worst.
    """
    q = Q05.get(k)
    if q is None or n_blocks < 2:
        return float("nan")
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


def blocks_required_for_rank_difference(k: int, delta: float,
                                        alpha: float = 0.05) -> int | None:
    """Blocks needed before a mean-rank difference of `delta` is detectable."""
    q = Q05.get(k)
    if q is None or delta <= 0:
        return None
    return int(np.ceil(q ** 2 * k * (k + 1) / (6.0 * delta ** 2)))


def design_report(k: int, n_blocks: int, k_a: int, k_b: int,
                  alpha: float = 0.05) -> dict:
    """Everything knowable from the design alone, before a single model is fit."""
    cd = min_detectable_rank_difference(k, n_blocks, alpha)
    span = k - 1                      # widest possible mean-rank separation
    w = detectable_win_counts(n_blocks, k_a, k_b, alpha)
    return dict(
        k_models=int(k), n_blocks=int(n_blocks), alpha=alpha, tail=w["tail"],
        min_detectable_rank_difference=round(cd, 3),
        rank_scale_span=span,
        cd_as_fraction_of_span=round(cd / span, 3) if span else None,
        blocks_for_half_span=blocks_required_for_rank_difference(k, span / 2, alpha),
        family_null_p=round(w["null_p"], 4),
        min_significant_wins=w["min_significant_wins"],
        max_significant_losses=w["max_significant_losses"],
        # named for what it means: some attainable win count is significant.
        # This is a different question from whether the MEAN RANKS separate,
        # which `statistics.family_win_counts` reports as `resolvable`.
        some_win_count_significant=w["resolvable"],
        min_significant_wins_two_sided=w["min_significant_wins_two_sided"],
        max_significant_losses_two_sided=w["max_significant_losses_two_sided"],
        blocks_for_clean_sweep=blocks_required_for_win_claim(k_a, k_b, alpha),
        blocks_for_zero_win_claim=blocks_required_for_win_claim(
            k_a, k_b, alpha, direction="less"),
        blocks_for_zero_win_claim_two_sided=blocks_required_for_win_claim(
            k_a, k_b, alpha, direction="two-sided"))


# ─────────────────────────────────────────────── how much is the pool's doing
def pool_sensitivity(mat: pd.DataFrame, family_a: list[str],
                     alpha: float = 0.05) -> pd.DataFrame:
    """Recompute the family comparison with each model left out in turn.

    Benavoli et al. (2016) for win counts: a conclusion that moves when an
    unrelated model is dropped was never about the two families.
    """
    rows = []
    full_a = [c for c in family_a if c in mat.columns]
    for drop in [None] + list(mat.columns):
        m = mat if drop is None else mat.drop(columns=[drop])
        a = [c for c in full_a if c in m.columns]
        b = [c for c in m.columns if c not in a]
        if not a or not b:
            continue
        wins = int((m[a].min(axis=1) < m[b].min(axis=1)).sum())
        t = family_win_test(wins, len(m), len(a), len(b), alpha=alpha)
        rows.append(dict(dropped=drop or "(none: full pool)",
                         k_a=len(a), k_b=len(b), wins=wins,
                         null_p=round(t["null_p"], 4),
                         expected=t["expected_wins"],
                         p_two_sided=round(t["p_two_sided"], 4),
                         significant=t["significant"]))
    out = pd.DataFrame(rows)
    if len(out):
        base = out.iloc[0]
        out["wins_shift_vs_full"] = out["wins"] - base["wins"]
        out["flips_verdict"] = out["significant"] != base["significant"]
    return out


# ───────────────────────────────────────────────────────── the evidence ladder
def evidence_ladder(claims: list[dict]) -> pd.DataFrame:
    """One row per claim, with what the design can say about it.

    Each claim is {name, kind, supported_by, resolvable, detail}. The point of
    tabulating it is that a reader should not have to reconstruct which of a
    paper's statements its design could actually bear.
    """
    out = pd.DataFrame(claims)
    if len(out):
        order = {"resolvable": 0, "not resolvable": 1, "not testable": 2}
        out = out.sort_values(by="resolvable", key=lambda s: s.map(order))
    return out.reset_index(drop=True)


def run_audit(res: pd.DataFrame, neural: set[str] | None = None,
              metric: str = "test_MASE", agg: str = "median",
              n_perm: int = 20000, seed: int = 0, alpha: float = 0.05):
    """The whole audit for one result table. Returns (design, wins, perm, pool)."""
    from .statistics import block_matrix
    d = res.dropna(subset=[metric]).copy()
    mat = block_matrix(d, metric=metric, agg=agg)
    if mat.empty:
        return {}, {}, {}, pd.DataFrame()
    if neural is None:
        fam = d.drop_duplicates("model").set_index("model")["family"].to_dict()
        neural = {m for m, f in fam.items() if f == "neural"}
    a = [c for c in mat.columns if c in neural]
    b = [c for c in mat.columns if c not in neural]
    if not a or not b:
        return {}, {}, {}, pd.DataFrame()

    wins = int((mat[a].min(axis=1) < mat[b].min(axis=1)).sum())
    ties = int((mat[a].min(axis=1) == mat[b].min(axis=1)).sum())
    design = design_report(mat.shape[1], mat.shape[0], len(a), len(b), alpha)
    test = family_win_test(wins, mat.shape[0], len(a), len(b), ties, alpha)
    perm = family_win_permutation(mat, a, n_perm=n_perm, seed=seed, alpha=alpha)
    pool = pool_sensitivity(mat, a, alpha=alpha)
    return design, test, perm, pool


# ──────────────────────────────── how often does the common reading mislead?
def fair_coin_error_rate(n_blocks: int, k_a: int, k_b: int,
                         alpha: float = 0.05) -> dict:
    """Exact probability that the FAIR-COIN test rejects when nothing is better.

    This is the type-I error rate of the reading in common use, computed under
    the correct null of Eq. (1) rather than simulated. It is exact: the support
    of W is finite, so the rejection probability is a finite sum.

    For balanced pools it is at or below alpha, as a valid test should be. As
    the pools become unbalanced the fair coin is testing against the wrong
    centre, and the rate climbs - without limit in N, because the discrepancy
    between p0 and 1/2 is a bias, not a variance.
    """
    p0 = family_win_probability(k_a, k_b)
    w = np.arange(n_blocks + 1)
    pmf = stats.binom.pmf(w, n_blocks, p0)
    rej_naive = np.array([stats.binomtest(int(x), n_blocks, 0.5).pvalue < alpha
                          for x in w])
    rej_aware = np.array([stats.binomtest(int(x), n_blocks, p0).pvalue < alpha
                          for x in w])
    return dict(n_blocks=int(n_blocks), k_a=int(k_a), k_b=int(k_b),
                null_p=round(p0, 4), alpha=alpha,
                fair_coin_type_I=float(pmf[rej_naive].sum()),
                candidate_aware_type_I=float(pmf[rej_aware].sum()),
                inflation=float(pmf[rej_naive].sum() / alpha),
                outcomes_where_verdicts_differ=int((rej_naive != rej_aware).sum()),
                n_outcomes=int(n_blocks + 1))


def design_space_sweep(blocks=(5, 9, 15, 25, 50, 100, 200),
                       pools=((1, 1), (2, 3), (3, 11), (1, 9), (1, 18), (5, 8)),
                       alpha: float = 0.05) -> pd.DataFrame:
    """Type-I error of the fair-coin reading across a grid of designs.

    The grid is not arbitrary: (1,1) is the one-against-one case the correction
    leaves alone, (3,11) and (5,8) are the two runs of the demonstration,
    (1,18) is the published design re-read in the paper, and the rest fill in
    between. Nothing here touches any data.
    """
    rows = []
    for n in blocks:
        for ka, kb in pools:
            r = fair_coin_error_rate(int(n), ka, kb, alpha)
            r["pool"] = f"{ka} vs {kb}"
            rows.append(r)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
#  Randomisation that actually responds to dependence
# ═══════════════════════════════════════════════════════════════════════
#
# An earlier version of this module permuted model labels INDEPENDENTLY
# within each block and described the result as preserving dependence
# between blocks. That is false, and the failure is algebraic rather than
# approximate. Under a within-block permutation the identity of the block
# winner J_i is fixed by the observed scores; the permutation only decides
# which labels land in family A. So
#
#     I_i* = 1[J_i in A*]  ~  Bernoulli(K_A / k),
#
# independently across blocks because the permutations are drawn
# independently, and therefore W* ~ Binomial(N, K_A/k) EXACTLY, whatever
# the data look like. Agreement between that "permutation null" and the
# binomial null is a tautology and is no evidence that dependence is
# negligible. `family_win_permutation` above is retained only to
# demonstrate this; `within_block_permutation_is_binomial` verifies it.
#
# What does respond to dependence is a COMMON relabelling: draw one
# permutation of the model labels and apply it to every block. Each
# model's profile of scores across blocks stays intact, so if a single
# model wins most blocks the win count becomes nearly all-or-nothing and
# its variance far exceeds the binomial. The null is correspondingly
# stronger - whole-profile exchangeability of models - and it is stated
# rather than hidden.


def family_win_profile_permutation(mat: pd.DataFrame, family_a: list[str],
                                   n_perm: int = 20000, seed: int = 0,
                                   alpha: float = 0.05) -> dict:
    """Common-relabelling randomisation test for a family win count.

    Null: the k models' whole score profiles are exchangeable, i.e. the
    labelling of profiles to families carries no information. One
    permutation is drawn and applied to every block, so each model keeps
    its scores in every block and the concentration of wins across models
    is preserved.

    This tests a stronger hypothesis than Proposition 1's within-block
    exchangeability, and it is the weaker test: when wins concentrate on a
    few models the effective number of independent blocks is smaller than
    N and the null spreads out accordingly. Where it disagrees with the
    binomial, the binomial is the one to distrust.

    It does not address spatial dependence between blocks arising from the
    environment rather than from the models; see `effective_candidates`
    and the simulation study for what it does and does not cover.
    """
    cols = list(mat.columns)
    a_idx = [cols.index(c) for c in family_a if c in cols]
    if not a_idx or len(a_idx) == len(cols):
        return {}
    X = mat.to_numpy(float)
    rng = np.random.default_rng(seed)
    k = len(cols)
    ka = len(a_idx)

    def wins(assign_a):
        a = X[:, assign_a].min(axis=1)
        b = X[:, [j for j in range(k) if j not in assign_a]].min(axis=1)
        return int((a < b).sum()), int((a == b).sum())

    obs, obs_ties = wins(a_idx)
    null = np.empty(n_perm, dtype=int)
    for i in range(n_perm):
        null[i] = wins(list(rng.permutation(k)[:ka]))[0]
    p_hi = float((null >= obs).mean())
    p_lo = float((null <= obs).mean())
    p_two = float(min(1.0, 2 * min(p_hi, p_lo)))
    binom_sd = float(np.sqrt(len(X) * (ka / k) * (1 - ka / k)))
    return dict(scheme="common relabelling (whole-profile exchangeability)",
                observed_wins=obs, ties=obs_ties, n_perm=int(n_perm),
                null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)),
                binomial_sd=round(binom_sd, 4),
                variance_ratio=round(float(null.std(ddof=1)) ** 2 / binom_sd ** 2, 3)
                if binom_sd else None,
                p_greater=p_hi, p_less=p_lo, p_two_sided=p_two,
                significant=bool(p_two < alpha))


def within_block_permutation_is_binomial(n_blocks=9, k=14, k_a=3,
                                         n_perm=40000, seed=0) -> dict:
    """Demonstrate the equivalence that invalidated the earlier procedure.

    Builds a block matrix in which one model wins every block - maximal
    concentration - and shows that the within-block permutation null is
    still Binomial(N, K_A/k), while the common relabelling is not.
    """
    rng = np.random.default_rng(seed)
    cols = [f"m{j}" for j in range(k)]
    X = pd.DataFrame(rng.normal(size=(n_blocks, k)), columns=cols)
    X["m0"] = X.min(axis=1) - 1.0            # m0 wins every block
    a = cols[:k_a]
    wb = family_win_permutation(X, a, n_perm=n_perm, seed=seed)
    pr = family_win_profile_permutation(X, a, n_perm=n_perm, seed=seed)
    th = stats.binom(n_blocks, k_a / k)
    return dict(binomial_mean=float(th.mean()), binomial_sd=float(th.std()),
                within_block_mean=wb["null_mean"], within_block_sd=wb["null_sd"],
                profile_mean=pr["null_mean"], profile_sd=pr["null_sd"],
                within_block_matches_binomial=bool(
                    abs(wb["null_sd"] - th.std()) < 0.05),
                profile_differs_from_binomial=bool(
                    pr["null_sd"] > 2 * th.std()))


def block_matrix_and_family(res: pd.DataFrame, metric="test_MASE", agg="median",
                            neural: set[str] | None = None):
    """(block matrix, family-A column list) for one long-format result table."""
    from .statistics import block_matrix
    d = res.dropna(subset=[metric]).copy()
    mat = block_matrix(d, metric=metric, agg=agg)
    if neural is None:
        fam = d.drop_duplicates("model").set_index("model")["family"].to_dict()
        neural = {m for m, f in fam.items() if f == "neural"}
    return mat, [c for c in mat.columns if c in neural]
