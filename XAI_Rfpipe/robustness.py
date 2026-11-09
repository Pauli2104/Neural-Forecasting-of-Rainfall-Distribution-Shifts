"""
XAI_Rfpipe.robustness - where the candidate-exchangeability reference null holds,
and where it does not.

Proposition 1 gives P(family A wins a block) = K_A/(K_A+K_B) under JOINT
exchangeability of the k candidate scores within the block. That is a
reference null, not a statement that the families perform equally: candidates
can have identical marginal distributions and still violate exchangeability
through their dependence, and then the probability of holding the block
minimum is not uniform over candidates.

The counterexample is small enough to state in a line. One candidate in A, two
in B, all three with the same marginal law. If the two B candidates are
independent of each other, P(A wins) = 1/3. If they are perfectly correlated
with each other - two variants of one architecture, say - they act as a single
competitor and P(A wins) = 1/2. Counting candidates cannot tell these apart.

This module therefore does three things the proposition alone cannot.

  effective_candidates   estimates the pool sizes that an equicorrelated model
                         would need in order to reproduce an observed or
                         assumed win probability, so a user can report a
                         correlation-adjusted reference instead of the raw
                         count.
  power_curve            the probability of rejection under an alternative,
                         which is a different quantity from whether any
                         attainable count lies in the rejection region.
  type_I_under_departures  the realised size of the candidate-aware test when
                         exchangeability fails in the ways benchmarking data
                         actually fail it.

Everything here is simulation under explicitly stated generative models. It
establishes when the reference null is usable; it does not establish that any
particular dataset satisfies it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .resolvability import family_win_probability


# ───────────────────────────────────────── correlation and effective pools
def win_probability_correlated(k_a: int, k_b: int, rho_a: float = 0.0,
                               rho_b: float = 0.0, n_sim: int = 200_000,
                               seed: int = 0, sd_a: float = 1.0,
                               sd_b: float = 1.0) -> float:
    """P(best of A beats best of B) under equicorrelated Gaussian candidates.

    Within each family the candidates are equicorrelated with rho; the two
    families are independent of each other. All marginals have mean zero, so
    no family is better in expectation: any departure from K_A/(K_A+K_B) is
    produced by the dependence or the scale, not by skill.
    """
    rng = np.random.default_rng(seed)

    def draw(k, rho, sd):
        if k == 1:
            return rng.normal(0, sd, size=(n_sim, 1))
        rho = float(np.clip(rho, 0.0, 0.999))
        common = rng.normal(size=(n_sim, 1)) * np.sqrt(rho)
        idio = rng.normal(size=(n_sim, k)) * np.sqrt(1 - rho)
        return (common + idio) * sd

    a = draw(k_a, rho_a, sd_a).min(axis=1)
    b = draw(k_b, rho_b, sd_b).min(axis=1)
    return float((a < b).mean())


def effective_candidates(k_a: int, k_b: int, rho_a: float = 0.0,
                         rho_b: float = 0.0, **kw) -> dict:
    """Pool sizes that reproduce the correlated win probability when independent.

    Solves K_eff_A / (K_eff_A + K_eff_B) = p for p the simulated probability,
    holding K_eff_B = K_B. The result is interpretable: perfectly correlated
    candidates within B collapse it towards a single competitor.
    """
    p = win_probability_correlated(k_a, k_b, rho_a, rho_b, **kw)
    p = min(max(p, 1e-6), 1 - 1e-6)
    k_eff_a = k_b * p / (1 - p)
    return dict(k_a=k_a, k_b=k_b, rho_a=rho_a, rho_b=rho_b,
                p_nominal=round(family_win_probability(k_a, k_b), 4),
                p_simulated=round(p, 4),
                effective_k_a=round(k_eff_a, 2), effective_k_b=k_b,
                nominal_minus_simulated=round(
                    family_win_probability(k_a, k_b) - p, 4))


def within_family_correlation(mat: pd.DataFrame, family: list[str]) -> float:
    """Mean pairwise rank correlation of a family's scores across blocks.

    Computed on within-block ranks so that differences in block difficulty do
    not masquerade as correlation between candidates. Returns NaN for a family
    of one, where the quantity is undefined.
    """
    cols = [c for c in family if c in mat.columns]
    if len(cols) < 2:
        return float("nan")
    r = mat.rank(axis=1)[cols]
    rs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = stats.spearmanr(r[cols[i]], r[cols[j]]).statistic
            if np.isfinite(v):
                rs.append(v)
    return float(np.mean(rs)) if rs else float("nan")


# ─────────────────────────────────────────────────────────────────── power
def power_curve(n_blocks: int, k_a: int, k_b: int, alpha: float = 0.05,
                p1=None) -> pd.DataFrame:
    """Rejection probability under alternatives, for the two-sided exact test.

    Attainability - whether any outcome lies in the rejection region - is a
    property of the design. Power is a property of the design AND an
    alternative, and the two are not interchangeable. The rejection region is
    fixed under the candidate-aware null; power is then evaluated under each
    alternative win probability p1.
    """
    p0 = family_win_probability(k_a, k_b)
    w = np.arange(n_blocks + 1)
    reject = np.array([stats.binomtest(int(x), n_blocks, p0).pvalue < alpha
                       for x in w])
    p1 = np.linspace(0.01, 0.99, 99) if p1 is None else np.asarray(p1, float)
    rows = []
    for q in p1:
        pmf = stats.binom.pmf(w, n_blocks, q)
        rows.append(dict(n_blocks=n_blocks, k_a=k_a, k_b=k_b, null_p=p0,
                         alt_p=round(float(q), 4),
                         power=float(pmf[reject].sum())))
    return pd.DataFrame(rows)


def blocks_for_power(k_a: int, k_b: int, alt_p: float, target: float = 0.8,
                     alpha: float = 0.05, max_blocks: int = 2000) -> int | None:
    """Smallest N reaching `target` power against a stated alternative."""
    p0 = family_win_probability(k_a, k_b)
    for n in range(2, max_blocks + 1):
        w = np.arange(n + 1)
        reject = np.array([stats.binomtest(int(x), n, p0).pvalue < alpha
                           for x in w])
        if float(stats.binom.pmf(w, n, alt_p)[reject].sum()) >= target:
            return n
    return None


# ───────────────────────────────── size of the test when the null is wrong
def type_I_under_departures(n_blocks: int = 9, k_a: int = 3, k_b: int = 11,
                            alpha: float = 0.05, n_sim: int = 20_000,
                            seed: int = 0) -> pd.DataFrame:
    """Realised size of the candidate-aware test under stated departures.

    Each scenario has NO family advantage in expectation: every candidate has
    mean zero. Any departure of the realised rejection rate from alpha is
    produced by the structure named in the scenario, which is the point.
    """
    rng = np.random.default_rng(seed)
    p0 = family_win_probability(k_a, k_b)
    k = k_a + k_b

    def run(sample):
        rej = 0
        for _ in range(n_sim):
            X = sample()
            w = int((X[:, :k_a].min(axis=1) < X[:, k_a:].min(axis=1)).sum())
            if stats.binomtest(w, n_blocks, p0).pvalue < alpha:
                rej += 1
        return rej / n_sim

    def equicorr(kk, rho, sd, n):
        if kk == 1:
            return rng.normal(0, sd, size=(n, 1))
        c = rng.normal(size=(n, 1)) * np.sqrt(rho)
        return (c + rng.normal(size=(n, kk)) * np.sqrt(1 - rho)) * sd

    SCEN = [
        ("exchangeable (the reference null)", 0.0, 0.0, 1.0, 1.0, 0.0, 1),
        ("candidates within B correlated 0.5", 0.0, 0.5, 1.0, 1.0, 0.0, 1),
        ("candidates within B correlated 0.9", 0.0, 0.9, 1.0, 1.0, 0.0, 1),
        ("candidates within A correlated 0.9", 0.9, 0.0, 1.0, 1.0, 0.0, 1),
        ("both families correlated 0.7", 0.7, 0.7, 1.0, 1.0, 0.0, 1),
        ("A twice as variable as B", 0.0, 0.0, 2.0, 1.0, 0.0, 1),
        ("A half as variable as B", 0.0, 0.0, 0.5, 1.0, 0.0, 1),
        ("blocks correlated 0.8", 0.0, 0.0, 1.0, 1.0, 0.8, 1),
        ("scores rounded to 2 dp (ties)", 0.0, 0.0, 1.0, 1.0, 0.0, 2),
    ]
    rows = []
    for name, ra, rb, sa, sb, rblock, roundto in SCEN:
        def sample(ra=ra, rb=rb, sa=sa, sb=sb, rblock=rblock, roundto=roundto):
            A = equicorr(k_a, ra, sa, n_blocks)
            B = equicorr(k_b, rb, sb, n_blocks)
            X = np.hstack([A, B])
            if rblock > 0:                     # a shared block effect
                X = X + np.sqrt(rblock) * rng.normal(size=(n_blocks, 1))
            return np.round(X, roundto) if roundto < 6 else X
        r = run(sample)
        rows.append(dict(scenario=name, rho_a=ra, rho_b=rb, sd_a=sa, sd_b=sb,
                         block_corr=rblock, n_blocks=n_blocks,
                         realised_size=round(r, 4), nominal=alpha,
                         ratio=round(r / alpha, 2)))
    return pd.DataFrame(rows)


def compare_tests_under_persistent_model_effects(
        n_blocks: int = 9, k_a: int = 3, k_b: int = 11, tau: float = 1.0,
        alpha: float = 0.05, n_sim: int = 2000, n_perm: int = 2000,
        seed: int = 0) -> dict:
    """Binomial versus common-relabelling test when model skill persists.

    Generative model: each candidate draws a persistent offset once, shared
    across all blocks, plus independent per-block noise. The offsets have mean
    zero over candidates, so no FAMILY is favoured in expectation - but some
    candidate is better than the others by chance, wins concentrate on it, and
    the number of effectively independent blocks falls below N.

    This is the structure the binomial null cannot see and that the common
    relabelling is built for, so it is the scenario on which the choice between
    them should be judged.
    """
    from .resolvability import family_win_profile_permutation
    rng = np.random.default_rng(seed)
    p0 = family_win_probability(k_a, k_b)
    k = k_a + k_b
    cols = [f"m{j}" for j in range(k)]
    rej_b = rej_p = 0
    for _ in range(n_sim):
        offs = rng.normal(0, tau, size=k)
        offs = offs - offs.mean()               # no family advantage on average
        X = offs + rng.normal(size=(n_blocks, k))
        w = int((X[:, :k_a].min(axis=1) < X[:, k_a:].min(axis=1)).sum())
        if stats.binomtest(w, n_blocks, p0).pvalue < alpha:
            rej_b += 1
        r = family_win_profile_permutation(pd.DataFrame(X, columns=cols),
                                           cols[:k_a], n_perm=n_perm,
                                           seed=int(rng.integers(1 << 31)))
        if r.get("p_two_sided", 1.0) < alpha:
            rej_p += 1
    return dict(scenario="persistent model effects", tau=tau,
                n_blocks=n_blocks, k_a=k_a, k_b=k_b, n_sim=n_sim,
                nominal=alpha,
                binomial_size=round(rej_b / n_sim, 4),
                profile_permutation_size=round(rej_p / n_sim, 4))
