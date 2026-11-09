"""
XAI_Rfpipe.postproc - ablation of statistical post-processing for daily precipitation.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This study has no forecast ensemble. The record is a single deterministic
reanalysis series (NASA POWER / MERRA-2), so there is nothing to post-process in
the sense of SEAS5-style ensemble calibration, and this module does not pretend
otherwise. Inventing ensemble members and calling them forecasts would be
fabrication.

What the data does support, and what is ablated here, is the *generative*
half of that machinery: the statistical model of daily precipitation that an
ensemble post-processor fits, and whose components are usually adopted together
without evidence that each one earns its place. The pipeline fits that model on
all but one hydrological year, draws an ensemble from it for the held-out year,
and scores the draw against the observed daily series. Leave-one-year-out over
the whole record, exactly as a post-processing study would cross-validate.

This matters for the paper directly: the divergence targets are functionals of
the daily wet-day distribution, so any component that changes how that
distribution is represented changes the targets.

COMPONENTS ABLATED
    occurrence   Bernoulli occurrence model (dry/wet), monthly
    intensity    Gamma intensity on wet days, monthly, versus resampling
    seasonal     monthly conditioning, versus one annual pool
    nqt          normal quantile transform with a meta-Gaussian joint step
    qmap         empirical quantile mapping onto the training climatology
    shuffle      Schaake shuffle, which restores day-to-day rank structure

CONFIGURATIONS
    full                 every component on
    minus_<component>    full with one component disabled (leave-one-out)
    climatology          none of them: resample training days at random
    qmap_only            the usual baseline in the post-processing literature

SCORES
    CRPS           continuous ranked probability score, empirical form
    Brier          occurrence at the 1 mm wet-day threshold
    rank_chi2      flatness of the rank histogram: calibration of the spread
    descriptor MAE error in the annual descriptors the ensemble implies
    target MAE     error in the divergence targets the ensemble implies
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .core import WET_DAY_MM, _hist, bhattacharyya, kl_divergence, w1_ordered

COMPONENTS = ("occurrence", "intensity", "seasonal", "nqt", "qmap", "shuffle")


def configurations() -> dict[str, dict]:
    full = {c: True for c in COMPONENTS}
    cfgs = {"full": dict(full)}
    for c in COMPONENTS:
        d = dict(full); d[c] = False
        cfgs[f"minus_{c}"] = d
    cfgs["climatology"] = {c: False for c in COMPONENTS}
    cfgs["qmap_only"] = {**{c: False for c in COMPONENTS}, "qmap": True}
    return cfgs


# ──────────────────────────────────────────────────────────── components
def fit_bernoulli_gamma(x: np.ndarray):
    """Occurrence probability and Gamma parameters for wet-day intensity."""
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return dict(p_wet=0.0, shape=np.nan, scale=np.nan, wet=np.array([]))
    wet = x[x >= WET_DAY_MM]
    p = float(len(wet) / len(x))
    if len(wet) < 5:
        return dict(p_wet=p, shape=np.nan, scale=np.nan, wet=wet)
    try:
        shape, loc, scale = stats.gamma.fit(wet, floc=0.0)
    except Exception:
        shape, scale = np.nan, np.nan
    return dict(p_wet=p, shape=float(shape), scale=float(scale), wet=wet)


def nqt_transform(x: np.ndarray):
    """Normal quantile transform: empirical CDF then the standard normal inverse."""
    r = stats.rankdata(x)
    u = (r - 0.5) / len(x)
    return stats.norm.ppf(u)


def nqt_inverse(z: np.ndarray, reference: np.ndarray):
    """Back-transform through the reference empirical quantiles."""
    u = np.clip(stats.norm.cdf(z), 1e-6, 1 - 1e-6)
    return np.quantile(reference, u)


def quantile_map(sample: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map a sample onto the reference climatology by matching quantiles."""
    if len(sample) == 0 or len(reference) == 0:
        return sample
    r = stats.rankdata(sample)
    u = np.clip((r - 0.5) / len(sample), 1e-6, 1 - 1e-6)
    return np.quantile(reference, u)


def schaake_shuffle(members: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Reorder each member in time to match the rank structure of a template.

    members: (M, T). template: (T,). Returns members with the same marginal
    values but the template's day-to-day ordering.
    """
    out = np.empty_like(members)
    order = np.argsort(np.argsort(template))
    for m in range(members.shape[0]):
        out[m] = np.sort(members[m])[order]
    return out


# ─────────────────────────────────────────────────────────────── scoring
def crps_empirical(members: np.ndarray, obs: np.ndarray) -> float:
    """CRPS from an ensemble, by the fair empirical estimator.

    CRPS = E|X - y| - 0.5 E|X - X'|, averaged over lead days.
    """
    M = members.shape[0]
    t1 = np.abs(members - obs[None, :]).mean(axis=0)
    s = np.sort(members, axis=0)
    w = (2 * np.arange(1, M + 1) - M - 1)[:, None]
    t2 = (2.0 / (M * (M - 1))) * (w * s).sum(axis=0) if M > 1 else 0.0
    return float(np.mean(t1 - 0.5 * t2))


def brier_occurrence(members: np.ndarray, obs: np.ndarray, thr=WET_DAY_MM) -> float:
    p = (members >= thr).mean(axis=0)
    o = (obs >= thr).astype(float)
    return float(np.mean((p - o) ** 2))


def variogram_score(members: np.ndarray, obs: np.ndarray, p: float = 0.5,
                    max_lag: int = 10) -> float:
    """Variogram score of order p (Scheuerer & Hamill 2015).

    A proper MULTIVARIATE score. CRPS judges each day separately and is blind to
    day-to-day structure, so it cannot assess the Schaake shuffle, whose whole
    purpose is that structure. Pairs are restricted to lags <= max_lag, which is
    where precipitation dependence lives and keeps the sum tractable.
    """
    T = len(obs)
    tot = 0.0
    for lag in range(1, min(max_lag, T - 1) + 1):
        o = np.abs(obs[:-lag] - obs[lag:]) ** p
        e = (np.abs(members[:, :-lag] - members[:, lag:]) ** p).mean(axis=0)
        w = 1.0 / lag                       # down-weight longer lags
        tot += w * float(np.sum((o - e) ** 2))
    return tot / T


def _spell(x: np.ndarray, wet: bool) -> int:
    best = cur = 0
    for v in (x >= WET_DAY_MM):
        cur = cur + 1 if bool(v) is wet else 0
        best = max(best, cur)
    return best


def structure_errors(members: np.ndarray, obs: np.ndarray) -> dict:
    """How well the ensemble reproduces temporal structure, not just marginals."""
    def acf1(x):
        x = np.asarray(x, float)
        if x.std() == 0:
            return np.nan
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])
    m_acf = np.nanmean([acf1(m) for m in members])
    return dict(
        acf1_err=float(abs(m_acf - acf1(obs))),
        CDD_err=float(abs(np.mean([_spell(m, False) for m in members])
                          - _spell(obs, False))),
        CWD_err=float(abs(np.mean([_spell(m, True) for m in members])
                          - _spell(obs, True))),
    )


def rank_histogram_chi2(members: np.ndarray, obs: np.ndarray) -> float:
    """Chi-square statistic for departure of the rank histogram from flat."""
    M = members.shape[0]
    ranks = (members < obs[None, :]).sum(axis=0)
    counts = np.bincount(ranks, minlength=M + 1)
    expected = counts.sum() / (M + 1)
    if expected <= 0:
        return float("nan")
    return float(((counts - expected) ** 2 / expected).sum())


# ────────────────────────────────────────────────────────────── generator
def generate(train: pd.DataFrame, n_days: int, months: np.ndarray, flags: dict,
             n_members: int, rng, templates: list[np.ndarray] | None = None
             ) -> np.ndarray:
    """Draw an ensemble for one held-out year under one configuration."""
    tr_p = train["PRECTOTCORR"].to_numpy(float)
    tr_m = train["month"].to_numpy(int)
    members = np.empty((n_members, n_days))

    for m in range(n_members):
        if flags["seasonal"]:
            draw = np.empty(n_days)
            for mo in np.unique(months):
                sel = months == mo
                pool = tr_p[tr_m == mo]
                pool = pool[np.isfinite(pool)]
                if len(pool) < 10:
                    pool = tr_p[np.isfinite(tr_p)]
                draw[sel] = _draw(pool, int(sel.sum()), flags, rng)
        else:
            pool = tr_p[np.isfinite(tr_p)]
            draw = _draw(pool, n_days, flags, rng)

        if flags["nqt"]:
            z = nqt_transform(draw)
            z = z * 0.98 + rng.normal(0, 0.2, len(z))       # meta-Gaussian step
            draw = nqt_inverse(z, tr_p[np.isfinite(tr_p)])
        if flags["qmap"]:
            draw = quantile_map(draw, tr_p[np.isfinite(tr_p)])
        members[m] = np.clip(draw, 0, None)

    if flags["shuffle"] and templates:
        # The Schaake shuffle reorders members to an OBSERVED daily trajectory,
        # so the template must be a real training year aligned by day of year,
        # not an arbitrary sequence. Using a sorted slice here would impose a
        # monotonic ordering and destroy the day-to-day correspondence.
        t = templates[rng.integers(0, len(templates))]
        if len(t) >= n_days:
            members = schaake_shuffle(members, t[:n_days])
    return members


def _draw(pool: np.ndarray, n: int, flags: dict, rng) -> np.ndarray:
    """One block of daily values from the chosen marginal model."""
    if not flags["occurrence"] and not flags["intensity"]:
        return rng.choice(pool, size=n, replace=True)        # plain resampling
    fit = fit_bernoulli_gamma(pool)
    wet = rng.random(n) < fit["p_wet"] if flags["occurrence"] else \
        rng.random(n) < (pool >= WET_DAY_MM).mean()
    out = np.zeros(n)
    k = int(wet.sum())
    if k == 0:
        return out
    if flags["intensity"] and np.isfinite(fit["shape"]) and np.isfinite(fit["scale"]):
        out[wet] = rng.gamma(fit["shape"], fit["scale"], k)
    else:
        src = fit["wet"] if len(fit["wet"]) else pool
        out[wet] = rng.choice(src, size=k, replace=True)
    return out


# ───────────────────────────────────────────────────────────── the ablation
def run_ablation(power: pd.DataFrame, cfg, n_members: int = 50,
                 min_days: int = 365, seed: int = 0, progress: bool = True
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-year-out ablation of the post-processing components."""
    rng = np.random.default_rng(seed)
    power = power.copy()
    power["month"] = power["date"].dt.month
    cfgs = configurations()
    rows = []

    for st, g in power.groupby("station"):
        years = [y for y, gg in g.groupby("hydro_year")
                 if gg["PRECTOTCORR"].notna().sum() >= min_days]
        for hy in years:
            te = g[g["hydro_year"] == hy]
            tr = g[(g["hydro_year"] != hy) & (g["hydro_year"].isin(years))]
            obs = te["PRECTOTCORR"].to_numpy(float)
            ok = np.isfinite(obs)
            obs = obs[ok]
            months = te["month"].to_numpy(int)[ok]
            if len(obs) < min_days:
                continue
            obs_h = _hist(obs, cfg)
            # candidate Schaake templates: the observed daily series of each
            # training year, which carries the real seasonal day-to-day shape
            templates = [gg["PRECTOTCORR"].to_numpy(float)
                         for yy, gg in tr.groupby("hydro_year")
                         if gg["PRECTOTCORR"].notna().sum() >= len(obs)]
            templates = [t[np.isfinite(t)] for t in templates]
            for cname, flags in cfgs.items():
                mem = generate(tr, len(obs), months, flags, n_members, rng,
                               templates)
                ens_h = _hist(mem.mean(axis=0), cfg)
                rows.append(dict(
                    station=st, hydro_year=int(hy), config=cname,
                    n_members=n_members,
                    CRPS=crps_empirical(mem, obs),
                    variogram=variogram_score(mem, obs),
                    **structure_errors(mem, obs),
                    Brier=brier_occurrence(mem, obs),
                    rank_chi2=rank_histogram_chi2(mem, obs),
                    PRCPTOT_err=float(abs(mem.sum(axis=1).mean() - obs.sum())),
                    wetdays_err=float(abs((mem >= WET_DAY_MM).sum(axis=1).mean()
                                          - (obs >= WET_DAY_MM).sum())),
                    RX1day_err=float(abs(mem.max(axis=1).mean() - obs.max())),
                    hist_KL=kl_divergence(ens_h, obs_h, cfg.eps),
                    hist_DB=bhattacharyya(ens_h, obs_h, cfg.eps),
                    hist_W1=w1_ordered(ens_h, obs_h),
                ))
        if progress:
            print(f"    post-processing ablation: {st} ({len(years)} held-out years)")

    res = pd.DataFrame(rows)
    return res, summarise(res)


def summarise(res: pd.DataFrame, metric: str = "CRPS") -> pd.DataFrame:
    """Per-configuration summary plus the degradation against the full model."""
    if res.empty:
        return pd.DataFrame()
    g = res.groupby("config")[metric]
    out = g.agg(n="size", mean="mean", sd="std", median="median",
                q05=lambda s: s.quantile(0.05),
                q95=lambda s: s.quantile(0.95)).reset_index()
    full = float(out.loc[out["config"] == "full", "mean"].iloc[0]) \
        if (out["config"] == "full").any() else np.nan
    out["delta_vs_full"] = out["mean"] - full
    out["pct_vs_full"] = 100 * out["delta_vs_full"] / full if full else np.nan
    return out.sort_values("mean")


def test_components(res: pd.DataFrame, metric: str = "CRPS") -> pd.DataFrame:
    """Paired tests of each leave-one-out configuration against the full model.

    Blocks are station-years, so the contrast is paired and Wilcoxon applies.
    Holm controls the family of component tests.
    """
    from .statistics import holm, rank_biserial
    if res.empty:
        return pd.DataFrame()
    piv = res.pivot_table(index=["station", "hydro_year"], columns="config",
                          values=metric)
    if "full" not in piv:
        return pd.DataFrame()
    rows = []
    for c in piv.columns:
        if c == "full":
            continue
        a, b = piv[c].to_numpy(float), piv["full"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 5 or np.allclose(a[ok], b[ok]):
            continue
        try:
            W, p = stats.wilcoxon(a[ok], b[ok])
        except ValueError:
            continue
        rows.append(dict(config=c, n=int(ok.sum()), W=float(W), p=float(p),
                         median_delta=float(np.median(a[ok] - b[ok])),
                         rank_biserial=rank_biserial(a[ok], b[ok]),
                         component=c.replace("minus_", "") if c.startswith("minus_") else "-"))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"])
        out["worse_than_full"] = (out["median_delta"] > 0) & (out["p_holm"] < 0.05)
        out["component_earns_its_place"] = out["worse_than_full"]
    return out.sort_values("median_delta", ascending=False)


def friedman_configs(res: pd.DataFrame, metric: str = "CRPS"):
    """Omnibus across configurations, with Nemenyi post-hoc."""
    from .statistics import friedman, nemenyi
    piv = res.pivot_table(index=["station", "hydro_year"], columns="config",
                          values=metric).dropna(axis=0, how="any")
    if piv.shape[1] < 3:
        return {}, float("nan"), pd.DataFrame()
    fr = friedman(piv)
    cd, pairs = nemenyi(piv)
    return fr, cd, pairs
