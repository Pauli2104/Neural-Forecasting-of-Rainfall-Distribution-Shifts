"""
XAI_Rfpipe.core - data layer.

Loading, integrity audit, hydrological-year aggregation, annual descriptors and
the distribution-shift targets.

Design notes
------------
* The hydrological year starts on 1 July, so the austral wet season sits in the
  middle of a year rather than across its boundary. Every downstream stage keys
  on `hydro_year`, defined as the calendar year in which that 1 July falls.
* NASA POWER writes -999 as its fill value. These are converted to NaN on load
  and counted in the audit; they are never silently imputed.
* The divergence targets are implemented from the definitions stated in the
  paper, not ported line-by-line from the exploratory notebooks. `verify_prop1`
  checks numerically that EMD and W1 coincide on this support, which is both a
  test of the implementation and the paper's Proposition 1.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

FILL = -999.0
HYDRO_START_MONTH = 7
WET_DAY_MM = 1.0           # ETCCDI wet-day threshold


# ───────────────────────────────────────────────────────────────── config
@dataclass(frozen=True)
class Station:
    name: str
    lat: float
    lon: float
    province: str


@dataclass
class Config:
    root: Path = Path(".")
    stations: tuple = (
        Station("Catamarca", -28.454637, -65.764005, "Catamarca"),
        Station("Cordoba", -31.446667, -64.512592, "Cordoba"),
        Station("Lamadrid", -27.655100, -65.245400, "Tucuman"),
    )
    # target construction
    ref_years: int = 10        # length of the reference window, hydrological years
    win_years: int = 5         # length of the moving comparison window
    n_bins: int = 40           # histogram bins for the density-based divergences
    bin_max_mm: float = 60.0   # upper edge; heavier days fall in the last bin
    eps: float = 1e-9          # smoothing floor for KL / Bhattacharyya
    quantiles: tuple = (0.5, 0.75, 0.9)
    # experiment
    lag_windows: tuple = (2, 3, 4, 5, 6)
    default_lag: int = 4
    seeds: tuple = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    test_years: int = 8
    val_years: int = 6
    n_boot: int = 2000
    conformal_alpha: float = 0.10
    out: Path = Path("results")

    @property
    def station_names(self):
        return [s.name for s in self.stations]

    def dump(self, path):
        d = {k: (str(v) if isinstance(v, Path) else v)
             for k, v in self.__dict__.items() if k != "stations"}
        d["stations"] = [s.__dict__ for s in self.stations]
        Path(path).write_text(json.dumps(d, indent=2, default=str))


# ─────────────────────────────────────────────────────────────────── io
def _demojibake(s: str) -> str:
    """The flood catalogues were written with mixed cp1252/latin-1 encodings."""
    if not isinstance(s, str):
        return s
    for bad, good in (("¡", "í"), ("¢", "ó"), ("­", "í"), ("Ï", "ñ"), ("?", ""),
                      ("¤", "ñ"), (" ", " ")):
        s = s.replace(bad, good)
    return unicodedata.normalize("NFC", s).strip()


def load_power(path, station: str) -> pd.DataFrame:
    """Load one NASA POWER daily export. Fill values become NaN."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.replace(FILL, np.nan)
    df["date"] = pd.to_datetime(
        dict(year=df["YEAR"], month=df["MO"], day=df["DY"]), errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["station"] = station
    df["hydro_year"] = np.where(df["date"].dt.month >= HYDRO_START_MONTH,
                                df["date"].dt.year, df["date"].dt.year - 1)
    return df


def load_floods(path, station: str) -> pd.DataFrame:
    """Load one flood catalogue. Returns one row per documented event."""
    raw = Path(path).read_bytes().decode("utf-8", errors="replace")
    rows = []
    for line in raw.splitlines()[1:]:
        parts = line.split(";")
        if len(parts) < 2 or not parts[0].strip().isdigit():
            continue
        year, month = int(parts[0]), int(parts[1])
        detail = _demojibake(parts[3]) if len(parts) > 3 else ""
        rows.append(dict(station=station, year=year, month=month,
                         hydro_year=year if month >= HYDRO_START_MONTH else year - 1,
                         detail=detail))
    return pd.DataFrame(rows)


def load_all(cfg: Config):
    power = pd.concat([load_power(cfg.root / "data" / "power" / f"{s.name}.csv", s.name)
                       for s in cfg.stations], ignore_index=True)
    floods = pd.concat([load_floods(cfg.root / "data" / "floods" / f"{s.name}.csv", s.name)
                        for s in cfg.stations], ignore_index=True)
    return power, floods


# ──────────────────────────────────────────────────────────────── audit
def audit(power: pd.DataFrame, floods: pd.DataFrame) -> pd.DataFrame:
    """Per-station integrity report. Reports; never repairs."""
    out = []
    for st, g in power.groupby("station"):
        span = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        missing_dates = len(span) - g["date"].nunique()
        numeric = g.select_dtypes(include=[np.number])
        rec = dict(
            station=st,
            first=str(g["date"].min().date()), last=str(g["date"].max().date()),
            n_days=len(g),
            calendar_gaps=int(missing_dates),
            duplicate_dates=int(g["date"].duplicated().sum()),
            n_variables=int(numeric.shape[1]),
            nan_cells=int(numeric.isna().sum().sum()),
            precip_nan=int(g["PRECTOTCORR"].isna().sum()),
            precip_negative=int((g["PRECTOTCORR"] < 0).sum()),
            precip_max_mm=float(np.nanmax(g["PRECTOTCORR"])),
            complete_hydro_years=int(
                (g.groupby("hydro_year").size() >= 365).sum()),
            flood_events=int((floods["station"] == st).sum()),
            flood_years=int(floods.loc[floods["station"] == st, "hydro_year"].nunique()),
        )
        out.append(rec)
    return pd.DataFrame(out)


# ──────────────────────────────────────────────────── annual descriptors
def _spell(mask: np.ndarray, wet: bool) -> int:
    """Longest run of wet (or dry) days."""
    best = cur = 0
    for m in mask:
        cur = cur + 1 if bool(m) is wet else 0
        best = max(best, cur)
    return int(best)


def descriptors(power: pd.DataFrame, min_days: int = 365) -> pd.DataFrame:
    """ETCCDI-style annual descriptors per station-hydrological year."""
    rows = []
    for (st, hy), g in power.groupby(["station", "hydro_year"]):
        p = g["PRECTOTCORR"].to_numpy(dtype=float)
        p = p[~np.isnan(p)]
        if len(p) < min_days:
            continue
        wet = p >= WET_DAY_MM
        wetvals = p[wet]
        s = pd.Series(p)
        q95 = np.quantile(wetvals, 0.95) if len(wetvals) else np.nan
        q99 = np.quantile(wetvals, 0.99) if len(wetvals) else np.nan
        rows.append(dict(
            station=st, hydro_year=int(hy), n_days=len(p),
            PRCPTOT=float(p.sum()),
            wet_days=int(wet.sum()),
            SDII=float(wetvals.mean()) if len(wetvals) else 0.0,
            RX1day=float(p.max()),
            RX3day=float(s.rolling(3).sum().max()),
            RX5day=float(s.rolling(5).sum().max()),
            CDD=_spell(wet, False), CWD=_spell(wet, True),
            R10mm=int((p >= 10).sum()), R20mm=int((p >= 20).sum()),
            R95p=float(wetvals[wetvals >= q95].sum()) if len(wetvals) else 0.0,
            R99p=float(wetvals[wetvals >= q99].sum()) if len(wetvals) else 0.0,
            wet_fraction=float(wet.mean()),
            mean_T2M=float(np.nanmean(g["T2M"])),
            mean_RH2M=float(np.nanmean(g["RH2M"])),
            mean_PS=float(np.nanmean(g["PS"])),
        ))
    return pd.DataFrame(rows).sort_values(["station", "hydro_year"]).reset_index(drop=True)


# ─────────────────────────────────────────────── distribution-shift targets
def _hist(p: np.ndarray, cfg: Config) -> np.ndarray:
    """Normalised histogram of wet-day amounts on a fixed support."""
    edges = np.linspace(0.0, cfg.bin_max_mm, cfg.n_bins + 1)
    v = np.clip(p[p >= WET_DAY_MM], 0, cfg.bin_max_mm)
    h, _ = np.histogram(v, bins=edges)
    h = h.astype(float)
    tot = h.sum()
    return h / tot if tot > 0 else np.full_like(h, 1.0 / len(h))


def kl_divergence(p, q, eps):
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    p = p / p.sum(); q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def bhattacharyya(p, q, eps):
    p = np.clip(p, eps, None); q = np.clip(q, eps, None)
    p = p / p.sum(); q = q / q.sum()
    bc = float(np.sum(np.sqrt(p * q)))
    return float(-math.log(min(max(bc, eps), 1.0)))


def emd_ordered(p, q):
    """Earth mover's distance on a unit-spaced ordered support.

    This is the discrete transport cost between two histograms whose bins are
    one unit apart, i.e. the L1 norm of the cumulative difference.
    """
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum(); q = q / q.sum()
    return float(np.abs(np.cumsum(p - q)).sum())


def w1_ordered(p, q):
    """Wasserstein-1 between the same two histograms, via scipy on bin indices."""
    idx = np.arange(len(p), dtype=float)
    return float(wasserstein_distance(idx, idx, u_weights=p, v_weights=q))


def build_targets(power: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per station-hydrological year, from `ref_years` onward.

    The reference window is the first `ref_years` complete hydrological years at
    that station; the moving window is the `win_years` years ending at `hy`.
    Divergences are computed between the two wet-day amount distributions.
    """
    rows = []
    for st, g in power.groupby("station"):
        years = sorted(y for y, gg in g.groupby("hydro_year")
                       if gg["PRECTOTCORR"].notna().sum() >= 365)
        if len(years) < cfg.ref_years + cfg.win_years:
            continue
        daily = {y: g.loc[g["hydro_year"] == y, "PRECTOTCORR"].to_numpy(float)
                 for y in years}
        ref = np.concatenate([daily[y] for y in years[:cfg.ref_years]])
        ref = ref[~np.isnan(ref)]
        h_ref = _hist(ref, cfg)
        for i, hy in enumerate(years):
            if i < cfg.win_years - 1:
                continue
            win = np.concatenate([daily[y] for y in years[i - cfg.win_years + 1:i + 1]])
            win = win[~np.isnan(win)]
            h_win = _hist(win, cfg)
            rows.append(dict(
                station=st, hydro_year=int(hy),
                D_KL=kl_divergence(h_win, h_ref, cfg.eps),
                D_B=bhattacharyya(h_win, h_ref, cfg.eps),
                W1=w1_ordered(h_win, h_ref),
                EMD=emd_ordered(h_win, h_ref),
            ))
    return pd.DataFrame(rows).sort_values(["station", "hydro_year"]).reset_index(drop=True)


def verify_prop1(targets: pd.DataFrame) -> dict:
    """Proposition 1: EMD == W1 on a unit-spaced ordered support."""
    a = targets["EMD"].to_numpy(float)
    b = targets["W1"].to_numpy(float)
    diff = np.abs(a - b)
    r = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")
    return dict(n=int(len(a)), pearson_r=r, max_abs_diff=float(diff.max()),
                mean_abs_diff=float(diff.mean()),
                identical=bool(diff.max() < 1e-9),
                independent_targets=3 if diff.max() < 1e-9 else 4)


def flood_labels(floods: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Binary flood-year label aligned to the target rows."""
    key = set(zip(floods["station"], floods["hydro_year"]))
    out = targets[["station", "hydro_year"]].copy()
    out["flood"] = [int((s, h) in key) for s, h in zip(out["station"], out["hydro_year"])]
    return out
