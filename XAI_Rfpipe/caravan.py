"""
XAI_Rfpipe.caravan - read a Caravan subset into the format this pipeline expects.

Caravan (Kratzert et al. 2023, doi:10.1038/s41597-023-01975-w, CC BY 4.0) is a
harmonised large-sample dataset of daily ERA5-Land forcings and discharge for
several thousand basins, assembled from CAMELS-US, CAMELS-AUS, CAMELS-BR,
CAMELS-CL, CAMELS-GB, HYSETS and LamaH-CE. Its forcings are reanalysis values
aggregated over a basin polygon, which is the same kind of object as the
reanalysis grid cell used elsewhere in this pipeline: swapping it in changes
the region and the record length while leaving the construct alone.

This module converts a Caravan subset into the daily CSV layout the rest of
the pipeline already reads, so nothing downstream needs to know where the data
came from.

Expected input layout, as published:

    <root>/timeseries/csv/<prefix>/<gauge_id>.csv

with a `date` column and, among others, `total_precipitation_sum`,
`temperature_2m_mean`, `temperature_2m_min`, `temperature_2m_max` and
`streamflow`. Gauge ids are `<prefix>_<id>`.

The archive is not redistributed here and is not downloaded by this code. Fetch
it from the Zenodo record, point `--root` at the unpacked directory, and cite
it as its licence requires.

UNITS ARE CHECKED, NOT ASSUMED. `load_basin` refuses a series whose wet-day
magnitudes are not consistent with millimetres per day rather than silently
rescaling, because a factor of 1000 in precipitation would pass through every
later stage as a plausible-looking number.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Caravan column -> the name this pipeline's loaders use
COLUMN_MAP = {
    "total_precipitation_sum": "PRECTOTCORR",
    "temperature_2m_mean": "T2M",
    "temperature_2m_min": "T2M_MIN",
    "temperature_2m_max": "T2M_MAX",
}
HYDRO_START_MONTH = 7
# A daily series in mm/day has a mean ANNUAL total in roughly this band - from
# hyper-arid to the wettest gauged places on earth. In metres/day the same
# series lands three orders of magnitude lower. The test is deliberately not
# keyed on a wet-day threshold: a metre-scale series has almost no days above
# 1 mm, so a wet-day statistic would skip the very case this guard exists for.
MM_PER_YEAR = (20.0, 15000.0)


def discover(root: str | Path, prefix: str | None = None,
             min_years: int = 30, max_missing: float = 0.02) -> pd.DataFrame:
    """Catalogue the basins in a Caravan tree that are long and complete enough.

    Returns one row per candidate basin with its record span, the number of
    complete hydrological years and its missing-value fraction, sorted by
    record length. Nothing is selected here: selection is a decision for the
    caller, and should be made before any model is fit.
    """
    root = Path(root)
    base = root / "timeseries" / "csv"
    if not base.exists():
        raise FileNotFoundError(
            f"{base} not found. Point --root at the unpacked Caravan directory, "
            "the one containing 'timeseries/'.")
    prefixes = [p for p in sorted(base.iterdir()) if p.is_dir()
                and (prefix is None or p.name == prefix)]
    rows = []
    for pdir in prefixes:
        for f in sorted(pdir.glob("*.csv")):
            try:
                d = pd.read_csv(f, usecols=["date", "total_precipitation_sum"],
                                parse_dates=["date"])
            except (ValueError, KeyError):
                continue
            if d.empty:
                continue
            p = d["total_precipitation_sum"]
            hy = d["date"].dt.year - (d["date"].dt.month < HYDRO_START_MONTH)
            counts = hy.value_counts()
            complete = int((counts >= 365).sum())
            rows.append(dict(
                gauge_id=f.stem, prefix=pdir.name,
                start=d["date"].min().date(), end=d["date"].max().date(),
                n_days=len(d), complete_hydro_years=complete,
                missing_fraction=round(float(p.isna().mean()), 4),
                wet_day_p99=round(float(np.nanpercentile(p[p > 1.0], 99))
                                  if (p > 1.0).any() else np.nan, 2)))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    ok = ((out["complete_hydro_years"] >= min_years)
          & (out["missing_fraction"] <= max_missing))
    out["eligible"] = ok
    return out.sort_values(["eligible", "complete_hydro_years"],
                           ascending=[False, False]).reset_index(drop=True)


def load_basin(root: str | Path, gauge_id: str, prefix: str | None = None) -> pd.DataFrame:
    """One basin's daily series, renamed to this pipeline's columns.

    Raises on a unit mismatch rather than rescaling: see the module docstring.
    """
    root = Path(root)
    pref = prefix or gauge_id.rsplit("_", 1)[0]
    f = root / "timeseries" / "csv" / pref / f"{gauge_id}.csv"
    if not f.exists():
        raise FileNotFoundError(f)
    d = pd.read_csv(f, parse_dates=["date"])
    missing = [c for c in COLUMN_MAP if c not in d.columns]
    if "total_precipitation_sum" in missing:
        raise KeyError(f"{f} has no total_precipitation_sum column; "
                       f"found {list(d.columns)[:8]}")
    d = d.rename(columns=COLUMN_MAP)

    p = d["PRECTOTCORR"]
    n_years = len(p) / 365.25
    if n_years >= 1.0 and p.notna().any():
        annual = float(np.nansum(p)) / n_years
        lo, hi = MM_PER_YEAR
        if not (lo <= annual <= hi):
            raise ValueError(
                f"{gauge_id}: mean annual precipitation is {annual:.4g}, outside "
                f"the range expected for mm/day inputs ({lo}-{hi} mm/yr). Check "
                "the units of total_precipitation_sum in this Caravan build "
                "before using it; this loader will not rescale silently.")
    d["YEAR"] = d["date"].dt.year
    d["MO"] = d["date"].dt.month
    d["DY"] = d["date"].dt.day
    keep = ["YEAR", "MO", "DY"] + [c for c in COLUMN_MAP.values() if c in d.columns]
    return d[keep]


def build_subset(root: str | Path, out_dir: str | Path, gauge_ids: list[str],
                 prefix: str | None = None) -> pd.DataFrame:
    """Write the chosen basins as daily CSVs the pipeline's loader can read.

    One file per basin, named for its gauge id, in the layout `load_power`
    expects. Returns a manifest, which is written beside the files so that the
    subset used is recoverable from the artefacts alone.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for g in gauge_ids:
        d = load_basin(root, g, prefix)
        d.to_csv(out_dir / f"{g}.csv", index=False)
        rows.append(dict(gauge_id=g, rows=len(d),
                         first=f"{d.YEAR.iloc[0]}-{d.MO.iloc[0]:02d}-{d.DY.iloc[0]:02d}",
                         last=f"{d.YEAR.iloc[-1]}-{d.MO.iloc[-1]:02d}-{d.DY.iloc[-1]:02d}"))
    man = pd.DataFrame(rows)
    man.to_csv(out_dir / "_subset_manifest.csv", index=False)
    return man


def select(catalogue: pd.DataFrame, n: int = 36, seed: int = 0,
           spread_across_prefixes: bool = True) -> list[str]:
    """Choose `n` eligible basins reproducibly.

    Selection is at random among eligible basins rather than by record length,
    because choosing the longest records would select for exactly the
    conditions under which a comparison resolves best. With
    `spread_across_prefixes`, the draw is stratified by source dataset so the
    subset is not one country.
    """
    el = catalogue[catalogue["eligible"]] if "eligible" in catalogue else catalogue
    if el.empty:
        return []
    rng = np.random.default_rng(seed)
    if not spread_across_prefixes:
        idx = rng.choice(len(el), size=min(n, len(el)), replace=False)
        return el.iloc[sorted(idx)]["gauge_id"].tolist()
    # round-robin over source datasets, drawing one basin at a time. Indexed by
    # position because two groups can compare equal as frames.
    groups = [g.reset_index(drop=True) for _, g in el.groupby("prefix")]
    picks: list[str] = []
    while len(picks) < n and any(len(g) for g in groups):
        for j, g in enumerate(groups):
            if len(picks) >= n or not len(g):
                continue
            i = int(rng.integers(0, len(g)))
            picks.append(str(g.iloc[i]["gauge_id"]))
            groups[j] = g.drop(index=i).reset_index(drop=True)
    return sorted(picks)
