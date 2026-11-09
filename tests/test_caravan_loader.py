#!/usr/bin/env python3
"""
Prove the Caravan loader against a synthetic tree with Caravan's real schema.

The archive is 25 GB and is not redistributed, so the loader is tested against
a fixture built to the published layout and column names rather than against
the data itself. That verifies the parsing, the eligibility logic, the
stratified draw and the unit guard; it cannot verify the units of any
particular Caravan build, which is exactly why the guard raises rather than
rescales.

    python tests/test_caravan_loader.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XAI_Rfpipe import caravan

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAIL.append(name)


def make_basin(path: Path, years: int, seed: int, metres: bool = False,
               missing: float = 0.0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1981-01-01", periods=int(years * 365.25), freq="D")
    wet = rng.random(len(dates)) < 0.25
    p = np.where(wet, rng.gamma(0.8, 9.0, len(dates)), 0.0)
    if metres:
        p = p / 1000.0
    if missing:
        p[rng.random(len(dates)) < missing] = np.nan
    pd.DataFrame({
        "date": dates,
        "total_precipitation_sum": p,
        "temperature_2m_mean": 15 + 8 * np.sin(np.arange(len(dates)) / 58.1),
        "temperature_2m_min": 10 + 8 * np.sin(np.arange(len(dates)) / 58.1),
        "temperature_2m_max": 21 + 8 * np.sin(np.arange(len(dates)) / 58.1),
        "streamflow": rng.gamma(2.0, 1.0, len(dates)),
    }).to_csv(path, index=False)


with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "Caravan"
    for prefix, n, yrs in [("camels_cl", 6, 40), ("camels_br", 5, 38), ("camels", 4, 20)]:
        d = root / "timeseries" / "csv" / prefix
        d.mkdir(parents=True)
        for i in range(n):
            make_basin(d / f"{prefix}_{1000+i}.csv", yrs, seed=i + hash(prefix) % 97)
    # one basin in the wrong units, and one too gappy to use
    make_basin(root / "timeseries" / "csv" / "camels_cl" / "camels_cl_9999.csv",
               40, seed=5, metres=True)
    make_basin(root / "timeseries" / "csv" / "camels_br" / "camels_br_8888.csv",
               38, seed=6, missing=0.20)

    print("discover()")
    cat = caravan.discover(root, min_years=30)
    check("every basin catalogued", len(cat) == 17, f"n={len(cat)}")
    check("short records are not eligible",
          not cat.loc[cat.prefix == "camels", "eligible"].any())
    check("long complete records are eligible",
          cat.loc[cat.gauge_id == "camels_cl_1000", "eligible"].item())
    gappy = cat.loc[cat.gauge_id == "camels_br_8888"]
    check("a gappy record is excluded", not bool(gappy["eligible"].item()),
          f"missing={gappy['missing_fraction'].item():.2f}")

    print("\nselect()")
    pick = caravan.select(cat, n=6, seed=0)
    check("draws the requested number", len(pick) == 6, f"n={len(pick)}")
    check("draw is reproducible", pick == caravan.select(cat, n=6, seed=0))
    check("draw is stratified across source datasets",
          len({g.rsplit('_', 1)[0] for g in pick}) > 1,
          f"prefixes={sorted({g.rsplit('_',1)[0] for g in pick})}")
    check("only eligible basins drawn",
          set(pick) <= set(cat.loc[cat.eligible, "gauge_id"]))

    print("\nload_basin()")
    d = caravan.load_basin(root, "camels_cl_1000")
    check("columns renamed for the pipeline",
          list(d.columns)[:4] == ["YEAR", "MO", "DY", "PRECTOTCORR"],
          str(list(d.columns)))
    check("calendar fields are consistent",
          bool(((d.MO >= 1) & (d.MO <= 12) & (d.DY >= 1) & (d.DY <= 31)).all()))
    try:
        caravan.load_basin(root, "camels_cl_9999")
        check("a unit error is refused, not rescaled", False, "no exception raised")
    except ValueError as e:
        check("a unit error is refused, not rescaled", "mm/yr" in str(e))

    print("\nbuild_subset()")
    out = Path(td) / "power"
    man = caravan.build_subset(root, out, pick[:3])
    check("one CSV written per basin", len(list(out.glob('*.csv'))) == 4,
          "3 basins + manifest")
    check("manifest records the subset", len(man) == 3)
    first = pd.read_csv(out / f"{pick[0]}.csv")
    check("written file reloads with the pipeline's columns",
          {"YEAR", "MO", "DY", "PRECTOTCORR"} <= set(first.columns))

print(f"\n{'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
