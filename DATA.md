# Data provenance and terms

Three kinds of file ship with this repository, under different terms from the MIT
licence that covers the code. This file states where each came from, what it may
be used for, and what it does *not* establish.

---

## `data/power/*.csv` — daily meteorology

**Source.** NASA Prediction Of Worldwide Energy Resources (POWER) project,
Langley Research Center, funded through the NASA Earth Science Division. The
series are MERRA-2-derived grid-cell values, not station observations: each file
is the single grid cell containing the named location, so "Catamarca" throughout
this repository means *the POWER cell containing the Catamarca valley site*, and
every spatial claim in the paper is bounded by that cell's footprint.

| file | cell centre | province | rows | period |
|---|---|---|---|---|
| `Catamarca.csv` | 28.4546 S, 65.7640 W | Catamarca | 15 888 | 1981-01-01 → 2024-07-01 |
| `Cordoba.csv` | 31.4467 S, 64.5126 W | Córdoba | 15 888 | 1981-01-01 → 2024-07-01 |
| `Lamadrid.csv` | 27.6551 S, 65.2454 W | Tucumán | 15 888 | 1981-01-01 → 2024-07-01 |

The record ends one day into the 2024 hydrological year, so the last complete
hydrological year is 2023 (1 July 2023 → 30 June 2024).

Columns are POWER's own names (`PRECTOTCORR`, `T2M`, `RH2M`, …). POWER writes
`-999` as its fill value; `XAI_Rfpipe.core` converts those to NaN on load and counts
them in `01_audit.csv`. They are never imputed.

**Attribution required.** POWER asks that both of the following appear in any
work product that uses the data:

> The data was obtained from National Aeronautics and Space Administration
> (NASA) Langley Research Center's Prediction Of Worldwide Energy Resources
> (POWER) project, funded through the NASA Earth Science Division.

> The data was obtained from the POWER Project's Daily 2.x.x version on
> YYYY/MM/DD.

Fill in the version and access date actually used before citing. POWER states no
formal use restrictions but makes two requests: notify
`larc-power-project@mail.nasa.gov` of published work, and notify the project when
POWER data is passed on to other researchers. **Redistributing these CSVs in a
public repository is such a pass-on, so send that notification.**

See <https://power.larc.nasa.gov/docs/referencing/> for the current wording.

---

## `data/floods/*.csv` — documented flood catalogue

**Source.** A catalogue of 43 damaging flood and river-overflow events affecting
the three study areas, compiled by the authors from published accounts and
regional reporting. Per station: Catamarca 20 (1973–2023), Córdoba 9
(1985–2023), La Madrid 14 (1992–2023).

**Format.** Semicolon-separated, Spanish column names, year and month only. Some
files carry a UTF-8 BOM and cp1252-encoded accents; `XAI_Rfpipe.core.load_floods`
handles both rather than requiring the files to be cleaned first, so the shipped
files are byte-identical to the working copies.

**What the catalogue does and does not support.** It records month and year, so
it supports annual labels — a hydrological year either contains a documented
event or it does not — and nothing finer. It cannot support warning-horizon or
lead-time claims, and the pipeline does not compute any.

**Spatial support is weaker than it looks.** Every locality named in the
catalogue was resolved against the Argentine national gazetteer (the Georef
service of the Instituto Geográfico Nacional) with matching constrained to the
province of the parent cell. Thirty-two of the 43 events carry at least one
resolvable point; the remaining eleven name dams, valleys or whole regions rather
than settlements, and were left unplaced rather than approximated to a nearby
town. Of the resolvable place-mentions, only a minority fall within 25 km of the
cell centre whose series supplies the predictors — distances run past 100 km.
Treat the catalogue as regional evidence of flooding, not as a label attached to
the grid cell. The paper reports this as a limitation; it is repeated here so
that nobody reusing these files assumes a tighter correspondence than exists.

The gazetteer resolution itself is not part of this pipeline: it belongs to the
manuscript's figure sources. These CSVs are the catalogue as transcribed.

---

## `results/` — generated artefacts

Everything in `results/` is produced by `run_pipeline.py` from the two data sets
above and is regenerable from them; it is committed so that a reader can inspect
the numbers behind the paper without a 70-minute run. Deleting the directory and
rerunning reproduces it.

`results/Fig5_xai_heatmap.{png,pdf}` is a figure in the paper. Figures in the
published article are subject to the publisher's terms; the copies here are the
authors' own output and are released with the repository.

---

## What is *not* here

* The exploratory notebooks the study began from. The pipeline is written from
  the definitions stated in the paper, not ported from them — see the closing
  section of the README.
* The cartographic sources behind the study-area and methodology figures
  (OpenStreetMap tiles under ODbL, Natural Earth, IGN gazetteer responses).
  Those figures are built in the manuscript's own source tree, and OSM's
  attribution requirement is a licence condition that travels with the figure,
  not with this code.
* Any third-party flood database. The catalogue here is the authors' own
  compilation.
