# Neural-Forecasting-of-Rainfall-Distribution-Shifts

This repository contains the notebook, datasets, and experimental outputs associated with the study:

**Neural-Forecasting-of-Rainfall-Distribution-Shifts**

The project evaluates whether neural sequence models improve annual rainfall-distribution shift forecasting over conventional statistical and machine-learning baselines, and whether deterministic uncertainty measures can be used to quantify rainfall-distribution shifts in data-scarce hydrometeorological settings.

## Repository link

```text
https://github.com/Pauli2104/Neural-Forecasting-of-Rainfall-Distribution-Shifts
```

## Research focus

Rainfall-derived proxy targets are useful when discharge, soil-moisture, and dense gauge observations are limited. However, forecasting a rainfall proxy is not the same as forecasting a damaging flood. This repository therefore focuses on leakage-aware rainfall-distribution shift forecasting rather than operational flood occurrence prediction.

The study addresses two research questions:

1. **RQ1:** To what extent do LSTM, GRU, TCN, and TFT-style attention models improve rainfall-distribution shift forecasting compared with conventional machine-learning and statistical baselines?
2. **RQ2:** To what extent are deterministic uncertainty methods useful to quantify rainfall-distribution shifts?

## Study area and data

The experiments use daily precipitation records for three stations in northwestern and central Argentina:

- Catamarca
- Cordoba
- La Madrid

Daily precipitation is aggregated into hydrological-year descriptors. The rainfall-distribution shift targets are derived from deterministic divergence and distance measures computed relative to a station-specific reference distribution.

## Deterministic rainfall-shift targets

The repository includes experiments using the following deterministic uncertainty indicators:

- Kullback--Leibler divergence
- Wasserstein-style distance
- Bhattacharyya distance
- Earth Mover's Distance

These indicators describe rainfall timing, spread, overlap, and transport-like redistribution. They should be interpreted as deterministic rainfall-distribution shift descriptors, not as calibrated flood-probability estimates.

## Models compared

The benchmark compares conventional statistical and machine-learning baselines with neural sequence models.

### Conventional baselines

- Persistence
- Climatology mean
- Linear regression
- Ridge regression
- Random forest
- Multilayer perceptron using flattened lagged inputs
- Exponential smoothing
- Bounded ARIMA / AutoARIMA-style models
- XGBoost, when available

### Neural sequence models

- LSTM
- GRU
- TCN
- Compact TFT-style attention model

## Repository structure

Recommended structure:

```text
Neural-Forecasting-of-Rainfall-Distribution-Shifts/
│
├── notebooks/
│   └── Flood_RQ1_RQ2_Experiments.ipynb
│
├── data/
│   ├── 2024_07_Catamarca.csv
│   ├── 2024_07_Cordoba.csv
│   └── 2024_07_Lamadrid.csv
│
├── results/
│   ├── Neural_vs_Baseline_RQ1_summary.csv
│   ├── RQ1_station_summary.csv
│   └── deterministic_uncertainty_summary.csv
│
├── requirements.txt
└── README.md
```

## How to run the notebook

Clone the repository:

```bash
git clone https://github.com/Pauli2104/Neural-Forecasting-of-Rainfall-Distribution-Shifts.git
cd Neural-Forecasting-of-Rainfall-Distribution-Shifts
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open Jupyter:

```bash
jupyter notebook
```

Then run the notebook inside the `notebooks/` folder from top to bottom.

## Suggested requirements.txt

```text
numpy
pandas
scipy
scikit-learn
matplotlib
statsmodels
xgboost
torch
jupyter
notebook
```

If the repository contains only a Scikit-learn implementation, remove `torch` from the requirements file.

## Main outputs

The notebook exports:

- Daily data audit tables
- Hydrological-year rainfall descriptors
- Deterministic uncertainty summaries
- Model predictions
- Regression metrics
- Best neural model versus best conventional baseline comparisons
- Paper-ready result tables

## Main finding

The results show that neural gains are limited under small annual samples. Using MASE, the best neural model improves over the best conventional baseline in only one of twelve station-target configurations. Deterministic uncertainty measures remain useful as interpretable rainfall-distribution shift descriptors, but they should not be interpreted as direct flood-occurrence predictions.

## Reproducibility notes

The experiments use chronological train-validation-test splits to avoid temporal leakage. Input scaling is fitted only on the training block and then applied to validation and test data. The default setup uses a six-year lookback window and evaluates model performance on held-out hydrological years.

Because the supervised annual sample is small, the results should be interpreted as a reproducible comparative experiment rather than as an operational hydrological forecasting system.

## Citation

If you use this repository, please cite the related paper and the previous ISCMI study:

```bibtex
@inproceedings{rivero2025flood,
  author    = {Rivero, C. R. and others},
  title     = {Flood occurrence prediction using Monte Carlo methods and machine learning for mitigating Climate Impact in Northwestern Argentina},
  booktitle = {Proceedings of the 2025 12th International Conference on Soft Computing \& Machine Intelligence (ISCMI)},
  address   = {Rio de Janeiro, Brazil},
  year      = {2025},
  pages     = {98--102},
  doi       = {10.1109/ISCMI67495.2025.11358627}
}
```

## License

Add a license before public reuse. Recommended options:

- MIT License for code
- CC BY 4.0 for documentation and paper materials
- A separate data license if dataset redistribution is restricted

## Contact

For questions about the repository or paper materials, please contact the repository owner through GitHub.
