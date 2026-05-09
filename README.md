# Formula 1 Championship Forecasting with Distributed Big Data Analytics

This project builds an end-to-end Formula 1 analytics and prediction pipeline using Apache Spark, Spark MLlib, PyTorch, and Python visualization tools. It transforms raw F1 race data into distributed feature datasets, trains championship and race-winner prediction models, and generates analytics tables and visual reports.

## Project Overview

The goal is to forecast Formula 1 outcomes using race history, driver momentum, constructor strength, reliability, qualifying performance, and circuit-specific trends.

The project includes:

- Distributed Spark-based feature engineering
- Rolling and cumulative driver performance features
- Constructor dominance and momentum features
- Circuit-specific historical performance features with reduced future-data leakage
- Championship prediction using a PyTorch Transformer
- Race winner prediction using Spark MLlib RandomForest
- Visual analytics for driver momentum, constructor dominance, championship probability, and DNF trends
- Final CSV result-table generation for reporting

## Repository Structure

```text
.
|-- config.py
|-- requirements.txt
|-- src/
|   |-- fetch_data.py
|   |-- feature_engineering.py
|   |-- champ_model.py
|   |-- race_winner_model.py
|   |-- visualize.py
|   `-- build_results_tables.py
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- features/
`-- results/
```

`data/` and `results/` are generated/local directories and are intentionally excluded from Git.

## Tech Stack

- Python
- Apache Spark / PySpark
- Spark MLlib
- PyTorch
- scikit-learn
- pandas
- NumPy
- Matplotlib
- XGBoost

## Setup

Create and activate a virtual environment:

```bash
python -m venv f1_env
f1_env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For Spark execution, make sure Java is installed and available on your system path.

## Pipeline

### 1. Feature Engineering

```bash
python src/feature_engineering.py
```

This creates distributed feature datasets:

- `data/features/champ_features.parquet`
- `data/features/race_features.parquet`

It also generates analytics parquet tables in `results/`.

### 2. Championship Prediction

```bash
python src/champ_model.py
```

This trains a PyTorch Transformer model that predicts championship outcomes from early-season race sequences.

Outputs include:

- `results/championship_predictions.csv`
- `results/champ_transformer.pt`
- training and prediction plots

### 3. Race Winner Prediction

```bash
python src/race_winner_model.py
```

This trains a Spark MLlib RandomForest classifier using race-level features.

Outputs include:

- `results/race_winner_predictions.parquet`
- `results/race_winner_random_forest/`

### 4. Visual Analytics

```bash
python src/visualize.py
```

Generated charts:

- Driver momentum
- Constructor dominance
- Championship probability curves
- Circuit DNF analysis

### 5. Final Results Tables

```bash
python src/build_results_tables.py
```

Creates final CSV outputs:

- `results/championship_predictions.csv`
- `results/race_predictions.csv`
- `results/feature_importance.csv`
- `results/constructor_dominance.csv`
- `results/driver_statistics.csv`

## Feature Engineering Highlights

The Spark pipeline includes:

- Rolling average finish
- Rolling points and wins
- Momentum score
- Finish consistency score
- Podium rate and win rate
- Reliability score
- Constructor cumulative points
- Constructor rolling points
- Constructor momentum
- Career race count
- Historical circuit performance
- Spark caching and repartitioned parquet writes
- Feature statistics profiling
- Execution timing logs

## Notes

- Generated data, trained models, plots, and result tables are ignored by Git to keep the repository lightweight.
- Raw datasets should be placed under `data/raw/` or regenerated using the project data-fetching scripts.
- Some race-winner features, such as `positions_gained`, are post-race features and should be removed for a strict pre-race prediction setup.

## Recommended Run Order

```bash
python src/feature_engineering.py
python src/champ_model.py
python src/race_winner_model.py
python src/visualize.py
python src/build_results_tables.py
```

## Project Title

**Deep Learning and Distributed Big Data Analytics for Formula 1 Championship Forecasting**
