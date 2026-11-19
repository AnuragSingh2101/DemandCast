# DemandCast

End-to-end demand forecasting, statistical analysis, and warehouse capacity planning platform built on historical supply-chain transactions.

## Overview

DemandCast answers three questions:
- **Descriptive**: What happened to historical demand?
- **Predictive**: What will demand probably do next?
- **Prescriptive**: What operational actions follow from the forecast?

## Dataset

DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS
- Source: [Mendeley Data](https://data.mendeley.com/datasets/8gx2fvg2k6/5)
- License: CC BY 4.0
- Citation: Constante, Fabian; Silva, Fernando; Pereira, Antonio (2019)

## Architecture

```
Raw CSV -> Ingestion -> Validation & Cleaning -> Analytical Dataset (Parquet) -> Two branches:
    1. EDA & Statistics
    2. Demand Aggregation -> Feature Engineering -> Temporal Validation -> Models -> Forecasts -> Capacity Planning -> Dashboard
```

## Stack

- Python 3.12+
- Pandas, NumPy, SciPy
- Statsmodels, Scikit-learn
- XGBoost, LightGBM
- DuckDB, PyArrow
- Streamlit, Plotly
- Pytest, Ruff, Joblib

## Installation

```powershell
# Clone and enter directory
cd DemandCast

# Create virtual environment and install dependencies
uv sync --cache-dir .uv-cache

# Activate environment
.\.venv\Scripts\Activate.ps1

# Run ETL pipeline
python -m src.data.pipeline

# Run tests
python -m pytest tests -v

# Launch dashboard
python -m streamlit run app/Home.py
```

## Project Structure

```
demandcast/
  data/ raw/ interim/ processed/ forecasts/
  notebooks/ 01_data_audit.ipynb  02_eda.ipynb  03_statistical_analysis.ipynb
             04_forecasting_experiments.ipynb  05_model_evaluation.ipynb
  src/ data/ features/ statistics/ models/ evaluation/ anomaly/ planning/ utils/
  app/ Home.py  pages/
  models/  reports/  docs/  scripts/  tests/
  .github/workflows/ci.yml
  pyproject.toml  uv.lock  requirements.txt  .gitignore  config.yaml  README.md  CLAUDE.md
```

## License

This project uses the DataCo dataset under CC BY 4.0. The code is MIT licensed.