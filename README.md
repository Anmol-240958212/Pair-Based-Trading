# Pair-Based Statistical Arbitrage on Indian Equities

This project uses a shared Python research layer for both a Jupyter notebook and
an interactive Streamlit application.

## Structure

- `notebooks/01_pair_research.ipynb` — readable exploratory research workflow
- `app.py` — interactive company and pair explorer
- `src/pairs_trading/` — reusable data, universe-selection, and statistical code
- `data_*.xlsx` and `STATIC_DATA_SHARANYA.xlsx` — unmodified source files

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run the frontend

```powershell
python scripts/build_processed_data.py
streamlit run app.py
```

The preparation command creates a derived Parquet file for fast repeated reads.
It does not change any Capitaline workbook and only needs to be rerun when the
source files change.

## Open the notebook

```powershell
jupyter lab notebooks/01_pair_research.ipynb
```

The application deliberately separates the formation period from the later
trading-period evaluation. Cointegration and hedge-ratio estimates use only the
selected formation period. The baseline backtest applies next-day positions,
transaction costs, mean-reversion exits, stop-losses and a maximum holding rule
to the selected out-of-sample period.

