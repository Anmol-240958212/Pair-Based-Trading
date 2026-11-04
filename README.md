# Pair-Based Statistical Arbitrage on Indian Equities

This project combines a reusable Python research layer, a FastAPI service and a
React/TypeScript research interface.

## Structure

- `notebooks/01_pair_research.ipynb` — readable exploratory research workflow
- `api/` — read-only FastAPI endpoints over research data and pair diagnostics
- `frontend/` — React, TypeScript and Plotly.js research interface
- `src/pairs_trading/` — reusable data, universe-selection, and statistical code
- `data_*.xlsx` and `STATIC_DATA_SHARANYA.xlsx` — unmodified source files

The annual source workbooks represent the contemporaneous F&O-company universe
for each year, and the supplied underlying equity prices are adjusted for
corporate actions. The project does not model individual futures contracts,
expiry rolls, margins or lot sizes.

## Setup

```powershell
python -m pip install -r requirements.txt
cd frontend
npm install
cd ..
```

## Run in development

Open two PowerShell terminals in the project folder.

Terminal 1 — API:

```powershell
python -m uvicorn api.main:app --reload --port 8000
```

Terminal 2 — React:

```powershell
cd frontend
npm run dev
```

Open `http://localhost:5173`. Interactive API documentation is available at
`http://localhost:8000/docs`.

## Run the production build

```powershell
cd frontend
npm run build
cd ..
python -m uvicorn api.main:app --port 8000
```

Open `http://localhost:8000`. FastAPI serves the compiled React application.

## Rebuild research data

```powershell
python scripts/build_processed_data.py
```

This creates a derived Parquet file for fast repeated reads. It does not change
any Capitaline workbook and only needs to be rerun when source files change.

Build the completed-window ML examples with:

```powershell
python scripts/build_ml_events.py
```

This uses trailing three-year formation windows and subsequent six-month
trading windows from 2018H1 through 2026H1. The incomplete 2026H2 period is not
labelled.

Reproduce the initial classifier comparison with:

```powershell
python scripts/compare_ml_models.py
```

That comparison trains on 2018–2019 completed outcomes and validates on
2020–2021. It deliberately leaves 2022 onward unused for final testing.

## Open the notebook

```powershell
jupyter lab notebooks/01_pair_research.ipynb
```

The application deliberately separates the formation period from the later
trading-period evaluation. Cointegration and hedge-ratio estimates use only the
selected formation period. The baseline backtest applies next-day positions,
transaction costs, mean-reversion exits, stop-losses and a maximum holding rule
to the selected out-of-sample period.

The ML layer converts completed historical trades into binary examples. Its
features use information available on the signal date only; the later outcome
is used only as the label. Logistic Regression, Decision Tree and Bagging
Decision Trees are compared with a chronological train/validation split.

The frozen out-of-sample methodology and unaltered findings are documented in
`METHODOLOGY.md`; academic sources are listed in `REFERENCES.md`. The locked
2022H1-2026H1 test found that ML reduced the baseline loss and drawdown, but the
10 bps portfolio remained slightly negative and the event-level improvement was
not statistically significant.

## Tests

```powershell
python -m unittest discover -s tests -v
cd frontend
npm run build
```

