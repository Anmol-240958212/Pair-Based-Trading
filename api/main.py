"""FastAPI entry point for the React research interface."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import services


app = FastAPI(
    title="Indian Equity Pair Research API",
    version="1.0.0",
    description="Read-only API over the frozen pairs-trading research pipeline.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/overview")
def overview():
    return services.overview()


@app.get("/api/windows")
def windows():
    return services.windows()


@app.get("/api/industries")
def industries(window_id: str):
    try:
        return services.industries(window_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown window") from error


@app.get("/api/pairs")
def pairs(
    window_id: str,
    industry: str | None = None,
    specification: str = "bidirectional_johansen",
):
    try:
        return services.pairs_for_window(window_id, industry, specification)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/pairs/{window_id}/{left_code}/{right_code}")
def pair_analysis(
    window_id: str,
    left_code: int,
    right_code: int,
    entry_z: float = Query(2.0, gt=0),
    exit_z: float = Query(0.5, ge=0),
    stop_z: float = Query(4.0, gt=0),
    maximum_holding_days: int = Query(60, ge=1, le=252),
    transaction_cost_bps: float = Query(10.0, ge=0, le=100),
):
    if not 0 <= exit_z < entry_z < stop_z:
        raise HTTPException(status_code=422, detail="Require exit < entry < stop.")
    try:
        return services.pair_analysis(
            window_id,
            left_code,
            right_code,
            entry_z,
            exit_z,
            stop_z,
            maximum_holding_days,
            transaction_cost_bps,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown pair or window") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/results/locked-test")
def locked_test():
    return services.csv_records("locked_test_summary.csv")


@app.get("/api/results/portfolio")
def portfolio():
    return services.csv_records("locked_test_portfolio_cost_sensitivity.csv")


@app.get("/api/results/portfolio-daily/{strategy}")
def portfolio_daily(strategy: str):
    try:
        return services.portfolio_daily(strategy)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/results/regimes")
def regimes():
    return services.csv_records("regime_summary.csv")


@app.get("/api/results/significance")
def significance():
    return services.csv_records("locked_test_significance.csv")[0]


@app.get("/api/results/validation")
def validation():
    return services.csv_records("ml_label_validation.csv")


@app.get("/api/config")
def config():
    return services.research_config()


FRONTEND_DIST = ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
