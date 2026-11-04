"""Read-only application services over the frozen research artefacts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pairs_trading.backtest import backtest_pair
from pairs_trading.data import load_price_data, load_static_data, price_data_audit
from pairs_trading.research_config import (
    CLASSIFIER_NAME,
    CLASSIFIER_TARGET,
    ENTRY_Z,
    EXIT_Z,
    LOCKED_TEST_END,
    LOCKED_TEST_START,
    MAXIMUM_HOLDING_DAYS,
    PAIR_SPECIFICATION,
    PROBABILITY_THRESHOLD,
    STOP_Z,
    TRANSACTION_COST_BPS,
)
from pairs_trading.statistics import analyze_pair


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def clean(value: Any) -> Any:
    """Convert pandas/numpy values into strict JSON-compatible objects."""
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return clean(frame.to_dict(orient="records"))


@lru_cache(maxsize=1)
def prices() -> pd.DataFrame:
    return load_price_data(ROOT, range(2015, 2027))


@lru_cache(maxsize=1)
def company_master() -> pd.DataFrame:
    master, _ = load_static_data(ROOT)
    return master


@lru_cache(maxsize=1)
def pair_screens() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "pair_screens.parquet")


@lru_cache(maxsize=1)
def walkforward_summary() -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED / "walkforward_summary.csv")
    for column in ["formation_start", "formation_end", "trading_start", "trading_end"]:
        frame[column] = pd.to_datetime(frame[column])
    return frame


def selected_screen(frame: pd.DataFrame, specification: str) -> pd.DataFrame:
    allowed = {
        "forward",
        "bidirectional",
        "bidirectional_fdr",
        "bidirectional_johansen",
        "bidirectional_fdr_johansen",
    }
    if specification not in allowed:
        raise ValueError(f"Unsupported pair specification: {specification}")
    selected = frame.loc[
        frame["cointegration_p_value"].le(0.05)
        & frame["hedge_ratio"].gt(0)
        & frame["spread_half_life_days"].between(2, 120)
    ]
    if specification != "forward":
        selected = selected.loc[selected["cointegration_p_value_reverse"].le(0.05)]
    if "fdr" in specification:
        selected = selected.loc[selected["passes_fdr"].astype(bool)]
    if "johansen" in specification:
        selected = selected.loc[selected["johansen_passes_5pct"].astype(bool)]
    return selected.copy()


def overview() -> dict[str, Any]:
    event_frame = pd.read_parquet(PROCESSED / "ml_events.parquet")
    significance = pd.read_csv(PROCESSED / "locked_test_significance.csv").iloc[0]
    portfolio = pd.read_csv(PROCESSED / "locked_test_portfolio_cost_sensitivity.csv")
    portfolio_10 = portfolio.loc[portfolio["transaction_cost_bps"].eq(10.0)]
    audit = price_data_audit(prices())
    return clean(
        {
            **audit,
            "adjusted_prices": True,
            "annual_point_in_time_fno_universe": True,
            "industries": int(company_master()["industry"].nunique()),
            "walkforward_windows": int(len(walkforward_summary())),
            "labelled_events": int(len(event_frame)),
            "successful_events": int(event_frame["target"].sum()),
            "locked_test_start": LOCKED_TEST_START,
            "locked_test_end": LOCKED_TEST_END,
            "filter_p_value": float(significance["one_sided_p_value"]),
            "portfolio_at_10bps": portfolio_10.to_dict(orient="records"),
        }
    )


def windows() -> list[dict[str, Any]]:
    return frame_records(walkforward_summary())


def industries(window_id: str) -> list[dict[str, Any]]:
    frame = pair_screens().loc[pair_screens()["window_id"].eq(window_id)]
    if frame.empty:
        raise KeyError(window_id)
    grouped = (
        frame.groupby("industry", as_index=False)
        .agg(
            candidate_pairs=("left_code", "size"),
            companies=(
                "left_code",
                lambda values: 0,
            ),
        )
        .drop(columns="companies")
    )
    company_counts = []
    for industry_name, group in frame.groupby("industry"):
        codes = set(group["left_code"]).union(group["right_code"])
        company_counts.append((industry_name, len(codes)))
    counts = pd.DataFrame(company_counts, columns=["industry", "companies"])
    return frame_records(grouped.merge(counts, on="industry").sort_values("industry"))


def pairs_for_window(
    window_id: str,
    industry: str | None,
    specification: str,
) -> list[dict[str, Any]]:
    frame = pair_screens().loc[pair_screens()["window_id"].eq(window_id)]
    if industry:
        frame = frame.loc[frame["industry"].eq(industry)]
    frame = selected_screen(frame, specification)
    columns = [
        "window_id",
        "industry",
        "left_code",
        "left_company",
        "right_code",
        "right_company",
        "cointegration_p_value",
        "cointegration_p_value_reverse",
        "cointegration_fdr_q_value",
        "passes_fdr",
        "johansen_passes_5pct",
        "return_correlation",
        "hedge_ratio",
        "spread_half_life_days",
        "latest_z_score",
        "minimum_pair_median_turnover",
    ]
    return frame_records(frame[columns].sort_values("cointegration_p_value"))


def pair_analysis(
    window_id: str,
    left_code: int,
    right_code: int,
    entry_z: float = ENTRY_Z,
    exit_z: float = EXIT_Z,
    stop_z: float = STOP_Z,
    maximum_holding_days: int = MAXIMUM_HOLDING_DAYS,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
) -> dict[str, Any]:
    window_rows = walkforward_summary().loc[
        walkforward_summary()["window_id"].eq(window_id)
    ]
    if window_rows.empty:
        raise KeyError(window_id)
    window = window_rows.iloc[0]
    screen = pair_screens().loc[
        pair_screens()["window_id"].eq(window_id)
        & pair_screens()["left_code"].eq(left_code)
        & pair_screens()["right_code"].eq(right_code)
    ]
    if screen.empty:
        raise KeyError(f"{window_id}:{left_code}:{right_code}")
    formation_metrics, formation = analyze_pair(
        prices(), left_code, right_code, window["formation_start"], window["formation_end"]
    )
    backtest_metrics, daily, trades = backtest_pair(
        prices(),
        left_code,
        right_code,
        window["formation_start"],
        window["formation_end"],
        window["trading_start"],
        window["trading_end"],
        entry_z,
        exit_z,
        stop_z,
        maximum_holding_days,
        transaction_cost_bps,
    )
    return clean(
        {
            "window": window.to_dict(),
            "pair": screen.iloc[0].to_dict(),
            "formation_metrics": formation_metrics,
            "formation_series": formation.to_dict(orient="records"),
            "backtest_metrics": backtest_metrics,
            "trading_series": daily.to_dict(orient="records"),
            "trades": trades.to_dict(orient="records"),
        }
    )


def csv_records(filename: str) -> list[dict[str, Any]]:
    return frame_records(pd.read_csv(PROCESSED / filename))


def portfolio_daily(strategy: str) -> list[dict[str, Any]]:
    if strategy not in {"baseline", "ml_filtered"}:
        raise ValueError("Strategy must be baseline or ml_filtered.")
    return csv_records(f"{strategy}_portfolio_daily.csv")


def research_config() -> dict[str, Any]:
    return {
        "pair_specification": PAIR_SPECIFICATION,
        "classifier": CLASSIFIER_NAME,
        "target": CLASSIFIER_TARGET,
        "probability_threshold": PROBABILITY_THRESHOLD,
        "entry_z": ENTRY_Z,
        "exit_z": EXIT_Z,
        "stop_z": STOP_Z,
        "maximum_holding_days": MAXIMUM_HOLDING_DAYS,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "locked_test_start": LOCKED_TEST_START,
        "locked_test_end": LOCKED_TEST_END,
    }
