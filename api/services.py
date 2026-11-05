"""Read-only application services over the frozen research artefacts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from pairs_trading.backtest import backtest_pair
from pairs_trading.data import load_price_data, load_static_data, price_data_audit
from pairs_trading.experiment3_config import (
    PREDICTION_HORIZON_DAYS as FINAL_HORIZON_DAYS,
    STOP_LOSS_RETURN as FINAL_STOP_LOSS_RETURN,
    TAKE_PROFIT_RETURN as FINAL_TAKE_PROFIT_RETURN,
    TRANSACTION_COST_BPS as FINAL_TRANSACTION_COST_BPS,
)
from pairs_trading.experiment5_config import (
    PROBABILITY_THRESHOLD as FINAL_PROBABILITY_THRESHOLD,
)
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


def trade_statistics(filename: str) -> dict[str, Any]:
    """Summarise selected trades for display without duplicating calculations in React."""
    trades = pd.read_parquet(PROCESSED / filename)
    winners = trades.loc[trades["trade_net_return"].gt(0), "trade_net_return"]
    losers = trades.loc[trades["trade_net_return"].le(0), "trade_net_return"]
    loss_total = abs(float(losers.sum()))
    outcomes = trades["barrier_outcome"].value_counts()
    return clean(
        {
            "trades": int(len(trades)),
            "profitable_trades": int(trades["profitable"].sum()),
            "win_rate": float(trades["profitable"].mean()),
            "target_rate": float(trades["target"].mean()),
            "take_profit_hits": int(outcomes.get("take_profit", 0)),
            "stop_loss_hits": int(outcomes.get("stop_loss", 0)),
            "horizon_exits": int(outcomes.get("horizon", 0)),
            "mean_trade_return": float(trades["trade_net_return"].mean()),
            "median_trade_return": float(trades["trade_net_return"].median()),
            "average_winner": float(winners.mean()),
            "average_loser": float(losers.mean()),
            "profit_factor": float(winners.sum() / loss_total) if loss_total else None,
            "average_holding_days": float(trades["holding_days"].mean()),
            "median_holding_days": float(trades["holding_days"].median()),
        }
    )


def prediction_statistics(filename: str, sample: str) -> dict[str, Any]:
    predictions = pd.read_parquet(PROCESSED / filename)
    probability = predictions["predicted_probability"]
    approved = predictions["approved"].astype(bool)
    return clean(
        {
            "sample": sample,
            "rows": int(len(predictions)),
            "roc_auc": float(roc_auc_score(predictions["target"], probability)),
            "brier_score": float(brier_score_loss(predictions["target"], probability)),
            "accuracy_at_0_5": float(
                accuracy_score(predictions["target"], probability.ge(0.5))
            ),
            "approved_rows": int(approved.sum()),
            "approved_target_rate": float(predictions.loc[approved, "target"].mean()),
            "approved_profit_rate": float(
                predictions.loc[approved, "profitable"].mean()
            ),
        }
    )


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


@lru_cache(maxsize=1)
def final_model_predictions() -> pd.DataFrame:
    """Frozen three-day Markov-Logistic walk-forward predictions."""
    frame = pd.read_parquet(PROCESSED / "experiment5_delay_3_predictions.parquet")
    for column in ["trigger_date", "signal_date", "entry_date", "exit_date"]:
        frame[column] = pd.to_datetime(frame[column])
    return frame


@lru_cache(maxsize=1)
def final_model_trades() -> pd.DataFrame:
    """Trades selected after model approval and portfolio-capacity controls."""
    frame = pd.read_parquet(PROCESSED / "experiment5_delay_3_trades.parquet")
    for column in ["trigger_date", "signal_date", "entry_date", "exit_date"]:
        frame[column] = pd.to_datetime(frame[column])
    return frame


def final_model_pair_analysis(
    window_id: str, left_code: int, right_code: int
) -> dict[str, Any]:
    """Explain the frozen final-model decisions for one pair and window."""
    predictions = final_model_predictions()
    rows = predictions.loc[
        predictions["window_id"].eq(window_id)
        & predictions["left_code"].eq(left_code)
        & predictions["right_code"].eq(right_code)
    ].copy()
    executed = final_model_trades().loc[
        final_model_trades()["window_id"].eq(window_id)
        & final_model_trades()["left_code"].eq(left_code)
        & final_model_trades()["right_code"].eq(right_code)
    ].copy()

    config = {
        "model": "Two-state Gaussian Markov regime filter + class-balanced Logistic Regression",
        "observation_days": 3,
        "probability_threshold": FINAL_PROBABILITY_THRESHOLD,
        "take_profit_return": FINAL_TAKE_PROFIT_RETURN,
        "stop_loss_return": FINAL_STOP_LOSS_RETURN,
        "maximum_holding_days": FINAL_HORIZON_DAYS,
        "transaction_cost_bps_per_side": FINAL_TRANSACTION_COST_BPS,
        "mandatory_news_veto": False,
    }
    if rows.empty:
        window_year = int(window_id[:4]) if len(window_id) >= 4 else 0
        reason = (
            "outside_post_2022_evaluation_period"
            if window_year < 2022
            else "no_three_day_candidate_signals"
        )
        return {
            "available": False,
            "reason": reason,
            "config": config,
            "summary": {
                "candidate_signals": 0,
                "confirmation_passed": 0,
                "probability_passed": 0,
                "approved_signals": 0,
                "executed_trades": 0,
                "profitable_trades": 0,
                "win_rate": None,
                "compounded_pair_return": 0.0,
                "average_trade_return": None,
                "average_holding_days": None,
            },
            "signals": [],
            "trades": [],
            "equity_series": [],
        }

    executed_ids = set(executed["event_id"])
    rows["probability_passed"] = rows["predicted_probability"].ge(
        FINAL_PROBABILITY_THRESHOLD
    )
    rows["executed"] = rows["event_id"].isin(executed_ids)
    rows["decision"] = np.select(
        [
            ~rows["passes_confirmation"].astype(bool),
            ~rows["probability_passed"],
            rows["executed"],
            rows["approved"].astype(bool),
        ],
        [
            "rejected_confirmation",
            "rejected_probability",
            "executed",
            "approved_not_selected",
        ],
        default="not_approved",
    )

    if executed.empty:
        compounded_return = 0.0
        win_rate = None
        average_return = None
        average_holding = None
        profitable_trades = 0
        equity_series: list[dict[str, Any]] = []
    else:
        executed = executed.sort_values(["exit_date", "entry_date"])
        compounded_return = float((1.0 + executed["trade_net_return"]).prod() - 1.0)
        profitable_trades = int(executed["profitable"].sum())
        win_rate = float(executed["profitable"].mean())
        average_return = float(executed["trade_net_return"].mean())
        average_holding = float(executed["holding_days"].mean())
        equity = 1.0
        equity_series = []
        for trade in executed.itertuples(index=False):
            equity *= 1.0 + float(trade.trade_net_return)
            equity_series.append(
                {"date": trade.exit_date, "net_equity": equity, "event_id": trade.event_id}
            )

    signal_columns = [
        "event_id", "trigger_date", "signal_date", "entry_date", "exit_date",
        "direction", "initial_z_score", "entry_z_score", "absolute_z_convergence",
        "same_direction_fraction", "factor_alignment_fraction",
        "lead_lag_agreement_fraction", "maximum_standardized_idiosyncratic_shock",
        "markov_high_volatility_probability", "markov_favorable_probability",
        "predicted_probability", "passes_confirmation", "probability_passed",
        "approved", "executed", "decision", "barrier_outcome",
        "trade_net_return", "holding_days",
    ]
    trade_columns = [
        "event_id", "trigger_date", "signal_date", "entry_date", "exit_date",
        "direction", "initial_z_score", "entry_z_score", "predicted_probability",
        "barrier_outcome", "trade_net_return", "profitable", "holding_days",
    ]
    return {
        "available": True,
        "reason": None,
        "config": config,
        "summary": {
            "candidate_signals": int(len(rows)),
            "confirmation_passed": int(rows["passes_confirmation"].sum()),
            "probability_passed": int(rows["probability_passed"].sum()),
            "approved_signals": int(rows["approved"].sum()),
            "executed_trades": int(len(executed)),
            "profitable_trades": profitable_trades,
            "win_rate": win_rate,
            "compounded_pair_return": compounded_return,
            "average_trade_return": average_return,
            "average_holding_days": average_holding,
        },
        "signals": rows[signal_columns].sort_values("signal_date").to_dict(orient="records"),
        "trades": executed[trade_columns].to_dict(orient="records"),
        "equity_series": equity_series,
    }


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
    model_rows = final_model_predictions().loc[
        final_model_predictions()["window_id"].eq(window_id)
    ]
    model_counts = (
        model_rows.groupby(["left_code", "right_code"], as_index=False)
        .agg(
            model_candidate_signals=("event_id", "size"),
            model_approved_signals=("approved", "sum"),
        )
    )
    trade_rows = final_model_trades().loc[
        final_model_trades()["window_id"].eq(window_id)
    ]
    trade_counts = (
        trade_rows.groupby(["left_code", "right_code"], as_index=False)
        .size()
        .rename(columns={"size": "model_executed_trades"})
    )
    frame = frame.merge(model_counts, on=["left_code", "right_code"], how="left")
    frame = frame.merge(trade_counts, on=["left_code", "right_code"], how="left")
    for column in [
        "model_candidate_signals", "model_approved_signals", "model_executed_trades"
    ]:
        frame[column] = frame[column].fillna(0).astype(int)
    frame["model_available"] = frame["model_candidate_signals"].gt(0)
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
        "model_available",
        "model_candidate_signals",
        "model_approved_signals",
        "model_executed_trades",
    ]
    return frame_records(
        frame[columns].sort_values(
            ["model_executed_trades", "model_approved_signals", "model_candidate_signals", "cointegration_p_value"],
            ascending=[False, False, False, True],
        )
    )


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
            "final_model": final_model_pair_analysis(window_id, left_code, right_code),
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


def experiment2_results() -> dict[str, Any]:
    """Return explicitly post-hoc development results for the second experiment."""
    return clean(
        {
            "interpretation": "post_hoc_development_not_locked_holdout",
            "validation": pd.read_csv(PROCESSED / "experiment2_validation.csv").iloc[
                0
            ].to_dict(),
            "significance": pd.read_csv(
                PROCESSED / "experiment2_significance.csv"
            ).iloc[0].to_dict(),
            "portfolio": pd.read_csv(
                PROCESSED / "experiment2_portfolio.csv"
            ).to_dict(orient="records"),
            "windows": pd.read_csv(
                PROCESSED / "experiment2_summary.csv"
            ).to_dict(orient="records"),
        }
    )


def experiment3_results() -> dict[str, Any]:
    """Return daily-panel development results without implying a fresh holdout."""
    return clean(
        {
            "interpretation": "post_hoc_daily_panel_development_not_locked_holdout",
            "panel": pd.read_csv(
                PROCESSED / "experiment3_panel_summary.csv"
            ).to_dict(orient="records"),
            "validation": pd.read_csv(
                PROCESSED / "experiment3_validation.csv"
            ).to_dict(orient="records"),
            "comparison": pd.read_csv(
                PROCESSED / "experiment3_comparison.csv"
            ).to_dict(orient="records"),
            "portfolio": pd.read_csv(
                PROCESSED / "experiment3_portfolio.csv"
            ).to_dict(orient="records"),
            "significance": pd.read_csv(
                PROCESSED / "experiment3_significance.csv"
            ).to_dict(orient="records"),
            "windows": pd.read_csv(
                PROCESSED / "experiment3_window_summary.csv"
            ).to_dict(orient="records"),
        }
    )


def experiment4_results() -> dict[str, Any]:
    """Return mathematical-model comparisons as a post-hoc development study."""
    return clean(
        {
            "interpretation": "post_hoc_mathematical_model_development_not_locked_holdout",
            "validation": pd.read_csv(PROCESSED / "experiment4_validation.csv").to_dict(orient="records"),
            "thresholds": pd.read_csv(PROCESSED / "experiment4_thresholds.csv").to_dict(orient="records"),
            "comparison": pd.read_csv(PROCESSED / "experiment4_comparison.csv").to_dict(orient="records"),
            "portfolio": pd.read_csv(PROCESSED / "experiment4_portfolio.csv").to_dict(orient="records"),
            "significance": pd.read_csv(PROCESSED / "experiment4_significance.csv").to_dict(orient="records"),
            "windows": pd.read_csv(PROCESSED / "experiment4_window_summary.csv").to_dict(orient="records"),
        }
    )


def experiment5_results() -> dict[str, Any]:
    """Return delayed-confirmation results with validation selection disclosed."""
    return clean(
        {
            "interpretation": "post_hoc_confirmation_study_not_locked_holdout",
            "validation": pd.read_csv(PROCESSED / "experiment5_validation.csv").to_dict(orient="records"),
            "comparison": pd.read_csv(PROCESSED / "experiment5_comparison.csv").to_dict(orient="records"),
            "portfolio": pd.read_csv(PROCESSED / "experiment5_portfolio.csv").to_dict(orient="records"),
            "significance": pd.read_csv(PROCESSED / "experiment5_significance.csv").to_dict(orient="records"),
            "windows": pd.read_csv(PROCESSED / "experiment5_window_summary.csv").to_dict(orient="records"),
            "three_day_trade_statistics": trade_statistics(
                "experiment5_delay_3_trades.parquet"
            ),
            "three_day_model_statistics": [
                prediction_statistics(
                    "experiment5_delay_3_validation_predictions.parquet",
                    "2020-2021 validation",
                ),
                prediction_statistics(
                    "experiment5_delay_3_predictions.parquet", "post-2022 diagnostic"
                ),
            ],
        }
    )


def experiment6_results() -> dict[str, Any]:
    """Return the point-in-time NSE announcement veto comparison."""
    announcements = pd.read_parquet(PROCESSED / "experiment6_nse_announcements.parquet")
    return clean(
        {
            "interpretation": "post_hoc_nse_announcement_veto_not_locked_holdout",
            "announcement_rows": int(len(announcements)),
            "covered_companies": int(announcements["capitaline_code"].nunique()),
            "comparison": pd.read_csv(PROCESSED / "experiment6_comparison.csv").to_dict(orient="records"),
            "portfolio": pd.read_csv(PROCESSED / "experiment6_portfolio.csv").to_dict(orient="records"),
            "significance": pd.read_csv(PROCESSED / "experiment6_significance.csv").to_dict(orient="records"),
            "without_news_trade_statistics": trade_statistics(
                "experiment6_approved_without_news_trades.parquet"
            ),
            "with_news_trade_statistics": trade_statistics(
                "experiment6_approved_with_news_veto_trades.parquet"
            ),
        }
    )


def experiment7_results() -> dict[str, Any]:
    """Return the alternative payoff-barrier comparison."""
    return clean(
        {
            "interpretation": "post_hoc_payoff_barrier_study_not_locked_holdout",
            "validation": pd.read_csv(PROCESSED / "experiment7_validation.csv").to_dict(orient="records"),
            "comparison": pd.read_csv(PROCESSED / "experiment7_comparison.csv").to_dict(orient="records"),
            "portfolio": pd.read_csv(PROCESSED / "experiment7_portfolio.csv").to_dict(orient="records"),
            "significance": pd.read_csv(PROCESSED / "experiment7_significance.csv").to_dict(orient="records"),
            "trade_statistics": [
                {
                    "configuration": configuration,
                    **trade_statistics(f"experiment7_{configuration}_trades.parquet"),
                }
                for configuration in [
                    "target_1_stop_2",
                    "symmetric_0_5",
                    "symmetric_1_0",
                    "target_1_5_stop_1",
                ]
            ],
        }
    )
