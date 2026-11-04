"""Out-of-sample significance and portfolio evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_filter_permutation_test(
    predictions: pd.DataFrame,
    repetitions: int = 10_000,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Compare the ML-approved mean return with random within-window filters."""
    approved = predictions.loc[predictions["approved_by_ml"]]
    if approved.empty:
        raise ValueError("The model approved no test trades.")
    observed = float(approved["trade_net_return"].mean())
    rng = np.random.default_rng(random_state)
    simulated = np.empty(repetitions)
    groups = [group for _, group in predictions.groupby("window_id")]
    approved_counts = {
        window: int(group["approved_by_ml"].sum())
        for window, group in predictions.groupby("window_id")
    }
    for repetition in range(repetitions):
        chosen_returns: list[float] = []
        for group in groups:
            window = group["window_id"].iloc[0]
            count = approved_counts[window]
            if count:
                indexes = rng.choice(len(group), size=count, replace=False)
                chosen_returns.extend(group.iloc[indexes]["trade_net_return"].tolist())
        simulated[repetition] = np.mean(chosen_returns)
    p_value = float((1 + np.sum(simulated >= observed)) / (repetitions + 1))
    return {
        "approved_trades": int(len(approved)),
        "observed_ml_mean_return": observed,
        "random_filter_mean_return": float(simulated.mean()),
        "random_filter_95pct_lower": float(np.quantile(simulated, 0.025)),
        "random_filter_95pct_upper": float(np.quantile(simulated, 0.975)),
        "one_sided_p_value": p_value,
        "repetitions": repetitions,
    }


def select_portfolio_trades(
    events: pd.DataFrame,
    maximum_positions: int = 10,
    avoid_shared_companies: bool = True,
) -> pd.DataFrame:
    """Accept signals chronologically subject to fixed slots and overlap rules."""
    if maximum_positions < 1:
        raise ValueError("Maximum positions must be positive.")
    candidates = events.copy()
    for column in ["entry_date", "exit_date"]:
        candidates[column] = pd.to_datetime(candidates[column])
    sort_columns = ["entry_date"]
    ascending = [True]
    if "predicted_probability" in candidates:
        sort_columns.append("predicted_probability")
        ascending.append(False)
    candidates = candidates.sort_values(sort_columns, ascending=ascending)

    accepted: list[pd.Series] = []
    active: list[pd.Series] = []
    for entry_date, day_candidates in candidates.groupby("entry_date", sort=True):
        active = [trade for trade in active if trade["exit_date"] >= entry_date]
        for _, candidate in day_candidates.iterrows():
            if len(active) >= maximum_positions:
                break
            if avoid_shared_companies:
                active_codes = {
                    int(code)
                    for trade in active
                    for code in [trade["left_code"], trade["right_code"]]
                }
                if (
                    int(candidate["left_code"]) in active_codes
                    or int(candidate["right_code"]) in active_codes
                ):
                    continue
            accepted.append(candidate)
            active.append(candidate)
    return pd.DataFrame(accepted).reset_index(drop=True)


def portfolio_from_trade_paths(
    selected_events: pd.DataFrame,
    trade_paths: pd.DataFrame,
    maximum_positions: int = 10,
    transaction_cost_bps: float = 10.0,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    """Create a fixed-slot daily portfolio; unused slots remain cash."""
    accepted = select_portfolio_trades(selected_events, maximum_positions)
    paths = trade_paths.loc[trade_paths["event_id"].isin(accepted["event_id"])].copy()
    paths["net_return_scenario"] = (
        paths["gross_return"]
        - paths["turnover"] * transaction_cost_bps / 10_000.0
    )
    daily = (
        paths.groupby("date", as_index=False)["net_return_scenario"]
        .sum()
        .rename(columns={"net_return_scenario": "portfolio_return"})
    )
    if daily.empty:
        raise ValueError("No accepted trade paths are available.")
    all_dates = pd.DataFrame(
        {"date": pd.bdate_range(daily["date"].min(), daily["date"].max())}
    )
    daily = all_dates.merge(daily, on="date", how="left").fillna(
        {"portfolio_return": 0.0}
    )
    daily["portfolio_return"] /= maximum_positions
    daily["equity"] = (1.0 + daily["portfolio_return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    standard_deviation = float(daily["portfolio_return"].std(ddof=1))
    metrics = {
        "accepted_trades": int(len(accepted)),
        "total_return": float(daily["equity"].iloc[-1] - 1.0),
        "annualized_sharpe": float(
            np.sqrt(252) * daily["portfolio_return"].mean() / standard_deviation
        )
        if standard_deviation > 0
        else float("nan"),
        "maximum_drawdown": float(daily["drawdown"].min()),
        "win_rate": float(accepted["profitable"].mean()),
        "transaction_cost_bps": float(transaction_cost_bps),
        "maximum_positions": maximum_positions,
    }
    return metrics, daily, accepted
