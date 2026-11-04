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
    ranking_column = next(
        (
            column
            for column in ["selection_score", "predicted_probability"]
            if column in candidates
        ),
        None,
    )
    if ranking_column:
        sort_columns.append(ranking_column)
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


def volatility_scaled_portfolio_from_trade_paths(
    selected_events: pd.DataFrame,
    trade_paths: pd.DataFrame,
    maximum_positions: int = 10,
    transaction_cost_bps: float = 10.0,
    daily_risk_budget: float = 0.002,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    """Create a cash-aware portfolio with static, signal-date volatility sizing."""
    if daily_risk_budget <= 0:
        raise ValueError("The daily risk budget must be positive.")
    accepted = select_portfolio_trades(selected_events, maximum_positions)
    if accepted.empty:
        raise ValueError("No trades passed the portfolio constraints.")
    required = {"event_id", "signal_pair_volatility"}
    if missing := required.difference(accepted.columns):
        raise ValueError(f"Selected events are missing columns: {sorted(missing)}")
    accepted = accepted.copy()
    maximum_allocation = 1.0 / maximum_positions
    accepted["portfolio_weight"] = (
        daily_risk_budget / accepted["signal_pair_volatility"].clip(lower=1e-6)
    ).clip(lower=0.0, upper=maximum_allocation)
    paths = trade_paths.merge(
        accepted[["event_id", "portfolio_weight"]], on="event_id", how="inner"
    )
    paths["weighted_return"] = paths["portfolio_weight"] * (
        paths["gross_return"]
        - paths["turnover"] * transaction_cost_bps / 10_000.0
    )
    daily = paths.groupby("date", as_index=False)["weighted_return"].sum().rename(
        columns={"weighted_return": "portfolio_return"}
    )
    all_dates = pd.DataFrame(
        {"date": pd.bdate_range(daily["date"].min(), daily["date"].max())}
    )
    daily = all_dates.merge(daily, on="date", how="left").fillna(
        {"portfolio_return": 0.0}
    )
    daily["equity"] = (1.0 + daily["portfolio_return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    standard_deviation = float(daily["portfolio_return"].std(ddof=1))
    metrics: dict[str, float | int] = {
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
        "maximum_positions": int(maximum_positions),
        "daily_risk_budget": float(daily_risk_budget),
        "average_trade_weight": float(accepted["portfolio_weight"].mean()),
    }
    return metrics, daily, accepted


def moving_block_bootstrap_mean_test(
    daily_returns: pd.Series,
    block_length: int = 20,
    repetitions: int = 10_000,
    random_state: int = 42,
) -> dict[str, float | int]:
    """One-sided zero-mean test preserving short-run return dependence."""
    values = pd.Series(daily_returns, dtype=float).dropna().to_numpy()
    if len(values) < block_length * 2:
        raise ValueError("The return series is too short for the requested blocks.")
    if block_length < 1 or repetitions < 1:
        raise ValueError("Block length and repetitions must be positive.")
    observed = float(values.mean())
    centered = values - observed
    rng = np.random.default_rng(random_state)
    blocks_needed = int(np.ceil(len(values) / block_length))
    offsets = np.arange(block_length)
    simulated = np.empty(repetitions)
    for repetition in range(repetitions):
        starts = rng.integers(0, len(values), size=blocks_needed)
        indexes = (starts[:, None] + offsets[None, :]) % len(values)
        sample = centered[indexes.ravel()[: len(values)]]
        simulated[repetition] = sample.mean()
    return {
        "observations": int(len(values)),
        "block_length": int(block_length),
        "repetitions": int(repetitions),
        "observed_daily_mean": observed,
        "annualized_observed_mean": float(observed * 252),
        "one_sided_p_value": float(
            (1 + np.sum(simulated >= observed)) / (repetitions + 1)
        ),
        "null_95pct_lower": float(np.quantile(simulated, 0.025)),
        "null_95pct_upper": float(np.quantile(simulated, 0.975)),
    }
