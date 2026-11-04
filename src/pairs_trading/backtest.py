"""Out-of-sample baseline backtesting for a formation-period equity pair."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .statistics import analyze_pair


def _trading_matrix(
    prices: pd.DataFrame,
    left_code: int,
    right_code: int,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    window = prices.loc[
        prices["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        & prices["capitaline_code"].isin([left_code, right_code]),
        ["date", "capitaline_code", "close"],
    ].copy()
    if window.duplicated(["capitaline_code", "date"], keep=False).any():
        raise ValueError("Duplicate company-date prices must be resolved before backtesting.")
    matrix = window.pivot(index="date", columns="capitaline_code", values="close").dropna()
    if left_code not in matrix or right_code not in matrix or len(matrix) < 20:
        raise ValueError("The selected pair has fewer than 20 overlapping out-of-sample observations.")
    return matrix


def _position_path(
    z_scores: pd.Series,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    maximum_holding_days: int,
) -> tuple[pd.Series, dict[pd.Timestamp, str]]:
    """Generate end-of-day decisions; positions become active the next day."""
    state = 0
    holding_days = 0
    entry_armed = True
    decisions: list[int] = []
    exit_reasons: dict[pd.Timestamp, str] = {}

    for date, z_score in z_scores.items():
        if state == 0:
            holding_days = 0
            if not entry_armed:
                if abs(z_score) < entry_z:
                    entry_armed = True
                decisions.append(state)
                continue
            if entry_z <= z_score < stop_z:
                state = -1
            elif -stop_z < z_score <= -entry_z:
                state = 1
        else:
            holding_days += 1
            reason = None
            if abs(z_score) <= exit_z:
                reason = "mean_reversion"
            elif abs(z_score) >= stop_z:
                reason = "stop_loss"
            elif holding_days >= maximum_holding_days:
                reason = "maximum_holding_period"
            if reason is not None:
                state = 0
                holding_days = 0
                entry_armed = reason == "mean_reversion"
                exit_reasons[pd.Timestamp(date)] = reason
        decisions.append(state)

    return pd.Series(decisions, index=z_scores.index, dtype="int64"), exit_reasons


def _trade_log(
    daily: pd.DataFrame,
    exit_reasons: dict[pd.Timestamp, str],
) -> pd.DataFrame:
    positions = daily["position"].to_numpy()
    records: list[dict[str, object]] = []
    index = 0
    while index < len(daily):
        if positions[index] == 0:
            index += 1
            continue
        entry_index = index
        direction = int(positions[index])
        while index + 1 < len(daily) and positions[index + 1] == direction:
            index += 1
        last_held_index = index
        if index + 1 < len(daily):
            exit_index = index + 1
            signal_date = pd.Timestamp(daily.iloc[last_held_index]["date"])
            exit_reason = exit_reasons.get(signal_date, "rule_exit")
        else:
            exit_index = index
            exit_reason = "end_of_test"

        trade_slice = daily.iloc[entry_index : exit_index + 1]
        trade_return = float((1.0 + trade_slice["net_return"]).prod() - 1.0)
        signal_index = max(entry_index - 1, 0)
        records.append(
            {
                "signal_date": daily.iloc[signal_index]["date"],
                "entry_date": daily.iloc[entry_index]["date"],
                "exit_date": daily.iloc[exit_index]["date"],
                "direction": "Long spread" if direction == 1 else "Short spread",
                "entry_z_score": float(daily.iloc[signal_index]["z_score"]),
                "exit_z_score": float(daily.iloc[last_held_index]["z_score"]),
                "holding_days": int(last_held_index - entry_index + 1),
                "exit_reason": exit_reason,
                "net_return": trade_return,
            }
        )
        index = exit_index + 1
    return pd.DataFrame.from_records(records)


def backtest_pair(
    prices: pd.DataFrame,
    left_code: int,
    right_code: int,
    formation_start: str | pd.Timestamp,
    formation_end: str | pd.Timestamp,
    trading_start: str | pd.Timestamp,
    trading_end: str | pd.Timestamp,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    maximum_holding_days: int = 60,
    transaction_cost_bps: float = 10.0,
) -> tuple[dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    """Backtest fixed formation estimates on a strictly later trading window."""
    if pd.Timestamp(trading_start) <= pd.Timestamp(formation_end):
        raise ValueError("The trading period must begin after the formation period ends.")
    if not 0 <= exit_z < entry_z < stop_z:
        raise ValueError("Thresholds must satisfy 0 <= exit < entry < stop.")
    if maximum_holding_days < 1 or transaction_cost_bps < 0:
        raise ValueError("Holding days must be positive and transaction costs cannot be negative.")

    formation_metrics, _ = analyze_pair(
        prices, left_code, right_code, formation_start, formation_end
    )
    matrix = _trading_matrix(prices, left_code, right_code, trading_start, trading_end)
    beta = float(formation_metrics["hedge_ratio"])
    intercept = float(formation_metrics["intercept"])
    spread_mean = float(formation_metrics["formation_spread_mean"])
    spread_std = float(formation_metrics["formation_spread_standard_deviation"])
    if spread_std <= 0:
        raise ValueError("Formation-period spread standard deviation is not positive.")

    log_left = np.log(matrix[left_code].astype(float))
    log_right = np.log(matrix[right_code].astype(float))
    spread = log_left - intercept - beta * log_right
    z_score = (spread - spread_mean) / spread_std
    decisions, exit_reasons = _position_path(
        z_score, entry_z, exit_z, stop_z, maximum_holding_days
    )
    position = decisions.shift(1, fill_value=0).astype("int64")

    denominator = 1.0 + abs(beta)
    left_weight = 1.0 / denominator
    right_weight = -beta / denominator
    left_return = matrix[left_code].pct_change().fillna(0.0)
    right_return = matrix[right_code].pct_change().fillna(0.0)
    gross_return = position * (left_weight * left_return + right_weight * right_return)
    turnover = position.diff().abs().fillna(position.abs()).astype(float)
    if int(position.iloc[-1]) != 0:
        turnover.iloc[-1] += 1.0
    cost_rate = transaction_cost_bps / 10_000.0
    transaction_cost = turnover * cost_rate
    net_return = gross_return - transaction_cost

    daily = pd.DataFrame(
        {
            "date": matrix.index,
            "left_close": matrix[left_code].to_numpy(),
            "right_close": matrix[right_code].to_numpy(),
            "spread": spread.to_numpy(),
            "z_score": z_score.to_numpy(),
            "position": position.to_numpy(),
            "gross_return": gross_return.to_numpy(),
            "turnover": turnover.to_numpy(),
            "transaction_cost": transaction_cost.to_numpy(),
            "net_return": net_return.to_numpy(),
        }
    )
    daily["gross_equity"] = (1.0 + daily["gross_return"]).cumprod()
    daily["net_equity"] = (1.0 + daily["net_return"]).cumprod()
    daily["drawdown"] = daily["net_equity"] / daily["net_equity"].cummax() - 1.0

    trades = _trade_log(daily, exit_reasons)
    net_std = float(daily["net_return"].std(ddof=1))
    sharpe = (
        float(np.sqrt(252) * daily["net_return"].mean() / net_std)
        if net_std > 0
        else float("nan")
    )
    trade_count = int(len(trades))
    metrics: dict[str, float | int] = {
        **formation_metrics,
        "trading_observations": int(len(daily)),
        "number_of_trades": trade_count,
        "winning_trades": int((trades["net_return"] > 0).sum()) if trade_count else 0,
        "win_rate": float((trades["net_return"] > 0).mean()) if trade_count else float("nan"),
        "convergence_rate": float((trades["exit_reason"] == "mean_reversion").mean())
        if trade_count
        else float("nan"),
        "average_holding_days": float(trades["holding_days"].mean())
        if trade_count
        else float("nan"),
        "gross_total_return": float(daily["gross_equity"].iloc[-1] - 1.0),
        "net_total_return": float(daily["net_equity"].iloc[-1] - 1.0),
        "annualized_sharpe": sharpe,
        "maximum_drawdown": float(daily["drawdown"].min()),
        "total_transaction_cost": float(daily["transaction_cost"].sum()),
    }
    return metrics, daily, trades

