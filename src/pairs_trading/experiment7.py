"""Re-label delayed-confirmation rows under alternative payoff barriers."""

from __future__ import annotations

import numpy as np
import pandas as pd


BARRIER_CONFIGURATIONS = {
    "target_1_stop_2": (0.010, -0.020),
    "symmetric_0_5": (0.005, -0.005),
    "symmetric_1_0": (0.010, -0.010),
    "target_1_5_stop_1": (0.015, -0.010),
}


def relabel_barriers(
    rows: pd.DataFrame,
    prices: pd.DataFrame,
    take_profit_return: float,
    stop_loss_return: float,
    horizon_days: int = 20,
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Recompute exits and targets from the actual later spread-return path."""
    if take_profit_return <= 0 or stop_loss_return >= 0:
        raise ValueError("Take profit must be positive and stop loss negative.")
    result = rows.copy()
    for column in ["signal_date", "entry_date", "exit_date", "label_available_date"]:
        result[column] = pd.to_datetime(result[column])
    price_frame = prices[["date", "capitaline_code", "close"]].copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    updates: list[dict[str, object]] = []
    cost_rate = transaction_cost_bps / 10_000.0
    group_columns = ["window_id", "left_code", "right_code"]
    for (_, left_code, right_code), group in result.groupby(group_columns, sort=False):
        start = group["signal_date"].min()
        end = pd.to_datetime(group["trading_end"]).max()
        pair = price_frame.loc[
            price_frame["date"].between(start, end)
            & price_frame["capitaline_code"].isin([left_code, right_code])
        ].pivot(index="date", columns="capitaline_code", values="close").dropna()
        if left_code not in pair or right_code not in pair:
            continue
        date_positions = {date: index for index, date in enumerate(pair.index)}
        left_return = pair[left_code].pct_change().fillna(0.0)
        right_return = pair[right_code].pct_change().fillna(0.0)
        for row in group.itertuples(index=False):
            signal_date = pd.Timestamp(row.signal_date)
            signal_index = date_positions.get(signal_date)
            if signal_index is None or signal_index + 1 >= len(pair):
                continue
            beta = float(row.absolute_hedge_ratio)
            denominator = 1.0 + beta
            long_spread_return = (
                left_return / denominator - beta * right_return / denominator
            )
            final_index = min(signal_index + horizon_days, len(pair) - 1)
            future_gross = int(row.direction) * long_spread_return.iloc[
                signal_index + 1 : final_index + 1
            ]
            equity = (1.0 + future_gross).cumprod() * (1.0 - cost_rate)
            net_if_closed = equity * (1.0 - cost_rate) - 1.0
            take_hits = np.flatnonzero(net_if_closed.to_numpy() >= take_profit_return)
            stop_hits = np.flatnonzero(net_if_closed.to_numpy() <= stop_loss_return)
            take_offset = int(take_hits[0]) if len(take_hits) else None
            stop_offset = int(stop_hits[0]) if len(stop_hits) else None
            if take_offset is not None and (stop_offset is None or take_offset < stop_offset):
                exit_offset, outcome, target = take_offset, "take_profit", 1
            elif stop_offset is not None:
                exit_offset, outcome, target = stop_offset, "stop_loss", 0
            else:
                exit_offset, outcome, target = len(net_if_closed) - 1, "horizon", 0
            value = float(net_if_closed.iloc[exit_offset])
            updates.append(
                {
                    "event_id": row.event_id,
                    "entry_date": pd.Timestamp(net_if_closed.index[0]),
                    "exit_date_new": pd.Timestamp(net_if_closed.index[exit_offset]),
                    "barrier_outcome_new": outcome,
                    "target_new": target,
                    "trade_net_return_new": value,
                    "profitable_new": int(value > 0),
                    "holding_days_new": exit_offset + 1,
                }
            )
    update_frame = pd.DataFrame(updates)
    result = result.merge(update_frame, on=["event_id", "entry_date"], how="inner")
    result["exit_date"] = result.pop("exit_date_new")
    result["label_available_date"] = result["exit_date"]
    result["barrier_outcome"] = result.pop("barrier_outcome_new")
    result["target"] = result.pop("target_new")
    result["trade_net_return"] = result.pop("trade_net_return_new")
    result["profitable"] = result.pop("profitable_new")
    result["holding_days"] = result.pop("holding_days_new")
    result["take_profit_return"] = take_profit_return
    result["stop_loss_return"] = stop_loss_return
    return result

