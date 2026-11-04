"""Build dynamic-hedge, boundary-safe events for Experiment 2."""

from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.backtest import backtest_pair
from pairs_trading.data import load_static_data
from pairs_trading.experiment2_config import (
    ENTRY_Z,
    EXIT_Z,
    HEDGE_LOOKBACK_DAYS,
    HEDGE_RATIO_MODE,
    MAXIMUM_HOLDING_DAYS,
    MINIMUM_ENTRY_DAYS_REMAINING,
    PAIR_SPECIFICATION,
    STOP_Z,
    TRANSACTION_COST_BPS,
)
from pairs_trading.ml import add_market_industry_features, build_labeled_events
from pairs_trading.selection import apply_pair_specification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hedge-mode", choices=["fixed", "rolling"], default=HEDGE_RATIO_MODE
    )
    parser.add_argument("--output-prefix", default="experiment2")
    args = parser.parse_args()
    output = ROOT / "data" / "processed"
    prices = pd.read_parquet(output / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    company_master, _ = load_static_data(ROOT)
    screens = pd.read_parquet(output / "pair_screens.parquet")
    windows = pd.read_csv(output / "walkforward_summary.csv")
    for column in ["formation_start", "formation_end", "trading_start", "trading_end"]:
        windows[column] = pd.to_datetime(windows[column])

    event_frames: list[pd.DataFrame] = []
    summary_records: list[dict[str, object]] = []
    for window in windows.itertuples(index=False):
        window_pairs = screens.loc[screens["window_id"].eq(window.window_id)]
        selected = apply_pair_specification(window_pairs, PAIR_SPECIFICATION)
        selected = selected.loc[
            selected["hedge_ratio"].gt(0)
            & selected["spread_half_life_days"].between(2, 120)
        ]
        event_count = 0
        successful_pairs = 0
        for pair in selected.itertuples(index=False):
            try:
                metrics, daily, trades = backtest_pair(
                    prices,
                    int(pair.left_code),
                    int(pair.right_code),
                    window.formation_start,
                    window.formation_end,
                    window.trading_start,
                    window.trading_end,
                    ENTRY_Z,
                    EXIT_Z,
                    STOP_Z,
                    MAXIMUM_HOLDING_DAYS,
                    TRANSACTION_COST_BPS,
                    minimum_entry_days_remaining=MINIMUM_ENTRY_DAYS_REMAINING,
                    hedge_ratio_mode=args.hedge_mode,
                    hedge_lookback_days=HEDGE_LOOKBACK_DAYS,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            successful_pairs += 1
            events = build_labeled_events(
                metrics,
                daily,
                trades,
                int(pair.left_code),
                int(pair.right_code),
                pair.industry,
            )
            if events.empty:
                continue
            events.insert(0, "window_id", window.window_id)
            events.insert(
                1,
                "event_id",
                [
                    f"exp2:{window.window_id}:{pair.left_code}:{pair.right_code}:"
                    f"{pd.Timestamp(date).date()}"
                    for date in events["signal_date"]
                ],
            )
            for column in ["formation_start", "formation_end", "trading_start", "trading_end"]:
                events[column] = getattr(window, column)
            events["cointegration_fdr_q_value"] = pair.cointegration_fdr_q_value
            events["passes_fdr"] = bool(pair.passes_fdr)
            events["cointegration_p_value_reverse"] = pair.cointegration_p_value_reverse
            events["johansen_passes_5pct"] = bool(pair.johansen_passes_5pct)
            events["minimum_pair_median_turnover"] = pair.minimum_pair_median_turnover
            event_frames.append(events)
            event_count += len(events)
        summary_records.append(
            {
                "window_id": window.window_id,
                "selected_pairs": int(len(selected)),
                "successful_pair_backtests": successful_pairs,
                "labelled_events": event_count,
            }
        )
        print(
            f"{window.window_id}: {len(selected)} pairs, {event_count} events",
            flush=True,
        )

    events = pd.concat(event_frames, ignore_index=True)
    events = add_market_industry_features(events, prices, company_master, screens)
    signal_year = pd.to_datetime(events["signal_date"]).dt.year
    events["market_regime"] = np.select(
        [signal_year.le(2019), signal_year.between(2020, 2021)],
        ["pre_covid", "covid_transition"],
        default="post_covid",
    )
    events.to_parquet(output / f"{args.output_prefix}_events.parquet", index=False)
    pd.DataFrame.from_records(summary_records).to_csv(
        output / f"{args.output_prefix}_event_summary.csv", index=False
    )
    print(
        f"Saved {len(events)} {args.hedge_mode}-hedge events to "
        f"{args.output_prefix}_events.parquet."
    )


if __name__ == "__main__":
    main()
