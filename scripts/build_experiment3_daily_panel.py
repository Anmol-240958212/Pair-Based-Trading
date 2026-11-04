"""Screen at 90% coverage and build the daily pair-signal panel."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.data import load_static_data
from pairs_trading.experiment3 import build_daily_pair_samples
from pairs_trading.experiment3_config import (
    MINIMUM_COVERAGE,
    MINIMUM_OBSERVATIONS,
    PAIR_SPECIFICATION,
)
from pairs_trading.ml import add_market_industry_features
from pairs_trading.selection import apply_pair_specification
from pairs_trading.walkforward import screen_same_industry_pairs


def main() -> None:
    output = ROOT / "data" / "processed"
    prices = pd.read_parquet(output / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    company_master, _ = load_static_data(ROOT)
    windows = pd.read_csv(output / "walkforward_summary.csv")
    for column in ["formation_start", "formation_end", "trading_start", "trading_end"]:
        windows[column] = pd.to_datetime(windows[column])

    screen_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    summary_records: list[dict[str, object]] = []
    for window in windows.itertuples(index=False):
        eligible, screened = screen_same_industry_pairs(
            prices,
            company_master,
            window.formation_start,
            window.formation_end,
            minimum_coverage=MINIMUM_COVERAGE,
            minimum_observations=MINIMUM_OBSERVATIONS,
        )
        screened = screened.copy()
        screened["window_id"] = window.window_id
        screen_frames.append(screened)
        selected = apply_pair_specification(screened, PAIR_SPECIFICATION)
        selected = selected.loc[
            selected["hedge_ratio"].gt(0)
            & selected["spread_half_life_days"].between(2, 120)
        ]
        window_rows = 0
        for pair in selected.to_dict(orient="records"):
            samples = build_daily_pair_samples(
                prices,
                pair,
                int(pair["left_code"]),
                int(pair["right_code"]),
                str(pair["industry"]),
                window.window_id,
                window.trading_start,
                window.trading_end,
                feature_history_start=window.formation_start,
            )
            if samples.empty:
                continue
            for column in ["formation_start", "formation_end", "trading_start", "trading_end"]:
                samples[column] = getattr(window, column)
            event_frames.append(samples)
            window_rows += len(samples)
        summary_records.append(
            {
                "window_id": window.window_id,
                "eligible_companies_90pct": int(len(eligible)),
                "same_industry_pairs_tested": int(len(screened)),
                "robust_pairs": int(len(selected)),
                "daily_signal_rows": int(window_rows),
            }
        )
        print(
            f"{window.window_id}: {len(eligible)} companies, "
            f"{len(selected)} robust pairs, {window_rows} daily rows",
            flush=True,
        )

    screens = pd.concat(screen_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)
    events = add_market_industry_features(events, prices, company_master, screens)
    signal_year = pd.to_datetime(events["signal_date"]).dt.year
    events["market_regime"] = np.select(
        [signal_year.le(2019), signal_year.between(2020, 2021)],
        ["pre_covid", "covid_transition"],
        default="post_covid",
    )
    screens.to_parquet(output / "experiment3_pair_screens.parquet", index=False)
    events.to_parquet(output / "experiment3_daily_panel.parquet", index=False)
    pd.DataFrame.from_records(summary_records).to_csv(
        output / "experiment3_panel_summary.csv", index=False
    )
    print(
        f"Saved {len(events):,} daily rows from {events.event_id.nunique():,} signals."
    )


if __name__ == "__main__":
    main()
