"""Rolling pair screening and event generation across chronological windows."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .backtest import backtest_pair
from .ml import add_market_industry_features, build_labeled_events
from .statistics import analyze_pair
from .universe import eligible_companies


def semiannual_windows(
    first_trading_start: str | pd.Timestamp,
    last_complete_trading_end: str | pd.Timestamp,
    formation_years: int = 3,
) -> pd.DataFrame:
    """Create non-overlapping half-year tests with trailing formation windows."""
    first = pd.Timestamp(first_trading_start)
    last = pd.Timestamp(last_complete_trading_end)
    if formation_years < 1 or last < first:
        raise ValueError("Window dates and formation length are invalid.")

    starts = pd.date_range(first, last, freq="6MS")
    records = []
    for trading_start in starts:
        natural_end = trading_start + pd.DateOffset(months=6) - pd.Timedelta(days=1)
        if natural_end > last:
            continue
        formation_start = trading_start - pd.DateOffset(years=formation_years)
        formation_end = trading_start - pd.Timedelta(days=1)
        half = "H1" if trading_start.month == 1 else "H2"
        records.append(
            {
                "window_id": f"{trading_start.year}{half}",
                "formation_start": formation_start,
                "formation_end": formation_end,
                "trading_start": trading_start,
                "trading_end": natural_end,
            }
        )
    return pd.DataFrame.from_records(records)


def screen_same_industry_pairs(
    prices: pd.DataFrame,
    company_master: pd.DataFrame,
    formation_start: str | pd.Timestamp,
    formation_end: str | pd.Timestamp,
    minimum_coverage: float = 0.95,
    minimum_observations: int = 500,
    minimum_median_turnover: float = 0.0,
    fdr_level: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test all eligible same-industry pairs and add Benjamini-Hochberg results."""
    eligible = eligible_companies(
        prices,
        company_master,
        formation_start,
        formation_end,
        minimum_coverage,
        minimum_observations,
        minimum_median_turnover,
    )
    records: list[dict[str, object]] = []
    for industry, members in eligible.dropna(subset=["industry"]).groupby("industry"):
        for (_, left), (_, right) in combinations(members.iterrows(), 2):
            try:
                metrics, _ = analyze_pair(
                    prices,
                    int(left["capitaline_code"]),
                    int(right["capitaline_code"]),
                    formation_start,
                    formation_end,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            records.append(
                {
                    "left_code": int(left["capitaline_code"]),
                    "left_company": left["company_name"],
                    "right_code": int(right["capitaline_code"]),
                    "right_company": right["company_name"],
                    "industry": industry,
                    "left_median_turnover": float(
                        left["median_turnover_rs_thousand"]
                    ),
                    "right_median_turnover": float(
                        right["median_turnover_rs_thousand"]
                    ),
                    "minimum_pair_median_turnover": float(
                        min(
                            left["median_turnover_rs_thousand"],
                            right["median_turnover_rs_thousand"],
                        )
                    ),
                    **metrics,
                }
            )

    screened = pd.DataFrame.from_records(records)
    if screened.empty:
        return eligible, screened
    rejected_forward, adjusted_forward, _, _ = multipletests(
        screened["cointegration_p_value"].to_numpy(),
        alpha=fdr_level,
        method="fdr_bh",
    )
    rejected_conservative, adjusted_conservative, _, _ = multipletests(
        screened["cointegration_p_value_conservative"].to_numpy(),
        alpha=fdr_level,
        method="fdr_bh",
    )
    screened["cointegration_fdr_q_value_forward"] = adjusted_forward
    screened["passes_fdr_forward"] = rejected_forward
    screened["cointegration_fdr_q_value"] = adjusted_conservative
    screened["passes_fdr"] = rejected_conservative
    screened = screened.sort_values(
        ["passes_fdr", "cointegration_p_value"], ascending=[False, True]
    ).reset_index(drop=True)
    return eligible, screened


def generate_walkforward_events(
    prices: pd.DataFrame,
    company_master: pd.DataFrame,
    windows: pd.DataFrame,
    cointegration_p_value_maximum: float = 0.05,
    require_fdr: bool = False,
    require_bidirectional: bool = True,
    require_johansen_confirmation: bool = False,
    minimum_half_life_days: float = 2.0,
    maximum_half_life_days: float = 120.0,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    maximum_holding_days: int = 60,
    transaction_cost_bps: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Screen, backtest and label signals independently within every window."""
    event_frames: list[pd.DataFrame] = []
    screen_frames: list[pd.DataFrame] = []
    window_records: list[dict[str, object]] = []

    for window in windows.itertuples(index=False):
        eligible, screened = screen_same_industry_pairs(
            prices,
            company_master,
            window.formation_start,
            window.formation_end,
        )
        screened = screened.copy()
        screened["window_id"] = window.window_id
        screen_frames.append(screened)

        selected = screened.loc[
            screened["cointegration_p_value"].le(cointegration_p_value_maximum)
            & screened["hedge_ratio"].gt(0)
            & screened["spread_half_life_days"].between(
                minimum_half_life_days, maximum_half_life_days
            )
        ]
        if require_bidirectional:
            selected = selected.loc[
                selected["cointegration_p_value_reverse"].le(
                    cointegration_p_value_maximum
                )
            ]
        if require_fdr:
            selected = selected.loc[selected["passes_fdr"]]
        if require_johansen_confirmation:
            selected = selected.loc[selected["johansen_passes_5pct"]]

        successful_backtests = 0
        event_count = 0
        for pair in selected.itertuples(index=False):
            try:
                metrics, daily, trades = backtest_pair(
                    prices,
                    pair.left_code,
                    pair.right_code,
                    window.formation_start,
                    window.formation_end,
                    window.trading_start,
                    window.trading_end,
                    entry_z,
                    exit_z,
                    stop_z,
                    maximum_holding_days,
                    transaction_cost_bps,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            successful_backtests += 1
            events = build_labeled_events(
                metrics,
                daily,
                trades,
                pair.left_code,
                pair.right_code,
                pair.industry,
            )
            if events.empty:
                continue
            events.insert(0, "window_id", window.window_id)
            events.insert(
                1,
                "event_id",
                [
                    f"{window.window_id}:{pair.left_code}:{pair.right_code}:"
                    f"{pd.Timestamp(date).date()}"
                    for date in events["signal_date"]
                ],
            )
            events["formation_start"] = window.formation_start
            events["formation_end"] = window.formation_end
            events["trading_start"] = window.trading_start
            events["trading_end"] = window.trading_end
            events["cointegration_fdr_q_value"] = pair.cointegration_fdr_q_value
            events["passes_fdr"] = bool(pair.passes_fdr)
            events["cointegration_p_value_reverse"] = (
                pair.cointegration_p_value_reverse
            )
            events["johansen_passes_5pct"] = bool(pair.johansen_passes_5pct)
            events["minimum_pair_median_turnover"] = (
                pair.minimum_pair_median_turnover
            )
            event_frames.append(events)
            event_count += len(events)

        window_records.append(
            {
                "window_id": window.window_id,
                "formation_start": window.formation_start,
                "formation_end": window.formation_end,
                "trading_start": window.trading_start,
                "trading_end": window.trading_end,
                "eligible_companies": int(len(eligible)),
                "same_industry_pairs_tested": int(len(screened)),
                "statistically_selected_pairs": int(len(selected)),
                "successful_pair_backtests": successful_backtests,
                "labelled_events": event_count,
                "requires_bidirectional_cointegration": require_bidirectional,
                "requires_fdr": require_fdr,
                "requires_johansen_confirmation": require_johansen_confirmation,
            }
        )

    all_events = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame()
    )
    all_screens = (
        pd.concat(screen_frames, ignore_index=True)
        if screen_frames
        else pd.DataFrame()
    )
    if not all_events.empty:
        all_events = add_market_industry_features(
            all_events, prices, company_master, all_screens
        )
        signal_year = pd.to_datetime(all_events["signal_date"]).dt.year
        all_events["market_regime"] = np.select(
            [signal_year.le(2019), signal_year.between(2020, 2021)],
            ["pre_covid", "covid_transition"],
            default="post_covid",
        )
    return all_events, all_screens, pd.DataFrame.from_records(window_records)
