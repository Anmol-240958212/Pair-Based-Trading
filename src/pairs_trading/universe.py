"""Point-in-time eligibility and economically sensible candidate generation."""

from __future__ import annotations

from itertools import combinations

import pandas as pd


def eligible_companies(
    prices: pd.DataFrame,
    company_master: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    minimum_coverage: float = 0.95,
    minimum_observations: int = 200,
    minimum_median_turnover: float = 0.0,
) -> pd.DataFrame:
    """Select companies using information inside the formation window only."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    window = prices.loc[prices["date"].between(start, end)].copy()
    if window.empty:
        return pd.DataFrame()

    expected_dates = int(window["date"].nunique())
    summary = (
        window.groupby("capitaline_code", as_index=False)
        .agg(
            observations=("date", "nunique"),
            first_observation=("date", "min"),
            last_observation=("date", "max"),
            median_turnover_rs_thousand=("turnover_rs_thousand", "median"),
        )
    )
    summary["coverage"] = summary["observations"] / expected_dates
    summary = summary.loc[
        (summary["coverage"] >= minimum_coverage)
        & (summary["observations"] >= minimum_observations)
        & (summary["median_turnover_rs_thousand"].fillna(0) >= minimum_median_turnover)
    ]
    result = summary.merge(company_master, on="capitaline_code", how="left")
    return result.sort_values(["industry", "company_name"]).reset_index(drop=True)


def same_industry_pairs(eligible: pd.DataFrame, industry: str) -> pd.DataFrame:
    """Create unordered pairs from one industry; no statistical claim is made."""
    members = eligible.loc[eligible["industry"].eq(industry)].copy()
    records = []
    for (_, left), (_, right) in combinations(members.iterrows(), 2):
        records.append(
            {
                "left_code": int(left["capitaline_code"]),
                "left_company": left["company_name"],
                "right_code": int(right["capitaline_code"]),
                "right_company": right["company_name"],
                "industry": industry,
            }
        )
    return pd.DataFrame.from_records(records)

