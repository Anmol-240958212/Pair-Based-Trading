"""Formation-window statistics for one candidate equity pair."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def _half_life(spread: pd.Series) -> float:
    lagged = spread.shift(1)
    delta = spread.diff()
    aligned = pd.concat([lagged.rename("lagged"), delta.rename("delta")], axis=1).dropna()
    if len(aligned) < 20:
        return float("nan")
    design = np.column_stack([np.ones(len(aligned)), aligned["lagged"].to_numpy()])
    coefficient = np.linalg.lstsq(design, aligned["delta"].to_numpy(), rcond=None)[0][1]
    if coefficient >= 0:
        return float("nan")
    return float(-np.log(2) / coefficient)


def analyze_pair(
    prices: pd.DataFrame,
    left_code: int,
    right_code: int,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Estimate an Engle-Granger relationship using the formation period only."""
    window = prices.loc[
        prices["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        & prices["capitaline_code"].isin([left_code, right_code]),
        ["date", "capitaline_code", "close"],
    ].copy()
    duplicated = window.duplicated(["capitaline_code", "date"], keep=False)
    if duplicated.any():
        raise ValueError("Duplicate company-date prices must be resolved before pair analysis.")

    matrix = window.pivot(index="date", columns="capitaline_code", values="close").dropna()
    if left_code not in matrix or right_code not in matrix or len(matrix) < 60:
        raise ValueError("The selected pair has fewer than 60 overlapping observations.")

    log_left = np.log(matrix[left_code].astype(float))
    log_right = np.log(matrix[right_code].astype(float))
    design = np.column_stack([np.ones(len(matrix)), log_right.to_numpy()])
    intercept, hedge_ratio = np.linalg.lstsq(design, log_left.to_numpy(), rcond=None)[0]
    spread = log_left - intercept - hedge_ratio * log_right
    coint_statistic, p_value, critical_values = coint(
        log_left, log_right, trend="c", autolag="aic"
    )
    reverse_statistic, reverse_p_value, reverse_critical_values = coint(
        log_right, log_left, trend="c", autolag="aic"
    )
    try:
        johansen = coint_johansen(
            np.column_stack([log_left.to_numpy(), log_right.to_numpy()]),
            det_order=0,
            k_ar_diff=1,
        )
        johansen_trace_statistic = float(johansen.lr1[0])
        johansen_trace_critical_value_5pct = float(johansen.cvt[0, 1])
        johansen_passes_5pct = bool(
            johansen_trace_statistic > johansen_trace_critical_value_5pct
        )
    except (np.linalg.LinAlgError, ValueError):
        johansen_trace_statistic = float("nan")
        johansen_trace_critical_value_5pct = float("nan")
        johansen_passes_5pct = False
    standard_deviation = float(spread.std(ddof=1))
    z_score = (spread - spread.mean()) / standard_deviation if standard_deviation > 0 else np.nan

    detail = pd.DataFrame(
        {
            "date": matrix.index,
            "left_close": matrix[left_code].to_numpy(),
            "right_close": matrix[right_code].to_numpy(),
            "spread": spread.to_numpy(),
            "z_score": np.asarray(z_score),
        }
    )
    metrics = {
        "observations": int(len(matrix)),
        "return_correlation": float(log_left.diff().corr(log_right.diff())),
        "hedge_ratio": float(hedge_ratio),
        "intercept": float(intercept),
        "formation_spread_mean": float(spread.mean()),
        "formation_spread_standard_deviation": standard_deviation,
        "cointegration_statistic": float(coint_statistic),
        "cointegration_p_value": float(p_value),
        "cointegration_statistic_reverse": float(reverse_statistic),
        "cointegration_p_value_reverse": float(reverse_p_value),
        "cointegration_p_value_conservative": float(max(p_value, reverse_p_value)),
        "cointegration_critical_value_1pct": float(critical_values[0]),
        "cointegration_critical_value_5pct": float(critical_values[1]),
        "cointegration_critical_value_10pct": float(critical_values[2]),
        "cointegration_critical_value_reverse_5pct": float(
            reverse_critical_values[1]
        ),
        "johansen_trace_statistic": johansen_trace_statistic,
        "johansen_trace_critical_value_5pct": johansen_trace_critical_value_5pct,
        "johansen_passes_5pct": johansen_passes_5pct,
        "spread_half_life_days": _half_life(spread),
        "latest_z_score": float(detail["z_score"].iloc[-1]),
    }
    return metrics, detail

