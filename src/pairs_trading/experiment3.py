"""Daily pair-level samples, barrier labels and chronological ML evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .experiment3_config import (
    MINIMUM_ABSOLUTE_Z,
    PREDICTION_HORIZON_DAYS,
    PROBABILITY_THRESHOLD,
    STOP_LOSS_RETURN,
    TAKE_PROFIT_RETURN,
    TRANSACTION_COST_BPS,
)


DAILY_FEATURE_COLUMNS = [
    "entry_z_score",
    "absolute_entry_z_score",
    "z_change_1d",
    "z_change_5d",
    "z_change_10d",
    "z_change_20d",
    "recent_absolute_z_mean_20d",
    "recent_z_volatility_20d",
    "recent_z_volatility_ratio_20d_60d",
    "spread_change_volatility_20d",
    "spread_volatility_ratio_20d_60d",
    "recent_return_correlation_20d",
    "recent_return_correlation_60d",
    "divergence_duration_days",
    "pair_momentum_gap_20d",
    "left_return_volatility_20d",
    "right_return_volatility_20d",
    "turnover_ratio_20d_60d",
    "cointegration_p_value",
    "cointegration_p_value_reverse",
    "return_correlation",
    "absolute_hedge_ratio",
    "spread_half_life_days",
    "formation_hedge_ratio_drift",
    "formation_spread_mean_shift",
    "formation_spread_volatility_ratio",
    "formation_zero_crossing_rate",
    "market_momentum_20d",
    "market_volatility_20d",
    "market_volatility_ratio_20d_60d",
    "cross_sectional_dispersion_20d",
    "industry_momentum_20d",
    "industry_volatility_20d",
    "minimum_pair_median_turnover",
]


def _consecutive_true_count(condition: pd.Series) -> pd.Series:
    groups = (~condition.fillna(False)).cumsum()
    return condition.fillna(False).astype(int).groupby(groups).cumsum()


def _pair_matrix(
    prices: pd.DataFrame,
    left_code: int,
    right_code: int,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    columns = ["close", "turnover_rs_thousand"]
    pair = prices.loc[
        prices["date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
        & prices["capitaline_code"].isin([left_code, right_code]),
        ["date", "capitaline_code", *columns],
    ]
    result: dict[str, pd.DataFrame] = {}
    for column in columns:
        matrix = pair.pivot(index="date", columns="capitaline_code", values=column)
        if left_code not in matrix or right_code not in matrix:
            raise ValueError("Both pair members require trading-period observations.")
        result[column] = matrix[[left_code, right_code]]
    result["close"] = result["close"].dropna()
    result["turnover_rs_thousand"] = result["turnover_rs_thousand"].reindex(
        result["close"].index
    )
    return result


def build_daily_pair_samples(
    prices: pd.DataFrame,
    pair_metrics: Mapping[str, float | int | str | bool],
    left_code: int,
    right_code: int,
    industry: str,
    window_id: str,
    trading_start: str | pd.Timestamp,
    trading_end: str | pd.Timestamp,
    feature_history_start: str | pd.Timestamp | None = None,
    minimum_absolute_z: float = MINIMUM_ABSOLUTE_Z,
    horizon_days: int = PREDICTION_HORIZON_DAYS,
    take_profit_return: float = TAKE_PROFIT_RETURN,
    stop_loss_return: float = STOP_LOSS_RETURN,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    """Create signal-date features and future barrier labels for one pair."""
    matrices = _pair_matrix(
        prices,
        left_code,
        right_code,
        feature_history_start if feature_history_start is not None else trading_start,
        trading_end,
    )
    close = matrices["close"].astype(float)
    turnover = matrices["turnover_rs_thousand"].astype(float)
    if len(close) <= horizon_days + 20:
        return pd.DataFrame()

    beta = float(pair_metrics["hedge_ratio"])
    intercept = float(pair_metrics["intercept"])
    spread_mean = float(pair_metrics["formation_spread_mean"])
    spread_std = float(pair_metrics["formation_spread_standard_deviation"])
    if spread_std <= 0:
        return pd.DataFrame()
    log_left = np.log(close[left_code])
    log_right = np.log(close[right_code])
    spread = log_left - intercept - beta * log_right
    z_score = (spread - spread_mean) / spread_std
    left_return = close[left_code].pct_change().fillna(0.0)
    right_return = close[right_code].pct_change().fillna(0.0)
    denominator = 1.0 + abs(beta)
    long_spread_return = (
        left_return / denominator - beta * right_return / denominator
    )

    history = pd.DataFrame(index=close.index)
    history["spread"] = spread
    history["z_score"] = z_score
    for lag in [1, 5, 10, 20]:
        history[f"z_change_{lag}d"] = z_score.diff(lag)
    history["recent_absolute_z_mean_20d"] = z_score.rolling(
        20, min_periods=10
    ).mean().abs()
    history["recent_z_volatility_20d"] = z_score.rolling(
        20, min_periods=10
    ).std(ddof=1)
    z_volatility_60d = z_score.rolling(60, min_periods=30).std(ddof=1)
    history["recent_z_volatility_ratio_20d_60d"] = (
        history["recent_z_volatility_20d"] / z_volatility_60d
    )
    history["spread_change_volatility_20d"] = spread.diff().rolling(
        20, min_periods=10
    ).std(ddof=1)
    spread_volatility_60d = spread.diff().rolling(60, min_periods=30).std(ddof=1)
    history["spread_volatility_ratio_20d_60d"] = (
        history["spread_change_volatility_20d"] / spread_volatility_60d
    )
    history["recent_return_correlation_20d"] = left_return.rolling(
        20, min_periods=10
    ).corr(right_return)
    history["recent_return_correlation_60d"] = left_return.rolling(
        60, min_periods=30
    ).corr(right_return)
    history["divergence_duration_days"] = _consecutive_true_count(
        z_score.abs().ge(1.0)
    )
    history["pair_momentum_gap_20d"] = (
        close[left_code].pct_change(20) - close[right_code].pct_change(20)
    ).abs()
    history["left_return_volatility_20d"] = left_return.rolling(
        20, min_periods=10
    ).std(ddof=1)
    history["right_return_volatility_20d"] = right_return.rolling(
        20, min_periods=10
    ).std(ddof=1)
    minimum_turnover = turnover.min(axis=1)
    history["turnover_ratio_20d_60d"] = minimum_turnover.rolling(
        20, min_periods=10
    ).median() / minimum_turnover.rolling(60, min_periods=30).median()

    cost_rate = transaction_cost_bps / 10_000.0
    records: list[dict[str, object]] = []
    for signal_index in range(60, len(history) - 1):
        signal = history.iloc[signal_index]
        if pd.Timestamp(history.index[signal_index]) < pd.Timestamp(trading_start):
            continue
        signal_z = float(signal["z_score"])
        if not np.isfinite(signal_z) or abs(signal_z) < minimum_absolute_z:
            continue
        final_index = min(signal_index + horizon_days, len(history) - 1)
        if final_index <= signal_index:
            continue
        direction = -1 if signal_z > 0 else 1
        future_gross = direction * long_spread_return.iloc[
            signal_index + 1 : final_index + 1
        ]
        equity_after_entry = (1.0 + future_gross).cumprod() * (1.0 - cost_rate)
        net_if_closed = equity_after_entry * (1.0 - cost_rate) - 1.0
        take_hits = np.flatnonzero(net_if_closed.to_numpy() >= take_profit_return)
        stop_hits = np.flatnonzero(net_if_closed.to_numpy() <= stop_loss_return)
        take_offset = int(take_hits[0]) if len(take_hits) else None
        stop_offset = int(stop_hits[0]) if len(stop_hits) else None
        if take_offset is not None and (
            stop_offset is None or take_offset < stop_offset
        ):
            exit_offset = take_offset
            outcome = "take_profit"
            target = 1
        elif stop_offset is not None:
            exit_offset = stop_offset
            outcome = "stop_loss"
            target = 0
        else:
            exit_offset = len(net_if_closed) - 1
            outcome = "horizon"
            target = 0
        exit_date = pd.Timestamp(net_if_closed.index[exit_offset])
        signal_date = pd.Timestamp(history.index[signal_index])
        record: dict[str, object] = {
            "window_id": window_id,
            "event_id": f"exp3:{window_id}:{left_code}:{right_code}:{signal_date.date()}",
            "left_code": int(left_code),
            "right_code": int(right_code),
            "industry": industry,
            "signal_date": signal_date,
            "entry_date": pd.Timestamp(history.index[signal_index + 1]),
            "exit_date": exit_date,
            "label_available_date": exit_date,
            "direction": direction,
            "entry_z_score": signal_z,
            "absolute_entry_z_score": abs(signal_z),
            "barrier_outcome": outcome,
            "target": target,
            "trade_net_return": float(net_if_closed.iloc[exit_offset]),
            "profitable": int(float(net_if_closed.iloc[exit_offset]) > 0),
            "holding_days": int(exit_offset + 1),
            "cointegration_p_value": float(pair_metrics["cointegration_p_value"]),
            "cointegration_p_value_reverse": float(
                pair_metrics["cointegration_p_value_reverse"]
            ),
            "return_correlation": float(pair_metrics["return_correlation"]),
            "absolute_hedge_ratio": abs(beta),
            "spread_half_life_days": float(pair_metrics["spread_half_life_days"]),
            "formation_hedge_ratio_drift": float(
                pair_metrics.get("formation_hedge_ratio_drift", np.nan)
            ),
            "formation_spread_mean_shift": float(
                pair_metrics.get("formation_spread_mean_shift", np.nan)
            ),
            "formation_spread_volatility_ratio": float(
                pair_metrics.get("formation_spread_volatility_ratio", np.nan)
            ),
            "formation_zero_crossing_rate": float(
                pair_metrics.get("formation_zero_crossing_rate", np.nan)
            ),
            "minimum_pair_median_turnover": float(
                pair_metrics["minimum_pair_median_turnover"]
            ),
        }
        for feature in DAILY_FEATURE_COLUMNS:
            if feature in history.columns:
                record[feature] = float(signal[feature])
        records.append(record)
    return pd.DataFrame.from_records(records)


def daily_candidate_models() -> dict[str, Pipeline]:
    scaled = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                DAILY_FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )
    unscaled = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), DAILY_FEATURE_COLUMNS)],
        remainder="drop",
    )
    return {
        "logistic_regression": Pipeline(
            [
                ("prepare", scaled),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced", C=0.5, max_iter=2_000
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("prepare", unscaled),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=200,
                        max_leaf_nodes=15,
                        min_samples_leaf=30,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def _fit_model(name: str, model: Pipeline, training: pd.DataFrame) -> Pipeline:
    fit_parameters: dict[str, np.ndarray] = {}
    if name == "hist_gradient_boosting":
        counts = training["target"].value_counts()
        weights = training["target"].map(
            {label: len(training) / (len(counts) * count) for label, count in counts.items()}
        )
        fit_parameters["model__sample_weight"] = weights.to_numpy()
    model.fit(training[DAILY_FEATURE_COLUMNS], training["target"], **fit_parameters)
    return model


def validation_comparison(
    events: pd.DataFrame,
    training_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = events.copy()
    for column in ["signal_date", "label_available_date"]:
        ordered[column] = pd.to_datetime(ordered[column])
    training_end = pd.Timestamp(training_end)
    validation_end = pd.Timestamp(validation_end)
    training = ordered.loc[ordered["label_available_date"].le(training_end)]
    validation = ordered.loc[
        ordered["signal_date"].gt(training_end)
        & ordered["signal_date"].le(validation_end)
        & ordered["label_available_date"].le(validation_end)
    ]
    predictions = validation[
        [
            "event_id", "window_id", "left_code", "right_code", "signal_date",
            "entry_date", "exit_date", "target", "trade_net_return", "profitable",
        ]
    ].copy()
    records: list[dict[str, object]] = []
    for name, model in daily_candidate_models().items():
        model = _fit_model(name, model, training)
        probability = model.predict_proba(validation[DAILY_FEATURE_COLUMNS])[:, 1]
        approved = probability >= PROBABILITY_THRESHOLD
        classification = probability >= 0.5
        predictions[f"{name}_probability"] = probability
        predictions[f"{name}_approved"] = approved
        records.append(
            {
                "model": name,
                "training_rows": int(len(training)),
                "validation_rows": int(len(validation)),
                "approved_rows": int(approved.sum()),
                "accuracy": float(accuracy_score(validation["target"], classification)),
                "precision": float(
                    precision_score(validation["target"], classification, zero_division=0)
                ),
                "recall": float(
                    recall_score(validation["target"], classification, zero_division=0)
                ),
                "roc_auc": float(roc_auc_score(validation["target"], probability)),
                "brier_score": float(brier_score_loss(validation["target"], probability)),
                "approved_mean_return": float(
                    validation.loc[approved, "trade_net_return"].mean()
                )
                if approved.any()
                else float("nan"),
                "approved_profit_rate": float(
                    validation.loc[approved, "profitable"].mean()
                )
                if approved.any()
                else float("nan"),
            }
        )
    return pd.DataFrame.from_records(records), predictions


def walkforward_daily_predictions(
    events: pd.DataFrame,
    comparison_start: str | pd.Timestamp,
    comparison_end: str | pd.Timestamp,
) -> pd.DataFrame:
    ordered = events.copy()
    for column in ["signal_date", "label_available_date", "trading_start"]:
        ordered[column] = pd.to_datetime(ordered[column])
    comparison = ordered.loc[
        ordered["signal_date"].between(
            pd.Timestamp(comparison_start), pd.Timestamp(comparison_end)
        )
    ]
    frames: list[pd.DataFrame] = []
    for _, window_events in comparison.groupby("window_id", sort=False):
        prediction_start = window_events["trading_start"].min()
        training = ordered.loc[ordered["label_available_date"].lt(prediction_start)]
        predicted = window_events.copy()
        for name, model in daily_candidate_models().items():
            model = _fit_model(name, model, training)
            probability = model.predict_proba(
                window_events[DAILY_FEATURE_COLUMNS]
            )[:, 1]
            predicted[f"{name}_probability"] = probability
            predicted[f"{name}_approved"] = probability >= PROBABILITY_THRESHOLD
        predicted["training_rows"] = len(training)
        frames.append(predicted)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["signal_date", "event_id"]
    ).reset_index(drop=True)


def build_daily_trade_path(
    prices: pd.DataFrame,
    event: Mapping[str, object],
) -> pd.DataFrame:
    """Reconstruct one signal's gross path; costs are applied by evaluation."""
    left_code = int(event["left_code"])
    right_code = int(event["right_code"])
    signal_date = pd.Timestamp(event["signal_date"])
    exit_date = pd.Timestamp(event["exit_date"])
    pair = prices.loc[
        prices["date"].between(signal_date, exit_date)
        & prices["capitaline_code"].isin([left_code, right_code]),
        ["date", "capitaline_code", "close"],
    ].pivot(index="date", columns="capitaline_code", values="close").dropna()
    if len(pair) < 2:
        raise ValueError("A signal trade path requires at least two price dates.")
    beta = float(event["absolute_hedge_ratio"])
    denominator = 1.0 + beta
    long_spread_return = (
        pair[left_code].pct_change() / denominator
        - beta * pair[right_code].pct_change() / denominator
    ).iloc[1:]
    gross_return = int(event["direction"]) * long_spread_return
    turnover = pd.Series(0.0, index=gross_return.index)
    turnover.iloc[0] += 1.0
    turnover.iloc[-1] += 1.0
    return pd.DataFrame(
        {
            "event_id": event["event_id"],
            "date": gross_return.index,
            "gross_return": gross_return.to_numpy(),
            "turnover": turnover.to_numpy(),
        }
    )
