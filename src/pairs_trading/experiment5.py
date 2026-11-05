"""Delayed confirmation features and leakage-safe walk-forward evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .experiment3 import DAILY_FEATURE_COLUMNS
from .experiment4 import REGIME_FEATURE_COLUMNS, add_markov_regime_features
from .experiment5_config import (
    MAXIMUM_OUTWARD_Z_CHANGE,
    MAXIMUM_STANDARDIZED_IDIOSYNCRATIC_SHOCK,
    MINIMUM_FACTOR_ALIGNMENT_FRACTION,
    MINIMUM_SAME_DIRECTION_FRACTION,
    PROBABILITY_THRESHOLD,
)


CONFIRMATION_FEATURE_COLUMNS = [
    "observation_days",
    "initial_z_score",
    "absolute_z_convergence",
    "same_direction_fraction",
    "factor_alignment_fraction",
    "lead_lag_agreement_fraction",
    "mean_common_factor_share",
    "maximum_standardized_idiosyncratic_shock",
    "maximum_abnormal_turnover_ratio",
]
EXPERIMENT5_FEATURE_COLUMNS = (
    DAILY_FEATURE_COLUMNS + REGIME_FEATURE_COLUMNS + CONFIRMATION_FEATURE_COLUMNS
)


def prepare_return_context(
    prices: pd.DataFrame, company_master: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create daily stock, industry and turnover-shock context."""
    frame = prices[["capitaline_code", "date", "close", "turnover_rs_thousand"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["capitaline_code", "date"])
    frame["stock_return"] = frame.groupby("capitaline_code")["close"].pct_change()
    frame["turnover_median_60d"] = frame.groupby("capitaline_code")[
        "turnover_rs_thousand"
    ].transform(lambda values: values.rolling(60, min_periods=20).median())
    frame["abnormal_turnover_ratio"] = (
        frame["turnover_rs_thousand"] / frame["turnover_median_60d"].replace(0, np.nan)
    )
    frame = frame.merge(
        company_master[["capitaline_code", "industry"]],
        on="capitaline_code",
        how="left",
        validate="many_to_one",
    )
    industry = (
        frame.groupby(["date", "industry"], as_index=False)["stock_return"]
        .median()
        .rename(columns={"stock_return": "industry_return"})
    )
    frame = frame.merge(industry, on=["date", "industry"], how="left")
    frame["industry_residual"] = frame["stock_return"] - frame["industry_return"]
    trading_dates = pd.DataFrame({"date": sorted(frame["date"].dropna().unique())})
    trading_dates["trading_day_index"] = np.arange(len(trading_dates), dtype=int)
    frame = frame.merge(trading_dates, on="date", how="left", validate="many_to_one")
    return frame, trading_dates


def build_delayed_confirmation_rows(
    panel: pd.DataFrame,
    return_context: pd.DataFrame,
    trading_dates: pd.DataFrame,
    delay: int,
) -> pd.DataFrame:
    """Move each trigger to a later observable signal and attach path diagnostics."""
    rows = panel.copy()
    for column in ["signal_date", "entry_date", "exit_date", "label_available_date"]:
        rows[column] = pd.to_datetime(rows[column])
    date_to_index = trading_dates.set_index("date")["trading_day_index"]
    rows["confirmation_index"] = rows["signal_date"].map(date_to_index)

    if delay == 0:
        result = rows.copy()
        result["trigger_date"] = result["signal_date"]
        result["observation_days"] = 0
        result["initial_z_score"] = result["entry_z_score"]
        result["absolute_z_convergence"] = 0.0
        result["same_direction_fraction"] = 1.0
        result["factor_alignment_fraction"] = 1.0
        result["lead_lag_agreement_fraction"] = 0.5
        result["mean_common_factor_share"] = 0.5
        result["maximum_standardized_idiosyncratic_shock"] = 0.0
        result["maximum_abnormal_turnover_ratio"] = 1.0
        result["passes_confirmation"] = True
        return result

    initial = rows[
        ["window_id", "left_code", "right_code", "signal_date", "entry_z_score", "confirmation_index"]
    ].rename(
        columns={
            "signal_date": "trigger_date",
            "entry_z_score": "initial_z_score",
            "confirmation_index": "trigger_index",
        }
    )
    initial["confirmation_index"] = initial["trigger_index"] + delay
    result = rows.merge(
        initial,
        on=["window_id", "left_code", "right_code", "confirmation_index"],
        how="inner",
        validate="one_to_one",
    )
    result = result.loc[result["initial_z_score"] * result["entry_z_score"] > 0].copy()
    result["observation_days"] = delay
    result["absolute_z_convergence"] = (
        result["initial_z_score"].abs() - result["entry_z_score"].abs()
    )

    context = return_context[
        [
            "capitaline_code", "trading_day_index", "stock_return", "industry_return",
            "industry_residual", "abnormal_turnover_ratio",
        ]
    ]
    daily_frames: list[pd.DataFrame] = []
    identity = ["event_id"]
    for offset in range(1, delay + 1):
        day = result[identity + ["left_code", "right_code", "trigger_index"]].copy()
        day["trading_day_index"] = day["trigger_index"] + offset
        left = context.rename(
            columns={
                "capitaline_code": "left_code",
                "stock_return": "left_return",
                "industry_return": "factor_return",
                "industry_residual": "left_residual",
                "abnormal_turnover_ratio": "left_turnover_ratio",
            }
        )
        right = context.rename(
            columns={
                "capitaline_code": "right_code",
                "stock_return": "right_return",
                "industry_return": "right_factor_return",
                "industry_residual": "right_residual",
                "abnormal_turnover_ratio": "right_turnover_ratio",
            }
        )
        day = day.merge(left, on=["left_code", "trading_day_index"], how="left")
        day = day.merge(right, on=["right_code", "trading_day_index"], how="left")
        day["offset"] = offset
        daily_frames.append(day)
    path = pd.concat(daily_frames, ignore_index=True)
    denominator = path["left_return"].abs() + path["right_return"].abs() + 1e-8
    residual_size = path["left_residual"].abs() + path["right_residual"].abs()
    path["common_factor_share"] = (1.0 - residual_size / denominator).clip(0, 1)
    path["same_direction"] = (
        np.sign(path["left_return"]) == np.sign(path["right_return"])
    ).astype(float)
    path["factor_alignment"] = (
        (np.sign(path["left_return"]) == np.sign(path["factor_return"]))
        & (np.sign(path["right_return"]) == np.sign(path["factor_return"]))
    ).astype(float)
    left_scale = result.set_index("event_id")["left_return_volatility_20d"]
    right_scale = result.set_index("event_id")["right_return_volatility_20d"]
    path["standardized_shock"] = np.maximum(
        path["left_residual"].abs() / path["event_id"].map(left_scale).clip(lower=1e-6),
        path["right_residual"].abs() / path["event_id"].map(right_scale).clip(lower=1e-6),
    )
    path["abnormal_turnover"] = path[
        ["left_turnover_ratio", "right_turnover_ratio"]
    ].max(axis=1)
    path = path.sort_values(["event_id", "offset"])
    path["previous_left_return"] = path.groupby("event_id")["left_return"].shift(1)
    path["previous_right_return"] = path.groupby("event_id")["right_return"].shift(1)
    path["lead_lag_agreement"] = 0.5 * (
        (
            np.sign(path["previous_left_return"])
            == np.sign(path["right_return"])
        ).astype(float)
        + (
            np.sign(path["previous_right_return"])
            == np.sign(path["left_return"])
        ).astype(float)
    )
    aggregate = path.groupby("event_id", as_index=False).agg(
        same_direction_fraction=("same_direction", "mean"),
        factor_alignment_fraction=("factor_alignment", "mean"),
        lead_lag_agreement_fraction=("lead_lag_agreement", "mean"),
        mean_common_factor_share=("common_factor_share", "mean"),
        maximum_standardized_idiosyncratic_shock=("standardized_shock", "max"),
        maximum_abnormal_turnover_ratio=("abnormal_turnover", "max"),
    )
    result = result.merge(aggregate, on="event_id", how="left", validate="one_to_one")
    result["passes_confirmation"] = (
        result["same_direction_fraction"].ge(MINIMUM_SAME_DIRECTION_FRACTION)
        & result["factor_alignment_fraction"].ge(MINIMUM_FACTOR_ALIGNMENT_FRACTION)
        & result["maximum_standardized_idiosyncratic_shock"].le(
            MAXIMUM_STANDARDIZED_IDIOSYNCRATIC_SHOCK
        )
        & result["absolute_z_convergence"].ge(-MAXIMUM_OUTWARD_Z_CHANGE)
    )
    return result


def _fit_predict(
    training: pd.DataFrame, prediction: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    augmented_training, augmented_prediction = add_markov_regime_features(
        training, prediction
    )
    preparation = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                EXPERIMENT5_FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("prepare", preparation),
            ("model", LogisticRegression(class_weight="balanced", C=0.5, max_iter=2_000)),
        ]
    )
    model.fit(augmented_training[EXPERIMENT5_FEATURE_COLUMNS], augmented_training["target"])
    return (
        model.predict_proba(augmented_prediction[EXPERIMENT5_FEATURE_COLUMNS])[:, 1],
        augmented_prediction,
    )


def validation_result(
    rows: pd.DataFrame, training_end: str, validation_end: str
) -> tuple[dict[str, object], pd.DataFrame]:
    frame = rows.copy()
    for column in ["signal_date", "label_available_date"]:
        frame[column] = pd.to_datetime(frame[column])
    training = frame.loc[frame["label_available_date"].le(pd.Timestamp(training_end))]
    validation = frame.loc[
        frame["signal_date"].gt(pd.Timestamp(training_end))
        & frame["signal_date"].le(pd.Timestamp(validation_end))
        & frame["label_available_date"].le(pd.Timestamp(validation_end))
    ].copy()
    probability, augmented = _fit_predict(training, validation)
    approved = (probability >= PROBABILITY_THRESHOLD) & augmented[
        "passes_confirmation"
    ].to_numpy(dtype=bool)
    predictions = augmented.copy()
    predictions["predicted_probability"] = probability
    predictions["approved"] = approved
    record = {
        "observation_days": int(frame["observation_days"].iloc[0]),
        "training_rows": int(len(training)),
        "validation_rows": int(len(validation)),
        "confirmation_pass_rows": int(validation["passes_confirmation"].sum()),
        "approved_rows": int(approved.sum()),
        "roc_auc": float(roc_auc_score(validation["target"], probability)),
        "brier_score": float(brier_score_loss(validation["target"], probability)),
        "approved_mean_return": float(validation.loc[approved, "trade_net_return"].mean())
        if approved.any()
        else float("nan"),
        "approved_profit_rate": float(validation.loc[approved, "profitable"].mean())
        if approved.any()
        else float("nan"),
    }
    return record, predictions


def walkforward_predictions(rows: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    frame = rows.copy()
    for column in ["signal_date", "label_available_date", "trading_start"]:
        frame[column] = pd.to_datetime(frame[column])
    comparison = frame.loc[frame["signal_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    outputs: list[pd.DataFrame] = []
    for _, window in comparison.groupby("window_id", sort=False):
        prediction_start = window["trading_start"].min()
        training = frame.loc[frame["label_available_date"].lt(prediction_start)]
        probability, augmented = _fit_predict(training, window)
        augmented["predicted_probability"] = probability
        augmented["approved"] = (
            probability >= PROBABILITY_THRESHOLD
        ) & augmented["passes_confirmation"].to_numpy(dtype=bool)
        augmented["training_rows"] = len(training)
        outputs.append(augmented)
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["signal_date", "event_id"]
    ).reset_index(drop=True)

