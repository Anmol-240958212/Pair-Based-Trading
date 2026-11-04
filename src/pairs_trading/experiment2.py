"""Two-stage, leakage-safe models for the post-hoc strategy experiment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .experiment2_config import (
    MAXIMUM_DYNAMIC_BETA_DRIFT,
    MAXIMUM_FORMATION_BETA_DRIFT,
    MAXIMUM_RECENT_Z_VOLATILITY_RATIO,
    MINIMUM_EXPECTED_NET_RETURN,
    REVERSION_PROBABILITY_THRESHOLD,
)
from .ml import EXPERIMENT2_FEATURE_COLUMNS


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                EXPERIMENT2_FEATURE_COLUMNS,
            )
        ],
        remainder="drop",
    )


def two_stage_models() -> tuple[Pipeline, Pipeline]:
    """Return an interpretable reversion classifier and net-return regressor."""
    classifier = Pipeline(
        [
            ("prepare", _preprocessor()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=2_000, C=0.5
                ),
            ),
        ]
    )
    regressor = Pipeline(
        [
            ("prepare", _preprocessor()),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    return classifier, regressor


def structural_stability_gate(events: pd.DataFrame) -> pd.Series:
    """Reject only large, observable relationship deterioration at entry."""
    return (
        events["formation_hedge_ratio_drift"].le(MAXIMUM_FORMATION_BETA_DRIFT)
        & events["dynamic_hedge_ratio_drift"].le(MAXIMUM_DYNAMIC_BETA_DRIFT)
        & events["recent_z_volatility_ratio_20d_60d"].le(
            MAXIMUM_RECENT_Z_VOLATILITY_RATIO
        )
    ).fillna(False)


def _predict_with_models(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
) -> pd.DataFrame:
    if training["reverted"].nunique() < 2:
        raise ValueError("Training data must contain reverted and non-reverted trades.")
    classifier, regressor = two_stage_models()
    classifier.fit(
        training[EXPERIMENT2_FEATURE_COLUMNS], training["reverted"].astype(int)
    )
    regressor.fit(
        training[EXPERIMENT2_FEATURE_COLUMNS], training["trade_net_return"]
    )
    result = prediction.copy()
    result["predicted_reversion_probability"] = classifier.predict_proba(
        prediction[EXPERIMENT2_FEATURE_COLUMNS]
    )[:, 1]
    result["predicted_net_return"] = regressor.predict(
        prediction[EXPERIMENT2_FEATURE_COLUMNS]
    )
    result["passes_structural_stability"] = structural_stability_gate(result)
    result["approved_by_experiment2"] = (
        result["predicted_reversion_probability"].ge(
            REVERSION_PROBABILITY_THRESHOLD
        )
        & result["predicted_net_return"].ge(MINIMUM_EXPECTED_NET_RETURN)
        & result["passes_structural_stability"]
    )
    pair_volatility = np.sqrt(
        result["left_return_volatility_20d"].pow(2)
        + result["right_return_volatility_20d"].pow(2)
    ).clip(lower=1e-6)
    result["signal_pair_volatility"] = pair_volatility
    result["selection_score"] = (
        result["predicted_reversion_probability"]
        * result["predicted_net_return"].clip(lower=0)
        / pair_volatility
    )
    result["training_events"] = len(training)
    return result


def chronological_validation(
    events: pd.DataFrame,
    training_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Evaluate the fixed two-stage rule on the immediately following period."""
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
    predicted = _predict_with_models(training, validation)
    approved = predicted.loc[predicted["approved_by_experiment2"]]
    binary = predicted["predicted_reversion_probability"].ge(0.5).astype(int)
    metrics: dict[str, float | int] = {
        "training_events": int(len(training)),
        "validation_events": int(len(validation)),
        "approved_trades": int(len(approved)),
        "approval_rate": float(len(approved) / len(validation)),
        "reversion_accuracy": float(accuracy_score(validation["reverted"], binary)),
        "reversion_precision": float(
            precision_score(validation["reverted"], binary, zero_division=0)
        ),
        "reversion_recall": float(
            recall_score(validation["reverted"], binary, zero_division=0)
        ),
        "reversion_roc_auc": float(
            roc_auc_score(
                validation["reverted"], predicted["predicted_reversion_probability"]
            )
        )
        if validation["reverted"].nunique() == 2
        else float("nan"),
        "reversion_brier_score": float(
            brier_score_loss(
                validation["reverted"], predicted["predicted_reversion_probability"]
            )
        ),
        "return_mean_absolute_error": float(
            mean_absolute_error(
                validation["trade_net_return"], predicted["predicted_net_return"]
            )
        ),
        "all_mean_trade_return": float(validation["trade_net_return"].mean()),
        "approved_mean_trade_return": float(approved["trade_net_return"].mean())
        if len(approved)
        else float("nan"),
        "approved_profit_rate": float(approved["profitable"].mean())
        if len(approved)
        else float("nan"),
    }
    return metrics, predicted


def walkforward_two_stage_predictions(
    events: pd.DataFrame,
    comparison_start: str | pd.Timestamp,
    comparison_end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Refit before each window using only outcomes already known at that date."""
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
        frames.append(_predict_with_models(training, window_events))
    return pd.concat(frames, ignore_index=True).sort_values(
        ["signal_date", "event_id"]
    ).reset_index(drop=True)
