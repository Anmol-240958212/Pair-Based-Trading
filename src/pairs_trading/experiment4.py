"""Discriminant, Bayesian and Markov-regime models for the daily pair panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .experiment3 import DAILY_FEATURE_COLUMNS
from .experiment4_config import PROBABILITY_THRESHOLD


REGIME_INPUT_COLUMNS = [
    "market_momentum_20d",
    "market_volatility_20d",
    "market_volatility_ratio_20d_60d",
    "cross_sectional_dispersion_20d",
    "industry_momentum_20d",
    "industry_volatility_20d",
]
REGIME_FEATURE_COLUMNS = [
    "markov_high_volatility_probability",
    "markov_favorable_probability",
]
MODEL_NAMES = [
    "logistic_regression",
    "shrinkage_lda",
    "gaussian_naive_bayes",
    "markov_regime_logistic",
]


def _daily_regime_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    daily = frame.copy()
    daily["signal_date"] = pd.to_datetime(daily["signal_date"])
    return (
        daily.groupby("signal_date", as_index=False)[REGIME_INPUT_COLUMNS]
        .mean()
        .sort_values("signal_date")
        .reset_index(drop=True)
    )


def _normal_log_density(
    values: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    difference = values[:, None, :] - means[None, :, :]
    return -0.5 * (
        np.log(2.0 * np.pi * variances)[None, :, :]
        + difference**2 / variances[None, :, :]
    ).sum(axis=2)


def _forward_filter(
    values: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    transition: np.ndarray,
    initial_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided state probabilities without using future observations."""
    log_emission = _normal_log_density(values, means, variances)
    probabilities = np.empty((len(values), len(initial_probability)))
    previous = initial_probability.astype(float)
    for index, emission_log in enumerate(log_emission):
        prior = previous if index == 0 else previous @ transition
        shifted = emission_log - emission_log.max()
        posterior = prior * np.exp(shifted)
        total = posterior.sum()
        posterior = posterior / total if total > 0 else prior / prior.sum()
        probabilities[index] = posterior
        previous = posterior
    return probabilities, previous


def add_markov_regime_features(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a two-state Gaussian Markov filter on training dates only.

    Gaussian emissions are estimated by a mixture model. State transitions are
    estimated from the training-state sequence with Laplace smoothing. Both the
    training and prediction probabilities are computed by a one-sided forward
    filter, so later dates never affect earlier state probabilities.
    """
    train_daily = _daily_regime_matrix(training)
    predict_daily = _daily_regime_matrix(prediction)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = scaler.fit_transform(imputer.fit_transform(train_daily[REGIME_INPUT_COLUMNS]))
    predict_values = scaler.transform(imputer.transform(predict_daily[REGIME_INPUT_COLUMNS]))

    mixture = GaussianMixture(
        n_components=2,
        covariance_type="diag",
        reg_covar=1e-4,
        n_init=3,
        max_iter=100,
        random_state=random_state,
    ).fit(train_values)
    hard_states = mixture.predict(train_values)
    transition_counts = np.ones((2, 2), dtype=float)
    for previous, current in zip(hard_states[:-1], hard_states[1:]):
        transition_counts[int(previous), int(current)] += 1.0
    transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)

    train_probabilities, last_probability = _forward_filter(
        train_values,
        mixture.means_,
        mixture.covariances_,
        transition,
        mixture.weights_,
    )
    # Continue from the final training posterior into the later prediction dates.
    predict_probabilities, _ = _forward_filter(
        predict_values,
        mixture.means_,
        mixture.covariances_,
        transition,
        last_probability @ transition,
    )

    market_volatility_index = REGIME_INPUT_COLUMNS.index("market_volatility_20d")
    high_volatility_state = int(np.argmax(mixture.means_[:, market_volatility_index]))
    training_probability_map = pd.DataFrame(
        {
            "signal_date": train_daily["signal_date"],
            "state_0_probability": train_probabilities[:, 0],
            "state_1_probability": train_probabilities[:, 1],
        }
    )
    training_with_states = training.copy()
    training_with_states["signal_date"] = pd.to_datetime(training_with_states["signal_date"])
    training_with_states = training_with_states.merge(
        training_probability_map, on="signal_date", how="left", validate="many_to_one"
    )
    success_rates = []
    for state in range(2):
        weights = training_with_states[f"state_{state}_probability"].to_numpy()
        success_rates.append(float(np.average(training_with_states["target"], weights=weights)))
    favorable_state = int(np.argmax(success_rates))

    def attach(
        source: pd.DataFrame, daily: pd.DataFrame, probabilities: np.ndarray
    ) -> pd.DataFrame:
        mapping = pd.DataFrame(
            {
                "signal_date": daily["signal_date"],
                "markov_high_volatility_probability": probabilities[:, high_volatility_state],
                "markov_favorable_probability": probabilities[:, favorable_state],
            }
        )
        result = source.copy()
        result["signal_date"] = pd.to_datetime(result["signal_date"])
        return result.merge(mapping, on="signal_date", how="left", validate="many_to_one")

    return attach(training, train_daily, train_probabilities), attach(
        prediction, predict_daily, predict_probabilities
    )


def _pipeline(model_name: str, feature_columns: list[str]) -> Pipeline:
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
                feature_columns,
            )
        ],
        remainder="drop",
    )
    if model_name in {"logistic_regression", "markov_regime_logistic"}:
        estimator = LogisticRegression(class_weight="balanced", C=0.5, max_iter=2_000)
    elif model_name == "shrinkage_lda":
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif model_name == "gaussian_naive_bayes":
        estimator = GaussianNB(var_smoothing=1e-8)
    else:
        raise ValueError(f"Unknown Experiment 4 model: {model_name}")
    return Pipeline([("prepare", preparation), ("model", estimator)])


def model_probabilities(
    model_name: str, training: pd.DataFrame, prediction: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    train_frame = training
    prediction_frame = prediction
    features = list(DAILY_FEATURE_COLUMNS)
    if model_name == "markov_regime_logistic":
        train_frame, prediction_frame = add_markov_regime_features(training, prediction)
        features += REGIME_FEATURE_COLUMNS
    model = _pipeline(model_name, features)
    model.fit(train_frame[features], train_frame["target"])
    probability = model.predict_proba(prediction_frame[features])[:, 1]
    return probability, prediction_frame


def select_validation_threshold(
    validation: pd.DataFrame, probability: np.ndarray
) -> tuple[float, float, int]:
    """Apply the pre-declared cutoff used by the preceding daily-panel study."""
    approved = probability >= PROBABILITY_THRESHOLD
    count = int(approved.sum())
    mean_return = (
        float(validation.loc[approved, "trade_net_return"].mean())
        if count
        else float("nan")
    )
    return PROBABILITY_THRESHOLD, mean_return, count


def validation_comparison(
    events: pd.DataFrame,
    training_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
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
    ].copy()
    predictions = validation[
        [
            "event_id", "window_id", "left_code", "right_code", "signal_date",
            "entry_date", "exit_date", "target", "trade_net_return", "profitable",
        ]
    ].copy()
    records: list[dict[str, object]] = []
    thresholds: dict[str, float] = {}
    for model_name in MODEL_NAMES:
        probability, _ = model_probabilities(model_name, training, validation)
        threshold, approved_return, approved_count = select_validation_threshold(
            validation, probability
        )
        thresholds[model_name] = threshold
        approved = probability >= threshold
        classification = probability >= 0.5
        predictions[f"{model_name}_probability"] = probability
        predictions[f"{model_name}_approved"] = approved
        records.append(
            {
                "model": model_name,
                "training_rows": int(len(training)),
                "validation_rows": int(len(validation)),
                "selected_threshold": threshold,
                "approved_rows": approved_count,
                "roc_auc": float(roc_auc_score(validation["target"], probability)),
                "brier_score": float(brier_score_loss(validation["target"], probability)),
                "precision_at_0_5": float(
                    precision_score(validation["target"], classification, zero_division=0)
                ),
                "recall_at_0_5": float(
                    recall_score(validation["target"], classification, zero_division=0)
                ),
                "approved_mean_return": approved_return,
                "approved_profit_rate": float(validation.loc[approved, "profitable"].mean()),
            }
        )
    return pd.DataFrame.from_records(records), predictions, thresholds


def walkforward_predictions(
    events: pd.DataFrame,
    thresholds: dict[str, float],
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
        for model_name in MODEL_NAMES:
            probability, augmented = model_probabilities(model_name, training, window_events)
            if model_name == "markov_regime_logistic":
                for feature in REGIME_FEATURE_COLUMNS:
                    predicted[feature] = augmented[feature].to_numpy()
            predicted[f"{model_name}_probability"] = probability
            predicted[f"{model_name}_approved"] = probability >= thresholds[model_name]
        predicted["training_rows"] = len(training)
        frames.append(predicted)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["signal_date", "event_id"]
    ).reset_index(drop=True)

