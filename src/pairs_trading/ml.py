"""Leakage-safe event labelling and classifier comparison for pair signals."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier
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
from sklearn.tree import DecisionTreeClassifier


FEATURE_COLUMNS = [
    "entry_z_score",
    "absolute_entry_z_score",
    "cointegration_p_value",
    "return_correlation",
    "absolute_hedge_ratio",
    "spread_half_life_days",
    "spread_change_volatility_20d",
    "z_score_momentum_5d",
    "recent_return_correlation_20d",
    "recent_return_correlation_60d",
    "spread_volatility_ratio_20d_60d",
    "absolute_z_momentum_1d",
    "maximum_absolute_z_20d",
    "divergence_duration_days",
    "pair_momentum_gap_20d",
    "left_return_volatility_20d",
    "right_return_volatility_20d",
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


def build_labeled_events(
    formation_metrics: Mapping[str, float | int],
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    left_code: int,
    right_code: int,
    industry: str | None = None,
) -> pd.DataFrame:
    """Turn completed historical trades into one-row binary ML examples.

    Feature values are calculated at ``signal_date`` using only observations on
    or before that date. The future trade outcome is used only to create labels.
    """
    if trades.empty:
        return pd.DataFrame()

    required_daily = {
        "date",
        "left_close",
        "right_close",
        "spread",
        "z_score",
    }
    required_trades = {
        "signal_date",
        "entry_date",
        "exit_date",
        "entry_z_score",
        "exit_reason",
        "net_return",
    }
    if missing := required_daily.difference(daily.columns):
        raise ValueError(f"Daily results are missing columns: {sorted(missing)}")
    if missing := required_trades.difference(trades.columns):
        raise ValueError(f"Trade results are missing columns: {sorted(missing)}")

    history = daily.sort_values("date").copy()
    history["date"] = pd.to_datetime(history["date"])
    history["spread_change_volatility_20d"] = (
        history["spread"].diff().rolling(20, min_periods=10).std(ddof=1)
    )
    history["z_score_momentum_5d"] = history["z_score"].diff(5)
    left_return = history["left_close"].pct_change()
    right_return = history["right_close"].pct_change()
    history["recent_return_correlation_20d"] = left_return.rolling(
        20, min_periods=10
    ).corr(right_return)
    history["recent_return_correlation_60d"] = left_return.rolling(
        60, min_periods=30
    ).corr(right_return)
    spread_volatility_60d = history["spread"].diff().rolling(60, min_periods=30).std(ddof=1)
    history["spread_volatility_ratio_20d_60d"] = (
        history["spread_change_volatility_20d"] / spread_volatility_60d
    )
    history["absolute_z_momentum_1d"] = history["z_score"].abs().diff()
    history["maximum_absolute_z_20d"] = history["z_score"].abs().rolling(
        20, min_periods=5
    ).max()
    history["divergence_duration_days"] = _consecutive_true_count(
        history["z_score"].abs().ge(1.0)
    )
    history["pair_momentum_gap_20d"] = (
        history["left_close"].pct_change(20)
        - history["right_close"].pct_change(20)
    ).abs()
    history["left_return_volatility_20d"] = left_return.rolling(
        20, min_periods=10
    ).std(ddof=1)
    history["right_return_volatility_20d"] = right_return.rolling(
        20, min_periods=10
    ).std(ddof=1)
    signal_features = history.set_index("date")

    records: list[dict[str, object]] = []
    for trade in trades.itertuples(index=False):
        signal_date = pd.Timestamp(trade.signal_date)
        if signal_date not in signal_features.index:
            raise ValueError(f"Signal date {signal_date.date()} is absent from daily results.")
        signal = signal_features.loc[signal_date]
        if isinstance(signal, pd.DataFrame):
            raise ValueError("Daily results contain duplicate dates.")

        reverted = trade.exit_reason == "mean_reversion"
        profitable = float(trade.net_return) > 0
        records.append(
            {
                "left_code": int(left_code),
                "right_code": int(right_code),
                "industry": industry,
                "signal_date": signal_date,
                "entry_date": pd.Timestamp(trade.entry_date),
                "exit_date": pd.Timestamp(trade.exit_date),
                "label_available_date": pd.Timestamp(trade.exit_date),
                "entry_z_score": float(trade.entry_z_score),
                "absolute_entry_z_score": abs(float(trade.entry_z_score)),
                "cointegration_p_value": float(
                    formation_metrics["cointegration_p_value"]
                ),
                "return_correlation": float(formation_metrics["return_correlation"]),
                "absolute_hedge_ratio": abs(float(formation_metrics["hedge_ratio"])),
                "spread_half_life_days": float(
                    formation_metrics["spread_half_life_days"]
                ),
                "spread_change_volatility_20d": float(
                    signal["spread_change_volatility_20d"]
                ),
                "z_score_momentum_5d": float(signal["z_score_momentum_5d"]),
                "recent_return_correlation_20d": float(
                    signal["recent_return_correlation_20d"]
                ),
                "recent_return_correlation_60d": float(
                    signal["recent_return_correlation_60d"]
                ),
                "spread_volatility_ratio_20d_60d": float(
                    signal["spread_volatility_ratio_20d_60d"]
                ),
                "absolute_z_momentum_1d": float(signal["absolute_z_momentum_1d"]),
                "maximum_absolute_z_20d": float(signal["maximum_absolute_z_20d"]),
                "divergence_duration_days": int(signal["divergence_duration_days"]),
                "pair_momentum_gap_20d": float(signal["pair_momentum_gap_20d"]),
                "left_return_volatility_20d": float(
                    signal["left_return_volatility_20d"]
                ),
                "right_return_volatility_20d": float(
                    signal["right_return_volatility_20d"]
                ),
                "exit_reason": trade.exit_reason,
                "trade_net_return": float(trade.net_return),
                "reverted": int(reverted),
                "profitable": int(profitable),
                "target": int(reverted and profitable),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("signal_date").reset_index(drop=True)


def add_market_industry_features(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    company_master: pd.DataFrame,
    pair_screens: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add point-in-time market, industry and formation-liquidity context."""
    if events.empty:
        return events.copy()
    required_prices = {"date", "capitaline_code", "close"}
    if missing := required_prices.difference(prices.columns):
        raise ValueError(f"Prices are missing columns: {sorted(missing)}")

    panel = prices[["date", "capitaline_code", "close"]].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["capitaline_code", "date"])
    panel["stock_return"] = panel.groupby("capitaline_code")["close"].pct_change()
    panel = panel.merge(
        company_master[["capitaline_code", "industry"]],
        on="capitaline_code",
        how="left",
    )

    market = (
        panel.groupby("date", as_index=False)["stock_return"]
        .agg(market_return="mean", cross_sectional_dispersion="std")
        .sort_values("date")
    )
    market["market_momentum_20d"] = market["market_return"].rolling(
        20, min_periods=10
    ).sum()
    market["market_volatility_20d"] = market["market_return"].rolling(
        20, min_periods=10
    ).std(ddof=1)
    market_volatility_60d = market["market_return"].rolling(
        60, min_periods=30
    ).std(ddof=1)
    market["market_volatility_ratio_20d_60d"] = (
        market["market_volatility_20d"] / market_volatility_60d
    )
    market["cross_sectional_dispersion_20d"] = market[
        "cross_sectional_dispersion"
    ].rolling(20, min_periods=10).mean()

    industry = (
        panel.dropna(subset=["industry"])
        .groupby(["industry", "date"], as_index=False)["stock_return"]
        .mean()
        .sort_values(["industry", "date"])
    )
    industry["industry_momentum_20d"] = industry.groupby("industry")[
        "stock_return"
    ].transform(lambda values: values.rolling(20, min_periods=10).sum())
    industry["industry_volatility_20d"] = industry.groupby("industry")[
        "stock_return"
    ].transform(lambda values: values.rolling(20, min_periods=10).std(ddof=1))

    enriched = events.copy()
    enriched["signal_date"] = pd.to_datetime(enriched["signal_date"])
    enriched = enriched.merge(
        market[
            [
                "date",
                "market_momentum_20d",
                "market_volatility_20d",
                "market_volatility_ratio_20d_60d",
                "cross_sectional_dispersion_20d",
            ]
        ],
        left_on="signal_date",
        right_on="date",
        how="left",
    ).drop(columns="date")
    enriched = enriched.merge(
        industry[
            [
                "industry",
                "date",
                "industry_momentum_20d",
                "industry_volatility_20d",
            ]
        ],
        left_on=["industry", "signal_date"],
        right_on=["industry", "date"],
        how="left",
    ).drop(columns="date")

    enriched["minimum_pair_median_turnover"] = np.nan
    if pair_screens is not None and not pair_screens.empty:
        liquidity_columns = {
            "window_id",
            "left_code",
            "right_code",
            "minimum_pair_median_turnover",
        }
        if liquidity_columns.issubset(pair_screens.columns):
            enriched = enriched.drop(columns="minimum_pair_median_turnover").merge(
                pair_screens[list(liquidity_columns)].drop_duplicates(),
                on=["window_id", "left_code", "right_code"],
                how="left",
            )
    return enriched.sort_values("signal_date").reset_index(drop=True)


def candidate_classifiers(random_state: int = 42) -> dict[str, ClassifierMixin]:
    """Return the three pre-declared candidate models on comparable pipelines."""
    numeric_preprocessor = ColumnTransformer(
        [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                ("scale", StandardScaler())]), FEATURE_COLUMNS)],
        remainder="drop",
    )
    tree_preprocessor = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), FEATURE_COLUMNS)],
        remainder="drop",
    )
    return {
        "logistic_regression": Pipeline(
            [
                ("prepare", numeric_preprocessor),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2_000)),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("prepare", tree_preprocessor),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=4,
                        min_samples_leaf=10,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "bagging_decision_tree": Pipeline(
            [
                ("prepare", tree_preprocessor),
                (
                    "model",
                    BaggingClassifier(
                        estimator=DecisionTreeClassifier(
                            max_depth=4,
                            min_samples_leaf=10,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                        n_estimators=200,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def compare_classifiers_chronologically(
    events: pd.DataFrame,
    training_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    probability_threshold: float = 0.5,
    random_state: int = 42,
    target_column: str = "target",
) -> tuple[pd.DataFrame, dict[str, ClassifierMixin], pd.DataFrame]:
    """Train on the past and compare models on the immediately following period."""
    if not 0 < probability_threshold < 1:
        raise ValueError("The probability threshold must lie strictly between 0 and 1.")
    if events.empty:
        raise ValueError("No labelled events were supplied.")
    if missing := set(FEATURE_COLUMNS + ["signal_date", target_column]).difference(events.columns):
        raise ValueError(f"Events are missing columns: {sorted(missing)}")

    ordered = events.copy()
    ordered["signal_date"] = pd.to_datetime(ordered["signal_date"])
    training_end = pd.Timestamp(training_end)
    validation_end = pd.Timestamp(validation_end)
    if validation_end <= training_end:
        raise ValueError("Validation must end after training.")

    label_date_column = (
        "label_available_date" if "label_available_date" in ordered else "signal_date"
    )
    ordered[label_date_column] = pd.to_datetime(ordered[label_date_column])
    train = ordered.loc[ordered[label_date_column] <= training_end]
    validation = ordered.loc[
        ordered["signal_date"].gt(training_end)
        & ordered["signal_date"].le(validation_end)
        & ordered[label_date_column].le(validation_end)
    ]
    if train.empty or validation.empty:
        raise ValueError("Both chronological training and validation sets must be non-empty.")
    if train[target_column].nunique() < 2:
        raise ValueError("Training events must contain both successful and failed trades.")

    fitted: dict[str, ClassifierMixin] = {}
    retained_columns = [
        column
        for column in [
            "window_id",
            "left_code",
            "right_code",
            "signal_date",
            "exit_date",
            target_column,
            "trade_net_return",
        ]
        if column in validation
    ]
    predictions = validation[retained_columns].copy()
    results: list[dict[str, float | int | str]] = []
    for name, model in candidate_classifiers(random_state).items():
        model.fit(train[FEATURE_COLUMNS], train[target_column].astype(int))
        probability = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
        prediction = (probability >= probability_threshold).astype(int)
        predictions[f"{name}_probability"] = probability
        predictions[f"{name}_prediction"] = prediction
        auc = (
            float(roc_auc_score(validation[target_column], probability))
            if validation[target_column].nunique() == 2
            else float("nan")
        )
        results.append(
            {
                "model": name,
                "training_events": int(len(train)),
                "validation_events": int(len(validation)),
                "approval_rate": float(prediction.mean()),
                "target": target_column,
                "accuracy": float(accuracy_score(validation[target_column], prediction)),
                "precision": float(
                    precision_score(validation[target_column], prediction, zero_division=0)
                ),
                "recall": float(
                    recall_score(validation[target_column], prediction, zero_division=0)
                ),
                "roc_auc": auc,
                "brier_score": float(
                    brier_score_loss(validation[target_column], probability)
                ),
                "approved_mean_trade_return": float(
                    validation.loc[prediction.astype(bool), "trade_net_return"].mean()
                )
                if "trade_net_return" in validation and prediction.any()
                else float("nan"),
                "all_signals_mean_trade_return": float(
                    validation["trade_net_return"].mean()
                )
                if "trade_net_return" in validation
                else float("nan"),
            }
        )
        fitted[name] = model

    comparison = pd.DataFrame.from_records(results).sort_values(
        ["roc_auc", "brier_score"], ascending=[False, True], na_position="last"
    )
    return comparison.reset_index(drop=True), fitted, predictions.reset_index(drop=True)


def tune_validation_thresholds(
    events: pd.DataFrame,
    training_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    thresholds: tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
    target_column: str = "target",
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare fixed probability thresholds using validation data only."""
    ordered = events.copy()
    ordered["signal_date"] = pd.to_datetime(ordered["signal_date"])
    label_column = (
        "label_available_date" if "label_available_date" in ordered else "signal_date"
    )
    ordered[label_column] = pd.to_datetime(ordered[label_column])
    training_end = pd.Timestamp(training_end)
    validation_end = pd.Timestamp(validation_end)
    train = ordered.loc[ordered[label_column].le(training_end)]
    validation = ordered.loc[
        ordered["signal_date"].gt(training_end)
        & ordered["signal_date"].le(validation_end)
        & ordered[label_column].le(validation_end)
    ].copy()
    if train.empty or validation.empty or train[target_column].nunique() < 2:
        raise ValueError("The chronological split does not contain usable classes.")

    records: list[dict[str, object]] = []
    for model_name, model in candidate_classifiers(random_state).items():
        model.fit(train[FEATURE_COLUMNS], train[target_column].astype(int))
        probabilities = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
        auc = (
            float(roc_auc_score(validation[target_column], probabilities))
            if validation[target_column].nunique() == 2
            else float("nan")
        )
        for threshold in thresholds:
            approved = probabilities >= threshold
            approved_returns = validation.loc[approved, "trade_net_return"]
            yearly = validation.loc[approved].groupby(
                validation.loc[approved, "signal_date"].dt.year
            )["trade_net_return"].mean()
            return_std = float(approved_returns.std(ddof=1))
            records.append(
                {
                    "model": model_name,
                    "target": target_column,
                    "probability_threshold": threshold,
                    "training_events": int(len(train)),
                    "validation_events": int(len(validation)),
                    "approved_trades": int(approved.sum()),
                    "approval_rate": float(approved.mean()),
                    "precision": float(
                        precision_score(
                            validation[target_column], approved.astype(int), zero_division=0
                        )
                    ),
                    "recall": float(
                        recall_score(
                            validation[target_column], approved.astype(int), zero_division=0
                        )
                    ),
                    "roc_auc": auc,
                    "mean_trade_return": float(approved_returns.mean())
                    if approved.any()
                    else float("nan"),
                    "trade_return_sharpe": float(
                        np.sqrt(len(approved_returns))
                        * approved_returns.mean()
                        / return_std
                    )
                    if len(approved_returns) > 1 and return_std > 0
                    else float("nan"),
                    "worst_calendar_year_mean_return": float(yearly.min())
                    if not yearly.empty
                    else float("nan"),
                }
            )
    return pd.DataFrame.from_records(records)


def walkforward_predict_locked_test(
    events: pd.DataFrame,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
    model_name: str,
    target_column: str,
    probability_threshold: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """Predict each test window using only labels available before that window."""
    required = set(
        FEATURE_COLUMNS
        + [
            "window_id",
            "signal_date",
            "label_available_date",
            "trading_start",
            target_column,
        ]
    )
    if missing := required.difference(events.columns):
        raise ValueError(f"Events are missing columns: {sorted(missing)}")
    if model_name not in candidate_classifiers(random_state):
        raise ValueError(f"Unknown classifier: {model_name}")

    ordered = events.copy()
    for column in ["signal_date", "label_available_date", "trading_start"]:
        ordered[column] = pd.to_datetime(ordered[column])
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)
    test = ordered.loc[
        ordered["signal_date"].between(test_start, test_end)
    ].copy()
    records: list[pd.DataFrame] = []
    for window_id, window_events in test.groupby("window_id", sort=False):
        prediction_start = window_events["trading_start"].min()
        training = ordered.loc[
            ordered["label_available_date"].lt(prediction_start)
        ]
        if training[target_column].nunique() < 2:
            raise ValueError(f"Window {window_id} has an unusable training sample.")
        model = candidate_classifiers(random_state)[model_name]
        model.fit(training[FEATURE_COLUMNS], training[target_column].astype(int))
        probability = model.predict_proba(window_events[FEATURE_COLUMNS])[:, 1]
        predicted = window_events.copy()
        predicted["training_events"] = len(training)
        predicted["model"] = model_name
        predicted["target_name"] = target_column
        predicted["predicted_probability"] = probability
        predicted["approved_by_ml"] = probability >= probability_threshold
        records.append(predicted)
    return pd.concat(records, ignore_index=True).sort_values(
        ["signal_date", "event_id"]
    ).reset_index(drop=True)
