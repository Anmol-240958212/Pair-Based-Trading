"""Point-in-time NSE announcement features and an asymmetric-event veto."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


MATERIAL_EVENT_PATTERN = re.compile(
    r"financial result|outcome of board|acquisition|merger|amalgamation|"
    r"scheme of arrangement|order|contract|credit rating|resignation|appointment|"
    r"default|litigation|regulat|fraud|fire|accident|insolvency|bankrupt|"
    r"buyback|dividend|bonus|stock split|press release|restructur|fund rais|"
    r"preferential|joint venture|disinvest|closure of operations",
    flags=re.IGNORECASE,
)


def classify_material_announcement(subject: object, text: object) -> bool:
    content = f"{subject or ''} {text or ''}"
    return bool(MATERIAL_EVENT_PATTERN.search(content))


def add_announcement_features(
    confirmation_rows: pd.DataFrame,
    announcements: pd.DataFrame,
) -> pd.DataFrame:
    """Attach announcements broadcast during trigger-to-confirmation [-3, 0]."""
    rows = confirmation_rows.copy()
    rows["trigger_date"] = pd.to_datetime(rows["trigger_date"]).dt.normalize()
    rows["signal_date"] = pd.to_datetime(rows["signal_date"]).dt.normalize()
    news = announcements.copy()
    news["broadcast_at"] = pd.to_datetime(news["broadcast_at"], errors="coerce")
    news["broadcast_date"] = news["broadcast_at"].dt.normalize()
    news["material_event"] = news["material_event"].astype(bool)

    def aggregate_for_leg(code_column: str, prefix: str) -> pd.DataFrame:
        candidates = rows[["event_id", code_column, "trigger_date", "signal_date"]].rename(
            columns={code_column: "capitaline_code"}
        )
        merged = candidates.merge(news, on="capitaline_code", how="left")
        in_window = merged["broadcast_date"].between(
            merged["trigger_date"], merged["signal_date"], inclusive="both"
        )
        matched = merged.loc[in_window]
        counts = matched.groupby("event_id", as_index=False).agg(
            **{
                f"{prefix}_announcement_count": ("seq_id", "nunique"),
                f"{prefix}_material_announcement_count": ("material_event", "sum"),
            }
        )
        return counts

    result = rows.merge(
        aggregate_for_leg("left_code", "left"), on="event_id", how="left"
    ).merge(
        aggregate_for_leg("right_code", "right"), on="event_id", how="left"
    )
    count_columns = [
        "left_announcement_count",
        "left_material_announcement_count",
        "right_announcement_count",
        "right_material_announcement_count",
    ]
    result[count_columns] = result[count_columns].fillna(0).astype(int)
    result["announcement_count_imbalance"] = (
        result["left_announcement_count"] - result["right_announcement_count"]
    ).abs()
    left_material = result["left_material_announcement_count"].gt(0)
    right_material = result["right_material_announcement_count"].gt(0)
    result["one_sided_material_event"] = left_material ^ right_material
    result["passes_news_veto"] = ~result["one_sided_material_event"]
    return result

