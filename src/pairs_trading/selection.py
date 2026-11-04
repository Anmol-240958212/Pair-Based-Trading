"""Paper-backed alternative pair-selection specifications."""

from __future__ import annotations

import pandas as pd


def apply_pair_specification(
    events: pd.DataFrame,
    specification: str,
    p_value_maximum: float = 0.05,
) -> pd.DataFrame:
    """Filter a superset of forward-selected events into a robustness sample."""
    supported = {
        "forward",
        "bidirectional",
        "bidirectional_fdr",
        "bidirectional_johansen",
        "bidirectional_fdr_johansen",
    }
    if specification not in supported:
        raise ValueError(f"Unknown specification: {specification}")
    selected = events.loc[
        events["cointegration_p_value"].le(p_value_maximum)
    ].copy()
    if specification != "forward":
        selected = selected.loc[
            selected["cointegration_p_value_reverse"].le(p_value_maximum)
        ]
    if "fdr" in specification:
        selected = selected.loc[selected["passes_fdr"].astype(bool)]
    if "johansen" in specification:
        selected = selected.loc[selected["johansen_passes_5pct"].astype(bool)]
    return selected.reset_index(drop=True)
