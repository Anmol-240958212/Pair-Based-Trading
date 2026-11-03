"""Research utilities for the Indian-equity pairs-trading project."""

from .data import load_price_data, load_static_data
from .backtest import backtest_pair
from .statistics import analyze_pair
from .universe import eligible_companies, same_industry_pairs

__all__ = [
    "analyze_pair",
    "backtest_pair",
    "eligible_companies",
    "load_price_data",
    "load_static_data",
    "same_industry_pairs",
]

