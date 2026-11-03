"""Loading and light standardisation for the Capitaline source workbooks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PRICE_COLUMNS = {
    "Capitaline Code": "capitaline_code",
    "Company Name": "company_name",
    "NSE Symbol": "nse_symbol",
    "Date": "date",
    "Open Price": "open",
    "Close Price": "close",
    "Total Volume (Adjusted)": "volume_adjusted",
    "Net Turnover - Rs. Thousand": "turnover_rs_thousand",
}

STATIC_COLUMNS = {
    "Capitaline Code": "capitaline_code",
    "Company Name": "company_name",
    "Company Long Name": "company_long_name",
    "NSE Symbol": "nse_symbol",
    "ISIN No": "isin",
    "Sector": "sector",
    "Industry": "industry",
    "Listing Status(NSE)": "nse_listing_status",
    "Delisting Date": "delisting_date",
    "List/First Avl Price Date(NSE)": "first_price_date_nse",
    "Market Capitalisation (NSE)": "market_cap_nse",
}


def _read_capitaline_excel(path: Path, columns: dict[str, str]) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="Screener", header=1, engine="openpyxl")
    available = [column for column in columns if column in frame.columns]
    return frame.loc[:, available].rename(columns=columns)


def load_price_data(data_dir: str | Path, years: Iterable[int]) -> pd.DataFrame:
    """Load requested annual files without modifying the source workbooks."""
    root = Path(data_dir)
    requested_years = sorted(set(int(year) for year in years))
    processed_path = root / "data" / "processed" / "prices.parquet"
    if processed_path.exists():
        prices = pd.read_parquet(
            processed_path,
            filters=[("source_year", "in", requested_years)],
        )
        return prices.sort_values(["capitaline_code", "date"]).reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for year in requested_years:
        path = root / f"data_{year}.xlsx"
        if not path.exists():
            raise FileNotFoundError(f"Missing annual price file: {path.name}")
        frame = _read_capitaline_excel(path, PRICE_COLUMNS)
        frame["source_year"] = year
        frames.append(frame)

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    for column in ["capitaline_code", "open", "close", "volume_adjusted", "turnover_rs_thousand"]:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices = prices.dropna(subset=["capitaline_code", "date", "close"])
    prices = prices.loc[prices["close"] > 0].copy()
    prices["capitaline_code"] = prices["capitaline_code"].astype("int64")
    return prices.sort_values(["capitaline_code", "date"]).reset_index(drop=True)


def load_static_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a one-row company master and a duplicate/ambiguity audit table.

    The original static records are not overwritten. When multiple records exist
    for one Capitaline code, the master uses the first non-null value and the
    audit table records whether sector or industry classifications conflict.
    """
    path = Path(data_dir) / "STATIC_DATA_SHARANYA.xlsx"
    static = _read_capitaline_excel(path, STATIC_COLUMNS)
    static["capitaline_code"] = pd.to_numeric(static["capitaline_code"], errors="coerce")
    static = static.dropna(subset=["capitaline_code"]).copy()
    static["capitaline_code"] = static["capitaline_code"].astype("int64")
    for column in ["delisting_date", "first_price_date_nse"]:
        if column in static:
            static[column] = pd.to_datetime(static[column], errors="coerce").dt.normalize()

    audit = (
        static.groupby("capitaline_code", as_index=False)
        .agg(
            static_record_count=("capitaline_code", "size"),
            sector_values=("sector", lambda values: values.dropna().nunique()),
            industry_values=("industry", lambda values: values.dropna().nunique()),
        )
    )
    audit["classification_conflict"] = (audit["sector_values"] > 1) | (audit["industry_values"] > 1)

    master = static.sort_values("capitaline_code").groupby("capitaline_code", as_index=False).first()
    master = master.merge(audit[["capitaline_code", "static_record_count", "classification_conflict"]],
                          on="capitaline_code", how="left")
    return master, audit


def price_data_audit(prices: pd.DataFrame) -> dict[str, int | float | str]:
    """Calculate a compact set of non-destructive source-data checks."""
    duplicate_rows = int(prices.duplicated(["capitaline_code", "date"], keep=False).sum())
    return {
        "rows": int(len(prices)),
        "companies": int(prices["capitaline_code"].nunique()),
        "first_date": str(prices["date"].min().date()),
        "last_date": str(prices["date"].max().date()),
        "duplicate_company_dates": duplicate_rows,
        "missing_close": int(prices["close"].isna().sum()),
    }

