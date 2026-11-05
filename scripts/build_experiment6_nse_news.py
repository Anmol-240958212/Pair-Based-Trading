"""Download and cache point-in-time NSE announcements for Experiment 6."""

from pathlib import Path
import sys
import time
from urllib.parse import quote

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.experiment6 import classify_material_announcement


LANDING = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
ENDPOINT = "https://www.nseindia.com/api/corporate-announcements"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/plain, */*",
}


def new_session() -> requests.Session:
    session = requests.Session()
    response = session.get(LANDING, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return session


def fetch_symbol(session: requests.Session, symbol: str) -> list[dict[str, object]]:
    url = (
        f"{ENDPOINT}?index=equities&from_date=01-01-2018&to_date=30-06-2026"
        f"&symbol={quote(symbol)}"
    )
    for attempt in range(3):
        response = session.get(url, headers={**HEADERS, "Referer": LANDING}, timeout=90)
        if response.status_code == 200:
            return response.json()
        if response.status_code not in {401, 403, 429}:
            response.raise_for_status()
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"NSE announcement request failed for {symbol}: {response.status_code}")


def main() -> None:
    output = ROOT / "data" / "processed"
    panel = pd.read_parquet(output / "experiment5_delay_3_panel.parquet")
    prices = pd.read_parquet(output / "prices.parquet")
    codes = sorted(set(panel["left_code"]).union(panel["right_code"]))
    symbol_map = (
        prices.loc[prices["capitaline_code"].isin(codes) & prices["nse_symbol"].notna()]
        .sort_values("date")
        .groupby("capitaline_code", as_index=False)
        .tail(1)[["capitaline_code", "nse_symbol"]]
    )
    missing = set(codes).difference(symbol_map["capitaline_code"])
    if missing:
        raise ValueError(f"Missing NSE symbols for Capitaline codes: {sorted(missing)}")

    cache_path = output / "experiment6_nse_announcements.parquet"
    existing = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    completed_codes = set(existing["capitaline_code"]) if not existing.empty else set()
    frames = [existing] if not existing.empty else []
    session = new_session()
    remaining = symbol_map.loc[~symbol_map["capitaline_code"].isin(completed_codes)]
    for number, row in enumerate(remaining.itertuples(index=False), start=1):
        try:
            records = fetch_symbol(session, str(row.nse_symbol))
        except RuntimeError:
            session = new_session()
            records = fetch_symbol(session, str(row.nse_symbol))
        if records:
            frame = pd.DataFrame(records)
            frame["capitaline_code"] = int(row.capitaline_code)
            frame["requested_symbol"] = str(row.nse_symbol)
            frames.append(frame)
        print(f"{number}/{len(remaining)} {row.nse_symbol}: {len(records):,}", flush=True)
        time.sleep(0.35)
    announcements = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if announcements.empty:
        raise ValueError("No NSE announcements were downloaded.")
    announcements = announcements.drop_duplicates(
        ["capitaline_code", "seq_id"], keep="last"
    )
    announcements["broadcast_at"] = pd.to_datetime(
        announcements["an_dt"], format="%d-%b-%Y %H:%M:%S", errors="coerce"
    )
    announcements["material_event"] = [
        classify_material_announcement(subject, text)
        for subject, text in zip(announcements.get("desc"), announcements.get("attchmntText"))
    ]
    keep = [
        "capitaline_code", "requested_symbol", "seq_id", "broadcast_at", "desc",
        "attchmntText", "material_event", "attchmntFile", "sm_name", "smIndustry",
    ]
    announcements[keep].sort_values(
        ["capitaline_code", "broadcast_at"]
    ).to_parquet(cache_path, index=False)
    print(f"Saved {len(announcements):,} point-in-time announcements.")


if __name__ == "__main__":
    main()

