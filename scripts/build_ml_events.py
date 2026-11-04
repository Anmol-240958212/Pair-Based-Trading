"""Build the completed-window ML event dataset from the source price history."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.data import load_price_data, load_static_data
from pairs_trading.walkforward import generate_walkforward_events, semiannual_windows


def main() -> None:
    prices = load_price_data(ROOT, range(2015, 2027))
    company_master, _ = load_static_data(ROOT)
    windows = semiannual_windows("2018-01-01", "2026-06-30", formation_years=3)
    events, screens, summary = generate_walkforward_events(
        prices,
        company_master,
        windows,
        require_fdr=False,
        require_bidirectional=False,
        require_johansen_confirmation=False,
    )

    output = ROOT / "data" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    events.to_parquet(output / "ml_events.parquet", index=False)
    screens.to_parquet(output / "pair_screens.parquet", index=False)
    summary.to_csv(output / "walkforward_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved {len(events):,} labelled events from {len(windows)} completed windows.")


if __name__ == "__main__":
    main()
