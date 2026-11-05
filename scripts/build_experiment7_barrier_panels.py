"""Build alternative payoff-label panels from three-day confirmation signals."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairs_trading.experiment7 import BARRIER_CONFIGURATIONS, relabel_barriers


def main() -> None:
    output = ROOT / "data" / "processed"
    rows = pd.read_parquet(output / "experiment5_delay_3_panel.parquet")
    prices = pd.read_parquet(output / "prices.parquet")
    summary = []
    for name, (target, stop) in BARRIER_CONFIGURATIONS.items():
        relabelled = relabel_barriers(rows, prices, target, stop)
        relabelled.to_parquet(output / f"experiment7_{name}_panel.parquet", index=False)
        summary.append(
            {
                "configuration": name,
                "take_profit_return": target,
                "stop_loss_return": stop,
                "rows": len(relabelled),
                "positive_targets": int(relabelled["target"].sum()),
                "positive_target_rate": float(relabelled["target"].mean()),
            }
        )
        print(f"{name}: {len(relabelled):,} rows, target rate {relabelled.target.mean():.3f}")
    pd.DataFrame(summary).to_csv(output / "experiment7_panel_summary.csv", index=False)


if __name__ == "__main__":
    main()

