from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl

from portfolio_core.data.universe import apply_point_in_time_universe, build_point_in_time_universe
from portfolio_core.features import build_technical_features, monthly_snapshots
from portfolio_core.ml.protocol import ResearchProtocol


def main() -> None:
    protocol = ResearchProtocol()
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    parts = sorted((data_dir / "silver" / "b3_prices_adjusted").glob("year=*/part-000.parquet"))
    if not parts:
        raise SystemExit("Adjusted B3 price history is required")
    prices = pl.concat([pl.read_parquet(path) for path in parts], how="vertical_relaxed")
    output = data_dir / "gold" / "features" / "horizon_challengers"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    universe = build_point_in_time_universe(prices)
    for horizon in (21, 42, 63):
        raw_labels = build_technical_features(prices, horizon_days=horizon)
        panel = apply_point_in_time_universe(monthly_snapshots(raw_labels), universe).filter(
            (pl.col("trade_date") <= pl.lit(protocol.development_knowledge_cutoff))
            & (pl.col("target_end_date") <= pl.lit(protocol.development_knowledge_cutoff))
        )
        path = output / f"monthly_features_{horizon}d.parquet"
        panel.write_parquet(path, compression="zstd")
        rows.append({"horizon_bars": horizon, "rows": panel.height, "path": str(path)})
    metadata = {
        "registered_candidates": [21, 42, 63],
        "primary_baseline": 21,
        "rebalance_interval": "monthly",
        "selection_knowledge_cutoff": str(protocol.development_knowledge_cutoff),
        "diagnostic_year_excluded": protocol.diagnostic_year,
        "challengers": rows,
    }
    (output / "protocol.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"challengers={output}")


if __name__ == "__main__":
    main()
