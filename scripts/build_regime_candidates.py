from __future__ import annotations

import os
from pathlib import Path

import polars as pl
from portfolio_core.features.regime import (
    attach_regime_candidates,
    attach_selic_candidate,
    build_market_regime_candidates,
)
from portfolio_core.features.technical import build_technical_features


def main() -> None:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    parts = sorted((data_dir / "silver" / "b3_prices_adjusted").glob("year=*/part-000.parquet"))
    panel_path = data_dir / "gold" / "features" / "monthly_features_with_fundamentals.parquet"
    if not panel_path.exists():
        panel_path = data_dir / "gold" / "features" / "monthly_features.parquet"
    selic_path = data_dir / "silver" / "bcb" / "series=selic_target" / "part-000.parquet"
    if not parts or not panel_path.exists() or not selic_path.exists():
        raise SystemExit("Adjusted prices, feature panel and SELIC SGS series are required")
    prices = pl.concat([pl.read_parquet(path) for path in parts], how="vertical_relaxed")
    daily = build_technical_features(prices)
    regime = attach_selic_candidate(
        build_market_regime_candidates(daily), pl.read_parquet(selic_path)
    )
    panel = attach_regime_candidates(pl.read_parquet(panel_path), regime)
    out = data_dir / "gold" / "features" / "monthly_features_regime_candidates.parquet"
    panel.write_parquet(out, compression="zstd")
    print(f"candidate_panel={out}")
    print("These features are development-only candidates; they are not added to DEFAULT_FEATURES.")


if __name__ == "__main__":
    main()
