from __future__ import annotations

import argparse
import os
from pathlib import Path

from portfolio_core.data.b3 import download_cotahist, materialize_cotahist
from portfolio_core.data.universe import apply_point_in_time_universe, build_point_in_time_universe
from portfolio_core.features import build_technical_features, monthly_snapshots
import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and materialize official B3 COTAHIST data")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--no-download", action="store_true", help="Use an already downloaded ZIP in data/raw/b3/YEAR")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_dir = Path(os.getenv("DATA_DIR", "data"))
    base_url = os.getenv("B3_COTAHIST_BASE_URL", "https://bvmf.bmfbovespa.com.br/InstDados/SerHist")
    expected = data_dir / "raw" / "b3" / str(args.year) / f"COTAHIST_A{args.year}.ZIP"
    if args.no_download:
        if not expected.exists():
            raise SystemExit(f"Expected manual B3 ZIP at {expected}")
        zip_path = expected
    else:
        zip_path, artifact = download_cotahist(
            year=args.year, base_url=base_url, data_dir=data_dir, force=args.force
        )
        print(f"downloaded sha256={artifact.sha256}")

    parquet = materialize_cotahist(zip_path, year=args.year, data_dir=data_dir, equities_only=True)
    print(f"prices={parquet}")

    # Rebuild the cross-year feature panel if enough historical partitions exist.
    parts = sorted((data_dir / "silver" / "b3_prices").glob("year=*/part-000.parquet"))
    if len(parts) >= 2:
        prices = pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed")
        features = build_technical_features(prices)
        monthly = apply_point_in_time_universe(
            monthly_snapshots(features), build_point_in_time_universe(prices)
        )
        target = data_dir / "gold" / "features" / "monthly_features.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        monthly.write_parquet(target, compression="zstd")
        print(f"features={target} rows={monthly.height}")
    else:
        print("Need at least two yearly partitions before building long-horizon features.")


if __name__ == "__main__":
    main()
