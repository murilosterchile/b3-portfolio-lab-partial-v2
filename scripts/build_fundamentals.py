from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl

from portfolio_core.data.issuer_bridge import load_issuer_bridge, validate_issuer_bridge
from portfolio_core.features.fundamental import (
    attach_fundamentals_point_in_time,
    build_company_fundamental_snapshots,
)


def _load_bridge(data_dir: Path, explicit: str | None) -> pl.DataFrame:
    if explicit:
        return load_issuer_bridge(Path(explicit))
    registry = data_dir / "silver" / "cvm_registry" / "issuer_bridge.parquet"
    if not registry.exists():
        raise SystemExit(
            "No issuer bridge. Run ingest_cvm_registry.py or pass --bridge with a governed CSV."
        )
    bridge = pl.read_parquet(registry).select(
        "ticker", "CD_CVM", "valid_from", "valid_to", "sector"
    ).with_columns(
        pl.col("ticker").cast(pl.Utf8).str.to_uppercase(),
        pl.col("CD_CVM").cast(pl.Utf8),
        pl.col("sector").fill_null("Unknown"),
    )
    validate_issuer_bridge(bridge)
    return bridge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bridge",
        required=False,
        default="",
        help="Optional governed ticker/CD_CVM bridge CSV. Defaults to official CVM FCA/CAD bridge.",
    )
    args = parser.parse_args()
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    monthly_path = data_dir / "gold" / "features" / "monthly_features.parquet"
    cvm_parts = sorted(
        (data_dir / "silver" / "cvm").glob("document=*/statement=*/year=*/part-000.parquet")
    )
    missing = []
    if not monthly_path.exists():
        missing.append("monthly technical features (run make ingest-corporate-actions)")
    if not cvm_parts:
        missing.append("CVM statement partitions (run make ingest-cvm for DFP/ITR years)")
    if missing:
        raise SystemExit("Missing: " + "; ".join(missing))
    bridge = _load_bridge(data_dir, args.bridge or None)
    statements = pl.concat([pl.read_parquet(p) for p in cvm_parts], how="diagonal_relaxed")
    fundamentals = build_company_fundamental_snapshots(statements)
    monthly = pl.read_parquet(monthly_path)
    combined = attach_fundamentals_point_in_time(monthly, fundamentals, bridge)
    out = data_dir / "gold" / "features" / "monthly_features_with_fundamentals.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(out, compression="zstd")
    print(
        f"wrote={out} rows={combined.height} bridge_rows={bridge.height} "
        f"bridge_source={'explicit_csv' if args.bridge else 'official_cvm_fca_cad'}"
    )


if __name__ == "__main__":
    main()
