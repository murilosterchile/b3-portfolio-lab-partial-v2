from __future__ import annotations

import argparse
import os
import time
from datetime import date
from pathlib import Path

_PROFILE_DEFAULTS = {
    "balanced": ("6", "4"),
    "low-impact": ("3", "2"),
    "fast": ("10", "6"),
}
_profile = os.getenv("PIPELINE_PROFILE", "balanced")
if _profile not in _PROFILE_DEFAULTS:
    raise SystemExit(f"Invalid PIPELINE_PROFILE={_profile!r}; use balanced, low-impact or fast")
_http_workers, _cpu_threads = _PROFILE_DEFAULTS[_profile]
if not os.getenv("B3_HTTP_CONCURRENCY"):
    os.environ["B3_HTTP_CONCURRENCY"] = _http_workers
if not os.getenv("PIPELINE_CPU_THREADS"):
    os.environ["PIPELINE_CPU_THREADS"] = _cpu_threads
if not os.getenv("POLARS_MAX_THREADS"):
    os.environ["POLARS_MAX_THREADS"] = os.environ["PIPELINE_CPU_THREADS"]

import polars as pl

from portfolio_core.data.b3 import download_cotahist, materialize_cotahist
from portfolio_core.data.bcb import materialize_default_macro
from portfolio_core.data.corporate_actions import (
    load_failed_corporate_action_roots,
    materialize_adjusted_price_history,
    sync_corporate_actions,
)
from portfolio_core.data.cvm import download_cvm_document, extract_cvm_statements
from portfolio_core.data.cvm_registry import materialize_cvm_registry
from portfolio_core.data.issuer_bridge import validate_issuer_bridge
from portfolio_core.data.universe import apply_point_in_time_universe, build_point_in_time_universe
from portfolio_core.features import build_technical_features, monthly_snapshots
from portfolio_core.features.fundamental import (
    attach_fundamentals_point_in_time,
    build_company_fundamental_snapshots,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end official-data refresh for research/backtesting"
    )
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-action-sync",
        action="store_true",
        help="Reuse existing B3 actions and failure manifest without HTTP requests",
    )
    parser.add_argument(
        "--strict-corporate-actions",
        action="store_true",
        help="Abort if any B3 supplemental issuer root fails instead of quarantining it",
    )
    args = parser.parse_args()
    if args.force and args.skip_action_sync:
        raise SystemExit("--force and --skip-action-sync cannot be used together")
    if args.start_year > args.end_year:
        raise SystemExit("start-year must be <= end-year")

    data_dir = Path(os.getenv("DATA_DIR", "data"))
    b3_base = os.getenv("B3_COTAHIST_BASE_URL", "https://bvmf.bmfbovespa.com.br/InstDados/SerHist")
    cvm_base = os.getenv("CVM_BASE_URL", "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC")
    bcb_base = os.getenv("BCB_SGS_BASE_URL", "https://api.bcb.gov.br/dados/serie/bcdata.sgs")

    print("[1/7] B3 COTAHIST")
    for year in range(args.start_year, args.end_year + 1):
        archive, artifact = download_cotahist(
            year=year,
            base_url=b3_base,
            data_dir=data_dir,
            force=args.force,
        )
        path = materialize_cotahist(archive, year=year, data_dir=data_dir, equities_only=True)
        print(f"- {year}: {path} sha256={artifact.sha256[:12]}")

    print("[2/7] B3 corporate actions + total-return analytical prices")
    print(
        f"- profile={_profile} http_concurrency={os.environ['B3_HTTP_CONCURRENCY']} "
        f"cpu_threads={os.environ['PIPELINE_CPU_THREADS']}"
    )
    raw_parts = sorted((data_dir / "silver" / "b3_prices").glob("year=*/part-000.parquet"))
    roots = (
        pl.concat([pl.scan_parquet(path).select("ticker") for path in raw_parts])
        .select(pl.col("ticker").str.slice(0, 4).alias("root"))
        .drop_nulls()
        .unique()
        .sort("root")
        .collect()["root"]
        .to_list()
    )
    if args.skip_action_sync:
        failed_root_names = load_failed_corporate_action_roots(data_dir)
        print(f"- corporate-action HTTP sync skipped; failed_roots={len(failed_root_names)}")
    else:
        action_summary = sync_corporate_actions(roots, data_dir=data_dir, force=args.force)
        print(action_summary)
        failed_root_names = action_summary.failed_root_names
    if failed_root_names and args.strict_corporate_actions:
        raise SystemExit(
            f"Corporate-action refresh is incomplete ({len(failed_root_names)} failures): "
            f"{failed_root_names[:20]}"
        )
    if failed_root_names:
        print(
            f"- quarantining {len(failed_root_names)} issuer roots with unavailable "
            "corporate-action coverage"
        )
    adjustment_started = time.monotonic()
    adjustment = materialize_adjusted_price_history(
        data_dir=data_dir, excluded_issuer_roots=failed_root_names
    )
    adjustment["build_seconds"] = time.monotonic() - adjustment_started
    print(adjustment)

    print("[3/7] Technical feature panel from adjusted prices")
    adjusted_parts = sorted(
        (data_dir / "silver" / "b3_prices_adjusted").glob("year=*/part-000.parquet")
    )
    adjusted_prices = pl.concat(
        [pl.read_parquet(path) for path in adjusted_parts], how="vertical_relaxed"
    )
    raw_labels = build_technical_features(adjusted_prices)
    monthly = monthly_snapshots(raw_labels)
    universe = build_point_in_time_universe(adjusted_prices)
    technical = apply_point_in_time_universe(monthly, universe)
    technical_path = data_dir / "gold" / "features" / "monthly_features.parquet"
    technical_path.parent.mkdir(parents=True, exist_ok=True)
    technical.write_parquet(technical_path, compression="zstd")
    print(f"- {technical_path} rows={technical.height}")

    print("[4/7] CVM historical registry (CAD + FCA)")
    registry = materialize_cvm_registry(
        data_dir=data_dir,
        years=range(max(2010, args.start_year - 1), args.end_year + 1),
        force=args.force,
    )
    print(registry)

    print("[5/7] CVM ITR/DFP statements")
    for year in range(args.start_year, args.end_year + 1):
        documents = ["DFP"] + (["ITR"] if year >= 2011 else [])
        for document in documents:
            archive = download_cvm_document(
                document=document,
                year=year,
                base_url=cvm_base,
                data_dir=data_dir,
            )
            outputs = extract_cvm_statements(
                archive,
                data_dir=data_dir,
                document=document,
                year=year,
            )
            print(f"- {year} {document}: {len(outputs)} statement partitions")

    print("[6/7] Point-in-time fundamentals")
    bridge_path = data_dir / "silver" / "cvm_registry" / "issuer_bridge.parquet"
    bridge = pl.read_parquet(bridge_path).select(
        "ticker", "CD_CVM", "valid_from", "valid_to", "sector"
    ).with_columns(pl.col("CD_CVM").cast(pl.Utf8), pl.col("sector").fill_null("Unknown"))
    validate_issuer_bridge(bridge)
    cvm_parts = sorted(
        (data_dir / "silver" / "cvm").glob("document=*/statement=*/year=*/part-000.parquet")
    )
    statements = pl.concat([pl.read_parquet(path) for path in cvm_parts], how="diagonal_relaxed")
    fundamentals = build_company_fundamental_snapshots(statements)
    combined = attach_fundamentals_point_in_time(technical, fundamentals, bridge)
    fundamental_path = data_dir / "gold" / "features" / "monthly_features_with_fundamentals.parquet"
    combined.write_parquet(fundamental_path, compression="zstd")
    print(f"- {fundamental_path} rows={combined.height}")

    print("[7/7] BCB macro")
    macro = materialize_default_macro(
        start=date(args.start_year, 1, 1),
        end=date.today(),
        base_url=bcb_base,
        data_dir=data_dir,
    )
    for path in macro:
        print(f"- {path}")

    print("DATA REFRESH COMPLETE")
    print("Next: make backtest")


if __name__ == "__main__":
    main()
