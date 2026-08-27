from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep the default useful on an interactive notebook. Environment overrides always win.
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

from portfolio_core.data.corporate_actions import (
    load_failed_corporate_action_roots,
    materialize_adjusted_price_history,
    sync_corporate_actions,
)
from portfolio_core.data.universe import apply_point_in_time_universe, build_point_in_time_universe
from portfolio_core.features import build_technical_features, monthly_snapshots



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch official B3 corporate actions and build adjusted analytical prices"
    )
    parser.add_argument("--force", action="store_true", help="Refresh cached raw B3 JSON payloads")
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Reuse existing actions and failure manifest without making HTTP requests",
    )
    parser.add_argument(
        "--strict-corporate-actions",
        action="store_true",
        help="Abort if any issuer root cannot be refreshed instead of quarantining that root",
    )
    parser.add_argument(
        "--no-rebuild-features",
        action="store_true",
        help="Only sync/adjust prices; do not rebuild monthly technical features",
    )
    args = parser.parse_args()
    if args.force and args.skip_sync:
        raise SystemExit("--force and --skip-sync cannot be used together")

    data_dir = Path(os.getenv("DATA_DIR", "data"))
    parts = sorted((data_dir / "silver" / "b3_prices").glob("year=*/part-000.parquet"))
    if not parts:
        raise SystemExit("Need B3 COTAHIST partitions before corporate-action ingestion")
    # Only ticker is needed to discover roots, so scan lazily to keep memory usage low.
    roots = (
        pl.concat([pl.scan_parquet(p).select("ticker") for p in parts])
        .select(pl.col("ticker").str.slice(0, 4).alias("root"))
        .drop_nulls()
        .unique()
        .sort("root")
        .collect()["root"]
        .to_list()
    )
    if args.skip_sync:
        failed_root_names = load_failed_corporate_action_roots(data_dir)
        print(f"corporate_action_sync=skipped failed_roots={len(failed_root_names)}")
    else:
        summary = sync_corporate_actions(roots, data_dir=data_dir, force=args.force)
        print(summary)
        failed_root_names = summary.failed_root_names
    if failed_root_names and args.strict_corporate_actions:
        raise SystemExit(
            f"Corporate-action sync incomplete: {len(failed_root_names)} issuer roots failed: "
            f"{failed_root_names[:20]}"
        )
    if failed_root_names:
        print(
            f"coverage_quarantine_roots={len(failed_root_names)} "
            f"examples={failed_root_names[:20]}"
        )
    adjusted = materialize_adjusted_price_history(
        data_dir=data_dir, excluded_issuer_roots=failed_root_names
    )
    print(f"adjusted_prices={adjusted}")

    if not args.no_rebuild_features:
        adjusted_parts = sorted(
            (data_dir / "silver" / "b3_prices_adjusted").glob("year=*/part-000.parquet")
        )
        prices = pl.concat(
            [pl.read_parquet(path) for path in adjusted_parts], how="vertical_relaxed"
        )
        features = build_technical_features(prices)
        monthly = apply_point_in_time_universe(
            monthly_snapshots(features), build_point_in_time_universe(prices)
        )
        target = data_dir / "gold" / "features" / "monthly_features.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        monthly.write_parquet(target, compression="zstd")
        print(f"features={target} rows={monthly.height} source=adjusted_close")


if __name__ == "__main__":
    main()
