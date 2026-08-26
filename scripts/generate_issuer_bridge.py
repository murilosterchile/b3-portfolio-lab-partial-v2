from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl
from portfolio_core.data.issuer_bridge import REQUIRED_BRIDGE_COLUMNS, suggest_issuer_bridge


def _load_existing(path: Path) -> pl.DataFrame:
    schema = {
        "ticker": pl.Utf8,
        "CD_CVM": pl.Utf8,
        "valid_from": pl.Utf8,
        "valid_to": pl.Utf8,
        "sector": pl.Utf8,
    }
    if not path.exists():
        return pl.DataFrame(schema=schema)
    frame = pl.read_csv(path, schema_overrides=schema)
    if missing := REQUIRED_BRIDGE_COLUMNS - set(frame.columns):
        raise ValueError(f"Existing issuer bridge missing columns: {sorted(missing)}")
    frame = frame.select(list(schema)).with_columns(
        pl.col(column).fill_null("").str.strip_chars() for column in schema
    ).with_columns(pl.col("ticker").str.to_uppercase())
    invalid = frame.filter(
        ~pl.col("ticker").str.contains(r"^[A-Z0-9]{4,12}$")
        | ~pl.col("CD_CVM").str.contains(r"^[0-9]+$")
        | pl.col("valid_from").str.strptime(pl.Date, "%Y-%m-%d", strict=False).is_null()
        | (
            (pl.col("valid_to") != "")
            & pl.col("valid_to").str.strptime(pl.Date, "%Y-%m-%d", strict=False).is_null()
        )
        | (pl.col("sector") == "")
    )
    if not invalid.is_empty():
        raise ValueError(
            f"Existing issuer bridge has invalid rows for tickers: "
            f"{invalid.get_column('ticker').to_list()}"
        )
    return frame


def _write_csv_atomic(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.part")
    frame.write_csv(partial)
    partial.replace(path)


def _official_registry_rows(registry_path: Path, b3_tickers: set[str]) -> pl.DataFrame:
    if not registry_path.exists():
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "CD_CVM": pl.Utf8,
                "valid_from": pl.Utf8,
                "valid_to": pl.Utf8,
                "sector": pl.Utf8,
            }
        )
    registry = pl.read_parquet(registry_path).filter(pl.col("ticker").is_in(sorted(b3_tickers)))
    if registry.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "CD_CVM": pl.Utf8,
                "valid_from": pl.Utf8,
                "valid_to": pl.Utf8,
                "sector": pl.Utf8,
            }
        )
    return registry.select(
        pl.col("ticker").cast(pl.Utf8),
        pl.col("CD_CVM").cast(pl.Utf8),
        pl.col("valid_from").dt.strftime("%Y-%m-%d"),
        pl.when(pl.col("valid_to") >= pl.date(2999, 1, 1))
        .then(pl.lit(""))
        .otherwise(pl.col("valid_to").dt.strftime("%Y-%m-%d"))
        .alias("valid_to"),
        pl.col("sector").fill_null("Unknown").replace("", "Unknown"),
    ).unique().sort(["ticker", "valid_from"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ticker/CD_CVM mappings, preferring official CVM FCA/CAD history"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--review-output", required=True)
    parser.add_argument(
        "--registry",
        default="",
        help="Optional issuer_bridge.parquet from ingest_cvm_registry.py; auto-detected by default",
    )
    args = parser.parse_args()

    data_dir = Path(os.getenv("DATA_DIR", "data"))
    b3_parts = sorted((data_dir / "silver" / "b3_prices").glob("year=*/part-000.parquet"))
    cvm_parts = sorted(
        (data_dir / "silver" / "cvm").glob("document=*/statement=*/year=*/part-000.parquet")
    )
    if not b3_parts or not cvm_parts:
        raise SystemExit("Need B3 price and CVM statement partitions before generating the bridge")

    b3 = (
        pl.concat(
            [
                pl.scan_parquet(part).select("ticker", "issuer_short_name", "trade_date")
                for part in b3_parts
            ]
        )
        .drop_nulls(["ticker", "issuer_short_name", "trade_date"])
        .group_by("ticker", "issuer_short_name")
        .agg(
            pl.col("trade_date").min().alias("first_trade_date"),
            pl.col("trade_date").max().alias("last_trade_date"),
            pl.len().alias("observations"),
        )
        .sort(["ticker", "last_trade_date", "observations"], descending=[False, True, True])
        .unique(subset=["ticker"], keep="first", maintain_order=True)
        .collect()
    )
    cvm = (
        pl.concat([pl.scan_parquet(part).select("CD_CVM", "DENOM_CIA") for part in cvm_parts])
        .drop_nulls(["CD_CVM", "DENOM_CIA"])
        .with_columns(pl.col("CD_CVM").cast(pl.Utf8))
        .unique()
        .collect()
    )

    output = Path(args.output)
    review_output = Path(args.review_output)
    existing = _load_existing(output)
    b3_tickers = set(b3["ticker"].to_list())
    registry_path = (
        Path(args.registry)
        if args.registry
        else data_dir / "silver" / "cvm_registry" / "issuer_bridge.parquet"
    )
    official = _official_registry_rows(registry_path, b3_tickers)

    # Existing reviewed rows have priority. Official identifier-based FCA/CAD mappings fill the
    # remainder. Only tickers absent from both paths reach the conservative name-matching fallback.
    existing_tickers = set(existing["ticker"].to_list()) if not existing.is_empty() else set()
    official = official.filter(~pl.col("ticker").is_in(sorted(existing_tickers)))
    covered = existing_tickers | set(official["ticker"].to_list())
    pending = b3.filter(~pl.col("ticker").is_in(sorted(covered)))
    generated, review = suggest_issuer_bridge(pending, cvm)
    combined = pl.concat([existing, official, generated], how="vertical_relaxed").unique().sort(
        ["ticker", "valid_from"]
    )

    if not official.is_empty():
        official_review = official.select(
            "ticker",
            pl.lit("").alias("issuer_short_name"),
            pl.col("CD_CVM").alias("candidate_CD_CVM"),
            pl.lit("").alias("candidate_DENOM_CIA"),
            pl.lit("official_fca_cad").alias("match_type"),
            pl.lit(1.0).alias("similarity"),
            pl.lit("auto_added_official").alias("status"),
        )
        review = pl.concat([official_review, review], how="vertical_relaxed")

    _write_csv_atomic(combined, output)
    _write_csv_atomic(review, review_output)
    official_added = official["ticker"].n_unique() if not official.is_empty() else 0
    name_added = review.filter(pl.col("status") == "auto_added").height if not review.is_empty() else 0
    unresolved = (
        review.filter(~pl.col("status").is_in(["auto_added", "auto_added_official"]))
        .get_column("ticker")
        .n_unique()
        if not review.is_empty()
        else 0
    )
    print(
        f"bridge={output} rows={combined.height} official_added={official_added} "
        f"name_added={name_added}"
    )
    print(f"review={review_output} unresolved_tickers={unresolved}")


if __name__ == "__main__":
    main()
