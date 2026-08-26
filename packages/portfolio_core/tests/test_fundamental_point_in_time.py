from datetime import date

import polars as pl
import pytest
from portfolio_core.data.issuer_bridge import validate_issuer_bridge
from portfolio_core.data_quality import DataQualityError
from portfolio_core.features.fundamental import (
    attach_fundamentals_point_in_time,
    build_company_fundamental_snapshots,
)


def _bridge() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["TEST3"],
            "CD_CVM": ["123"],
            "valid_from": [date(2020, 1, 1)],
            "valid_to": [date(2999, 12, 31)],
            "sector": ["Test"],
        }
    )


def test_statement_received_later_is_not_available_early() -> None:
    technical = pl.DataFrame(
        {
            "ticker": ["TEST3", "TEST3"],
            "trade_date": [date(2024, 5, 1), date(2024, 6, 1)],
        }
    )
    fundamentals = pl.DataFrame(
        {
            "CD_CVM": ["123"],
            "DT_REFER": [date(2024, 3, 31)],
            "DT_RECEB": [date(2024, 5, 10)],
            "net_margin": [0.2],
        }
    )

    joined = attach_fundamentals_point_in_time(technical, fundamentals, _bridge())

    assert joined.filter(pl.col("trade_date") == date(2024, 5, 1))["net_margin"].item() is None
    assert joined.filter(pl.col("trade_date") == date(2024, 6, 1))["net_margin"].item() == 0.2


def test_unmapped_ticker_keeps_technical_features_without_fundamentals() -> None:
    technical = pl.DataFrame(
        {
            "ticker": ["TEST3", "OLD3"],
            "trade_date": [date(2024, 6, 1), date(2024, 6, 1)],
            "return_21d": [0.1, 0.2],
        }
    )
    fundamentals = pl.DataFrame(
        {
            "CD_CVM": ["123"],
            "DT_REFER": [date(2024, 3, 31)],
            "DT_RECEB": [date(2024, 5, 10)],
            "net_margin": [0.2],
        }
    )
    joined = attach_fundamentals_point_in_time(technical, fundamentals, _bridge())
    old = joined.filter(pl.col("ticker") == "OLD3")
    assert old["return_21d"].item() == 0.2
    assert old["net_margin"].item() is None


def test_same_receipt_date_selects_latest_reference_deterministically() -> None:
    statements = pl.DataFrame(
        {
            "CD_CVM": [123, 123],
            "DT_REFER": [date(2023, 12, 31), date(2024, 3, 31)],
            "DT_RECEB": [date(2024, 5, 10), date(2024, 5, 10)],
            "CD_CONTA": ["1", "1"],
            "VL_CONTA": [100.0, 120.0],
        }
    )

    snapshots = build_company_fundamental_snapshots(statements)

    assert snapshots.height == 1
    assert snapshots["DT_REFER"].item() == date(2024, 3, 31)
    assert snapshots["assets"].item() == 120.0


def test_zero_account_denominator_produces_missing_ratio_not_extreme_value() -> None:
    statements = pl.DataFrame(
        {
            "CD_CVM": [123, 123],
            "DT_REFER": [date(2024, 3, 31)] * 2,
            "DT_RECEB": [date(2024, 5, 10)] * 2,
            "CD_CONTA": ["3.01", "3.11"],
            "VL_CONTA": [0.0, 50.0],
        }
    )

    snapshot = build_company_fundamental_snapshots(statements)

    assert snapshot["net_margin"].item() is None


def test_negative_equity_is_flagged_and_never_made_positive() -> None:
    statements = pl.DataFrame(
        {
            "CD_CVM": [123, 123, 123],
            "DT_REFER": [date(2024, 3, 31)] * 3,
            "DT_RECEB": [date(2024, 5, 10)] * 3,
            "CD_CONTA": ["1", "2.03", "3.11"],
            "VL_CONTA": [100.0, -20.0, 5.0],
        }
    )
    snapshot = build_company_fundamental_snapshots(statements)
    assert snapshot["negative_equity"].item()
    assert snapshot["roe_proxy"].item() is None
    assert snapshot["log_equity"].item() is None


def test_cumulative_flows_are_converted_to_point_in_time_ttm() -> None:
    period_ends = [
        date(2024, 3, 31),
        date(2024, 6, 30),
        date(2024, 9, 30),
        date(2024, 12, 31),
    ]
    receipts = [date(2024, 5, 10), date(2024, 8, 10), date(2024, 11, 10), date(2025, 3, 10)]
    statements = pl.DataFrame(
        {
            "CD_CVM": [123] * 8,
            "DT_REFER": [value for value in period_ends for _ in range(2)],
            "DT_RECEB": [value for value in receipts for _ in range(2)],
            "DT_INI_EXERC": [date(2024, 1, 1)] * 8,
            "DT_FIM_EXERC": [value for value in period_ends for _ in range(2)],
            "CD_CONTA": [value for _ in period_ends for value in ("1", "3.01")],
            "VL_CONTA": [value for cumulative in (100.0, 220.0, 360.0, 500.0) for value in (1_000.0, cumulative)],
        }
    )
    snapshots = build_company_fundamental_snapshots(statements)
    latest = snapshots.sort("DT_RECEB").tail(1)
    assert latest["revenue"].item() == 500.0
    assert latest["revenue_basis"].item() == "TTM"


def test_same_receipt_and_reference_selects_highest_cvm_version() -> None:
    statements = pl.DataFrame(
        {
            "CD_CVM": [123, 123],
            "DT_REFER": [date(2024, 3, 31)] * 2,
            "DT_RECEB": [date(2024, 5, 10)] * 2,
            "VERSAO": [1, 2],
            "CD_CONTA": ["1", "1"],
            "VL_CONTA": [100.0, 150.0],
        }
    )

    snapshot = build_company_fundamental_snapshots(statements)

    assert snapshot.height == 1
    assert snapshot["assets"].item() == 150.0


def test_overlapping_bridge_mappings_fail() -> None:
    bridge = pl.DataFrame(
        {
            "ticker": ["TEST3", "TEST3"],
            "CD_CVM": ["123", "456"],
            "valid_from": [date(2020, 1, 1), date(2022, 1, 1)],
            "valid_to": [date(2023, 1, 1), date(2024, 1, 1)],
            "sector": ["Test", "Test"],
        }
    )

    with pytest.raises(DataQualityError, match="multiple CD_CVM"):
        validate_issuer_bridge(bridge)
