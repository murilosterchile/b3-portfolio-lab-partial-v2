from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from portfolio_core.data_quality import DataQualityError, validate_prices


@dataclass(frozen=True)
class UniverseConfig:
    participation_window_sessions: int = 63
    min_session_participation: float = 0.90
    adv_window_sessions: int = 63
    min_adv_brl: float = 1_000_000.0
    min_history_sessions: int = 252
    min_price: float | None = None

    def __post_init__(self) -> None:
        if self.participation_window_sessions <= 0 or self.adv_window_sessions <= 0:
            raise ValueError("universe windows must be positive")
        if not 0 < self.min_session_participation <= 1:
            raise ValueError("min_session_participation must be in (0, 1]")
        if self.min_adv_brl < 0 or self.min_history_sessions < 1:
            raise ValueError("universe liquidity/history thresholds are invalid")


def build_point_in_time_universe(
    prices: pl.DataFrame, *, config: UniverseConfig = UniverseConfig()
) -> pl.DataFrame:
    """Compute investibility only from observations available at each market date."""
    required = {"trade_date", "ticker", "close"}
    if missing := required - set(prices.columns):
        raise DataQualityError(f"universe prices missing columns: {sorted(missing)}")
    if "traded_value_brl" in prices.columns and "volume" in prices.columns:
        traded_value = pl.coalesce("traded_value_brl", "volume")
    elif "traded_value_brl" in prices.columns:
        traded_value = pl.col("traded_value_brl")
    elif "volume" in prices.columns:
        traded_value = pl.col("volume")
    else:
        raise DataQualityError("universe prices missing financial traded value in BRL")
    validate_prices(prices, context="point-in-time universe prices")
    market_dates = sorted(prices.get_column("trade_date").unique().to_list())
    date_index = {value: index for index, value in enumerate(market_dates)}
    rows: list[dict[str, object]] = []
    for ticker_frame in prices.select(
        "trade_date", "ticker", "close", traded_value.alias("traded_value_brl")
    ).sort(
        ["ticker", "trade_date"]
    ).partition_by("ticker", maintain_order=True):
        ticker = str(ticker_frame.get_column("ticker").item(0))
        dates = ticker_frame.get_column("trade_date").to_list()
        indices = np.asarray([date_index[value] for value in dates], dtype=int)
        traded_values = np.asarray(
            ticker_frame.get_column("traded_value_brl").to_list(), dtype=float
        )
        closes = np.asarray(ticker_frame.get_column("close").to_list(), dtype=float)
        for position, (trade_date, market_index) in enumerate(zip(dates, indices, strict=True)):
            participation_start = int(
                np.searchsorted(indices, market_index - config.participation_window_sessions + 1)
            )
            observed_sessions = position - participation_start + 1
            expected_sessions = min(config.participation_window_sessions, market_index + 1)
            participation = observed_sessions / expected_sessions
            adv_start = int(np.searchsorted(indices, market_index - config.adv_window_sessions + 1))
            adv = float(np.median(traded_values[adv_start : position + 1]))
            history_sessions = position + 1
            reasons: list[str] = []
            if history_sessions < config.min_history_sessions:
                reasons.append("insufficient_history")
            if participation < config.min_session_participation:
                reasons.append("low_session_participation")
            if not np.isfinite(adv) or adv < config.min_adv_brl:
                reasons.append("low_adv")
            if config.min_price is not None and closes[position] < config.min_price:
                reasons.append("penny_stock")
            rows.append(
                {
                    "trade_date": trade_date,
                    "ticker": ticker,
                    "quote_observed": True,
                    "history_sessions": history_sessions,
                    "session_participation_63d": participation,
                    "adv_brl_63d": adv,
                    "universe_eligible": not reasons,
                    "universe_rejection_reason": ",".join(reasons) if reasons else None,
                }
            )
    return pl.DataFrame(rows).sort(["trade_date", "ticker"])


def apply_point_in_time_universe(
    signals: pl.DataFrame,
    universe: pl.DataFrame,
    *,
    eligible_only: bool = True,
) -> pl.DataFrame:
    universe_metadata = [
        name
        for name in universe.columns
        if name not in {"trade_date", "ticker"} and name in signals.columns
    ]
    signals = signals.drop(universe_metadata)
    joined = signals.join(universe, on=["trade_date", "ticker"], how="left", validate="1:1")
    joined = joined.with_columns(
        pl.col("quote_observed").fill_null(False),
        pl.col("universe_eligible").fill_null(False),
        pl.col("universe_rejection_reason").fill_null("no_quote_on_signal_date"),
    )
    eligible = joined.filter(pl.col("universe_eligible"))
    # Cross-sectional ranks must describe the investible set, not names that
    # could never have been traded by the strategy on that date.
    for rank_column in [name for name in eligible.columns if name.startswith("rank_")]:
        source = rank_column.removeprefix("rank_")
        if source in eligible.columns:
            eligible = eligible.with_columns(
                (
                    pl.col(source).rank(method="average").over("trade_date")
                    / (pl.col(source).is_not_null() & pl.col(source).is_finite())
                    .sum()
                    .over("trade_date")
                ).alias(rank_column)
            )
    eligible = add_investible_targets(eligible)
    if eligible_only:
        return eligible

    target_columns = [
        "market_forward_return",
        "target_excess_return",
        "target_cross_sectional_rank",
    ]
    joined = joined.drop([name for name in target_columns if name in joined.columns])
    return joined.join(
        eligible.select("trade_date", "ticker", *target_columns),
        on=["trade_date", "ticker"],
        how="left",
        validate="1:1",
    )


def add_investible_targets(features: pl.DataFrame) -> pl.DataFrame:
    """Normalize raw forward returns inside the eligible PIT cross-section."""
    if "future_return" not in features.columns:
        raise DataQualityError("future_return is required before target normalization")
    derived = {
        "market_forward_return",
        "target_excess_return",
        "target_cross_sectional_rank",
    }
    frame = features.drop([name for name in derived if name in features.columns])
    return frame.with_columns(
        pl.col("future_return").mean().over("trade_date").alias("market_forward_return")
    ).with_columns(
        (pl.col("future_return") - pl.col("market_forward_return")).alias(
            "target_excess_return"
        ),
        (
            pl.col("future_return").rank(method="average").over("trade_date")
            / pl.col("future_return").is_not_null().sum().over("trade_date")
            - 0.5
        ).alias("target_cross_sectional_rank"),
    )
