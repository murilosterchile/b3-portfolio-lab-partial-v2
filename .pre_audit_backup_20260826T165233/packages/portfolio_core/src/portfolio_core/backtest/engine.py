from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from portfolio_core.data_quality import (
    DataQualityError,
    validate_finite_array,
    validate_prices,
)


@dataclass(frozen=True)
class BacktestConfig:
    top_k: int = 10
    transaction_cost_bps: float = 15.0
    initial_capital: float = 100_000.0
    max_abs_unadjusted_return: float = 1.0
    max_abs_adjusted_return: float = 5.0
    # Rank-buffer/hysteresis: a current holding can remain while it is inside top_k + buffer.
    # This is a deterministic turnover-control rule based only on the current cross-section.
    selection_buffer: int = 0


@dataclass(frozen=True)
class BacktestSummary:
    start_value: float
    end_value: float
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    turnover: float
    annualized_turnover: float
    rebalance_periods: int
    average_positions: float
    corporate_action_transitions_excluded: int


_DEFAULT_CONFIG = BacktestConfig()


def _one_way_turnover(
    current: np.ndarray,
    target: np.ndarray,
    *,
    current_cash: float = 0.0,
    target_cash: float = 0.0,
) -> float:
    """Half-L1 turnover, including cash so initial funding has turnover 1.0."""
    return float(
        0.5
        * (
            np.abs(np.asarray(target) - np.asarray(current)).sum()
            + abs(target_cash - current_cash)
        )
    )


def _metrics(
    equity: np.ndarray,
    daily_returns: np.ndarray,
    turnover: float,
    *,
    rebalance_periods: int = 0,
    average_positions: float = 0.0,
    corporate_action_transitions_excluded: int = 0,
) -> BacktestSummary:
    if len(equity) < 2:
        raise ValueError("Backtest has insufficient observations")
    validate_finite_array(equity, context="equity curve")
    validate_finite_array(daily_returns, context="net portfolio returns")
    if np.any(equity <= 0):
        raise DataQualityError("equity curve contains non-positive values")
    total = float(equity[-1] / equity[0] - 1.0)
    years = max((len(daily_returns) / 252.0), 1.0 / 252.0)
    cagr = float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)
    vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0
    annualized_mean = float(np.mean(daily_returns) * 252) if len(daily_returns) else 0.0
    # The prototype has no governed risk-free series yet; zero is explicit and consistent.
    sharpe = annualized_mean / vol if vol > 1e-12 else 0.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    return BacktestSummary(
        start_value=float(equity[0]),
        end_value=float(equity[-1]),
        total_return=total,
        cagr=cagr,
        annualized_volatility=vol,
        sharpe=sharpe,
        max_drawdown=float(np.min(drawdowns)),
        turnover=float(turnover),
        annualized_turnover=float(turnover / years),
        rebalance_periods=rebalance_periods,
        average_positions=average_positions,
        corporate_action_transitions_excluded=corporate_action_transitions_excluded,
    )


def _buffered_selection(
    ranked_tickers: list[str],
    *,
    previous: set[str],
    top_k: int,
    buffer: int,
) -> list[str]:
    """Keep existing names within the rank buffer, then fill by current rank.

    This avoids selling/rebuying around a hard top-K boundary. The rule is deterministic and uses
    only the current ranking, so it does not introduce look-ahead.
    """
    if top_k <= 0:
        return []
    eligible_existing = {
        ticker for ticker in ranked_tickers[: top_k + max(buffer, 0)] if ticker in previous
    }
    selected = [ticker for ticker in ranked_tickers if ticker in eligible_existing]
    for ticker in ranked_tickers:
        if len(selected) >= top_k:
            break
        if ticker not in eligible_existing and ticker not in selected:
            selected.append(ticker)
    return selected[:top_k]


def run_monthly_topk_backtest(
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    *,
    score_column: str,
    config: BacktestConfig = _DEFAULT_CONFIG,
) -> tuple[pl.DataFrame, BacktestSummary]:
    """Auditable monthly top-K simulation using only point-in-time signals.

    When ``adjusted_close`` exists it is used for performance accounting and corporate-action
    transitions are no longer neutralized. Raw COTAHIST remains a diagnostic fallback: its
    ``distribution_number`` transitions are excluded because those price jumps are not economic
    returns. A rank buffer can reduce turnover without peeking into future observations.
    """
    if config.top_k <= 0:
        raise ValueError("top_k must be positive")
    if config.selection_buffer < 0:
        raise ValueError("selection_buffer must be non-negative")
    if config.max_abs_unadjusted_return <= 0:
        raise ValueError("max_abs_unadjusted_return must be positive")
    if config.max_abs_adjusted_return <= 0:
        raise ValueError("max_abs_adjusted_return must be positive")
    required_signals = {"trade_date", "ticker", score_column}
    if missing := required_signals - set(signals.columns):
        raise DataQualityError(f"signals missing required columns: {sorted(missing)}")
    invalid_signals = signals.filter(
        pl.col(score_column).is_null()
        | pl.col(score_column).is_nan()
        | pl.col(score_column).is_infinite()
    )
    if not invalid_signals.is_empty():
        raise DataQualityError(
            f"signals contain non-finite {score_column}: "
            f"{invalid_signals.select('ticker', 'trade_date', score_column).head(10).to_dicts()}"
        )

    signal_rows = (
        signals.select("trade_date", "ticker", score_column)
        .with_columns(pl.col("trade_date").dt.truncate("1mo").alias("signal_month"))
        .sort(["signal_month", "ticker", "trade_date"])
        .group_by(["signal_month", "ticker"], maintain_order=True)
        .tail(1)
        .sort(["signal_month", "trade_date", "ticker"])
    )
    if signal_rows.is_empty():
        raise DataQualityError("backtest has no signals")

    signal_tickers = signal_rows["ticker"].unique().to_list()
    used_prices = prices.filter(pl.col("ticker").is_in(signal_tickers))
    missing_prices = sorted(set(signal_tickers) - set(used_prices["ticker"].unique().to_list()))
    if missing_prices:
        raise DataQualityError(f"signals have tickers without prices: {missing_prices[:20]}")
    validate_prices(used_prices, context="backtest prices")

    price_column = "adjusted_close" if "adjusted_close" in used_prices.columns else "close"
    if price_column == "adjusted_close":
        invalid = used_prices.filter(
            pl.col(price_column).is_null()
            | ~pl.col(price_column).is_finite()
            | (pl.col(price_column) <= 0)
        )
        if not invalid.is_empty():
            raise DataQualityError(
                "backtest adjusted prices contain invalid values: "
                f"{invalid.select('ticker', 'trade_date', price_column).head(10).to_dicts()}"
            )

    selected_price_columns = ["trade_date", "ticker", price_column]
    has_distribution_number = "distribution_number" in used_prices.columns
    if has_distribution_number:
        selected_price_columns.append("distribution_number")
    price_long = used_prices.select(selected_price_columns).sort(["trade_date", "ticker"])
    close_frame = (
        price_long.select("trade_date", "ticker", price_column)
        .pivot(index="trade_date", on="ticker", values=price_column)
        .sort("trade_date")
    )
    dates = close_frame["trade_date"].to_list()
    tickers = [column for column in close_frame.columns if column != "trade_date"]
    observed = close_frame.select(pl.col(tickers).is_not_null()).to_numpy()
    close = close_frame.select(tickers).fill_null(strategy="forward").to_numpy()
    if has_distribution_number:
        distribution = (
            price_long.select("trade_date", "ticker", "distribution_number")
            .pivot(index="trade_date", on="ticker", values="distribution_number")
            .sort("trade_date")
            .select(tickers)
            .fill_null(strategy="forward")
            .to_numpy()
        )
    else:
        distribution = np.full_like(close, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        daily_ret = close[1:] / close[:-1] - 1.0
    max_abs_return = (
        config.max_abs_adjusted_return
        if price_column == "adjusted_close"
        else config.max_abs_unadjusted_return
    )
    valid_distribution = np.isfinite(distribution[1:]) & np.isfinite(distribution[:-1])
    corporate_action = valid_distribution & (distribution[1:] != distribution[:-1])

    ticker_to_col = {ticker: index for index, ticker in enumerate(tickers)}
    date_to_index = {value: index for index, value in enumerate(dates)}
    weights = np.zeros(len(tickers))
    cash_weight = 1.0
    rebalance_lookup: dict[object, np.ndarray] = {}
    position_counts: list[int] = []
    previous_selection: set[str] = set()
    for month in sorted(set(signal_rows["signal_month"].to_list())):
        cross = signal_rows.filter(pl.col("signal_month") == month)
        rebalance_date = cross["trade_date"].max()
        if rebalance_date not in date_to_index:
            raise DataQualityError(f"signal date {rebalance_date} has no market price date")
        ranked_tickers = [
            ticker
            for ticker in cross.sort(score_column, descending=True)["ticker"].to_list()
            if ticker in ticker_to_col
        ]
        selected = _buffered_selection(
            ranked_tickers,
            previous=previous_selection,
            top_k=config.top_k,
            buffer=config.selection_buffer,
        )
        previous_selection = set(selected)
        target = np.zeros(len(tickers))
        if selected:
            for ticker in selected:
                target[ticker_to_col[ticker]] = 1.0 / len(selected)
        rebalance_lookup[rebalance_date] = target
        position_counts.append(len(selected))

    first_rebalance = min(rebalance_lookup)
    start_index = date_to_index[first_rebalance]
    capital = config.initial_capital
    equity = [capital]
    portfolio_daily: list[float] = []
    total_turnover = 0.0
    excluded_transitions: set[tuple[str, object, object]] = set()

    for index in range(start_index + 1, len(dates)):
        previous_date = dates[index - 1]
        current_date = dates[index]
        previous_capital = capital
        if previous_date in rebalance_lookup:
            target = rebalance_lookup[previous_date]
            target_cash = max(0.0, 1.0 - float(target.sum()))
            turnover = _one_way_turnover(
                weights,
                target,
                current_cash=cash_weight,
                target_cash=target_cash,
            )
            cost_fraction = turnover * config.transaction_cost_bps / 10_000.0
            if cost_fraction >= 1.0:
                raise DataQualityError(f"transaction costs consume all capital on {previous_date}")
            total_turnover += turnover
            capital *= 1.0 - cost_fraction
            weights = target
            cash_weight = target_cash

        active = weights > 0
        asset_returns = daily_ret[index - 1].copy()
        # Adjusted prices should retain ordinary market returns across an already-modeled action.
        # A still-extreme jump at a COTAHIST distribution boundary indicates an action missing
        # from the adjustment source, so use the same conservative fallback as raw prices.
        unadjusted_action = corporate_action[index - 1] & (
            (price_column == "close")
            | (np.abs(asset_returns) > max_abs_return)
        )
        active_actions = active & unadjusted_action
        for column in np.flatnonzero(active_actions):
            excluded_transitions.add((tickers[column], previous_date, current_date))
        asset_returns[active_actions] = 0.0

        invalid_active = active & ~np.isfinite(asset_returns)
        if np.any(invalid_active):
            details = [
                {
                    "ticker": tickers[column],
                    "previous_date": previous_date,
                    "date": current_date,
                    "previous_close": close[index - 1, column],
                    "close": close[index, column],
                }
                for column in np.flatnonzero(invalid_active)[:10]
            ]
            raise DataQualityError(f"held assets have non-finite returns: {details}")
        # The limit is a daily-return guard. After forward-filling an illiquid asset, the next
        # observed price contains the return accumulated since its last trade, not a daily jump.
        consecutive_quotes = observed[index - 1] & observed[index]
        suspicious = (
            active
            & consecutive_quotes
            & ~active_actions
            & (np.abs(asset_returns) > max_abs_return)
        )
        if np.any(suspicious):
            details = [
                {
                    "ticker": tickers[column],
                    "previous_date": previous_date,
                    "date": current_date,
                    "previous_close": float(close[index - 1, column]),
                    "close": float(close[index, column]),
                    "return": float(asset_returns[column]),
                    "price_basis": price_column,
                }
                for column in np.flatnonzero(suspicious)[:10]
            ]
            basis = "adjusted" if price_column == "adjusted_close" else "unadjusted"
            raise DataQualityError(
                f"held assets have suspicious {basis} returns requiring investigation: {details}"
            )

        day_return = float(weights[active] @ asset_returns[active]) if np.any(active) else 0.0
        if not np.isfinite(day_return) or day_return <= -1.0:
            raise DataQualityError(
                f"invalid portfolio return {day_return} for {previous_date} -> {current_date}"
            )
        capital *= 1.0 + day_return
        net_day_return = float(capital / previous_capital - 1.0)
        portfolio_daily.append(net_day_return)
        equity.append(capital)

        growth = 1.0 + day_return
        drifted = np.zeros_like(weights)
        drifted[active] = weights[active] * (1.0 + asset_returns[active]) / growth
        weights = drifted
        cash_weight = cash_weight / growth

    equity_array = np.asarray(equity)
    daily_array = np.asarray(portfolio_daily)
    summary = _metrics(
        equity_array,
        daily_array,
        total_turnover,
        rebalance_periods=len(rebalance_lookup),
        average_positions=float(np.mean(position_counts)) if position_counts else 0.0,
        corporate_action_transitions_excluded=len(excluded_transitions),
    )
    curve = pl.DataFrame(
        {"trade_date": dates[start_index:], "portfolio_value": equity_array}
    ).with_columns(pl.lit(price_column).alias("price_basis"))
    return curve, summary
