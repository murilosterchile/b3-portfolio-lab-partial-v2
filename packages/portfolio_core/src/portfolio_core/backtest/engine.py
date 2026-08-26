from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl
from portfolio_core.data_quality import DataQualityError, validate_finite_array, validate_prices


@dataclass(frozen=True)
class BacktestConfig:
    top_k: int = 10
    transaction_cost_bps: float = 15.0
    initial_capital: float = 100_000.0
    max_abs_unadjusted_return: float = 1.0
    max_abs_adjusted_return: float = 5.0
    selection_buffer: int = 0
    # Signals are formed after the signal-date close. The earliest executable
    # price is therefore the next observed market close.
    execution_delay_bars: int = 1
    max_stale_valuation_sessions: int = 20


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
    execution_delay_bars: int = 1


@dataclass(frozen=True)
class DetailedBacktest:
    curve: pl.DataFrame
    summary: BacktestSummary
    selections: pl.DataFrame
    contributions: pl.DataFrame
    rejected_orders: pl.DataFrame


_DEFAULT_CONFIG = BacktestConfig()


def _one_way_turnover(
    current: np.ndarray,
    target: np.ndarray,
    *,
    current_cash: float = 0.0,
    target_cash: float = 0.0,
) -> float:
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
    execution_delay_bars: int = 1,
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
        execution_delay_bars=execution_delay_bars,
    )


def _buffered_selection(
    ranked_tickers: list[str],
    *,
    previous: set[str],
    top_k: int,
    buffer: int,
) -> list[str]:
    if top_k <= 0:
        return []
    eligible_existing = {
        ticker for ticker in ranked_tickers[: top_k + max(buffer, 0)] if ticker in previous
    }
    selected = [ticker for ticker in ranked_tickers if ticker in eligible_existing]
    for ticker in ranked_tickers:
        if len(selected) >= top_k:
            break
        if ticker not in selected:
            selected.append(ticker)
    return selected[:top_k]


def _execution_date(
    dates: list[date], date_to_index: dict[date, int], signal_date: date, delay_bars: int
) -> date | None:
    index = date_to_index.get(signal_date)
    if index is None:
        return None
    execution_index = index + delay_bars
    return dates[execution_index] if execution_index < len(dates) else None


def run_monthly_topk_backtest_detailed(
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    *,
    score_column: str,
    config: BacktestConfig = _DEFAULT_CONFIG,
) -> DetailedBacktest:
    """Monthly top-K simulation with explicit next-close execution.

    Signal-date data are assumed to become available after that close. Portfolio
    weights held during signal_close -> next_close therefore remain unchanged;
    the new target is installed only at the execution close. This removes the
    same-close leakage present in the previous engine.
    """
    if config.top_k <= 0:
        raise ValueError("top_k must be positive")
    if config.selection_buffer < 0:
        raise ValueError("selection_buffer must be non-negative")
    if config.execution_delay_bars < 1:
        raise ValueError("execution_delay_bars must be >= 1")
    if config.transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative")
    if config.max_stale_valuation_sessions < 1:
        raise ValueError("max_stale_valuation_sessions must be positive")
    required_signals = {"trade_date", "ticker", score_column}
    if missing := required_signals - set(signals.columns):
        raise DataQualityError(f"signals missing required columns: {sorted(missing)}")
    invalid_signals = signals.filter(
        pl.col(score_column).is_null()
        | pl.col(score_column).is_nan()
        | pl.col(score_column).is_infinite()
    )
    if not invalid_signals.is_empty():
        raise DataQualityError(f"signals contain non-finite {score_column}")

    # A cross-section must refer to one common signal date. monthly_snapshots()
    # enforces this upstream; the assertion catches legacy/mixed-date panels.
    signal_columns = ["trade_date", "ticker", score_column]
    issuer_column = next(
        (name for name in ("issuer_id", "CD_CVM", "issuer_identifier") if name in signals.columns),
        None,
    )
    if issuer_column is not None:
        signal_columns.append(issuer_column)
    signal_rows = signals.select(signal_columns).with_columns(
        pl.col("trade_date").dt.truncate("1mo").alias("signal_month")
    )
    mixed = signal_rows.group_by("signal_month").agg(
        pl.col("trade_date").n_unique().alias("n_dates")
    ).filter(pl.col("n_dates") != 1)
    if not mixed.is_empty():
        raise DataQualityError(
            "monthly signal cross-sections mix different trade dates; rebuild monthly features "
            "with a common market month-end"
        )
    signal_rows = signal_rows.sort(["signal_month", "trade_date", "ticker"])
    if signal_rows.is_empty():
        raise DataQualityError("backtest has no signals")

    signal_tickers = signal_rows["ticker"].unique().to_list()
    used_prices = prices.filter(pl.col("ticker").is_in(signal_tickers))
    if used_prices.is_empty():
        raise DataQualityError("signals have no matching price history")
    validate_prices(used_prices, context="backtest prices")

    price_column = "adjusted_close" if "adjusted_close" in used_prices.columns else "close"
    if price_column == "adjusted_close":
        invalid = used_prices.filter(
            pl.col(price_column).is_null() | ~pl.col(price_column).is_finite() | (pl.col(price_column) <= 0)
        )
        if not invalid.is_empty():
            raise DataQualityError("backtest adjusted prices contain invalid values")

    selected_price_columns = ["trade_date", "ticker", price_column]
    has_distribution_number = "distribution_number" in used_prices.columns
    if has_distribution_number:
        selected_price_columns.append("distribution_number")
    price_long = used_prices.select(selected_price_columns).sort(["trade_date", "ticker"])
    close_frame = (
        price_long.select("trade_date", "ticker", price_column)
        .pivot(index="trade_date", on="ticker", values=price_column)
    )
    close_frame = prices.select("trade_date").unique().join(
        close_frame, on="trade_date", how="left", validate="1:1"
    ).sort("trade_date")
    dates = close_frame["trade_date"].to_list()
    tickers = [column for column in close_frame.columns if column != "trade_date"]
    observed = close_frame.select(pl.col(tickers).is_not_null()).to_numpy()
    close = close_frame.select(tickers).fill_null(strategy="forward").to_numpy()
    if has_distribution_number:
        distribution_frame = (
            price_long.select("trade_date", "ticker", "distribution_number")
            .pivot(index="trade_date", on="ticker", values="distribution_number")
        )
        distribution = (
            prices.select("trade_date").unique()
            .join(distribution_frame, on="trade_date", how="left", validate="1:1")
            .sort("trade_date")
            .select(tickers)
            .fill_null(strategy="forward")
            .to_numpy()
        )
    else:
        distribution = np.full_like(close, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        daily_ret = close[1:] / close[:-1] - 1.0
    max_abs_return = config.max_abs_adjusted_return if price_column == "adjusted_close" else config.max_abs_unadjusted_return
    valid_distribution = np.isfinite(distribution[1:]) & np.isfinite(distribution[:-1])
    corporate_action = valid_distribution & (distribution[1:] != distribution[:-1])

    ticker_to_col = {ticker: index for index, ticker in enumerate(tickers)}
    date_to_index = {value: index for index, value in enumerate(dates)}
    rebalance_lookup: dict[date, np.ndarray] = {}
    rebalance_signal_dates: dict[date, date] = {}
    selection_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    position_counts: list[int] = []
    previous_selection: set[str] = set()

    for month in sorted(set(signal_rows["signal_month"].to_list())):
        cross = signal_rows.filter(pl.col("signal_month") == month)
        signal_date = cross["trade_date"].item(0)
        execution_date = _execution_date(dates, date_to_index, signal_date, config.execution_delay_bars)
        if execution_date is None:
            continue
        execution_index = date_to_index[execution_date]
        ranked_tickers: list[str] = []
        seen_issuers: set[str] = set()
        for row in cross.sort(score_column, descending=True).iter_rows(named=True):
            if len(ranked_tickers) >= config.top_k + config.selection_buffer:
                break
            ticker = str(row["ticker"])
            if ticker not in ticker_to_col:
                rejected_rows.append(
                    {"signal_date": signal_date, "execution_date": execution_date, "ticker": ticker, "reason": "missing_price_history"}
                )
                continue
            if not bool(observed[execution_index, ticker_to_col[ticker]]):
                rejected_rows.append(
                    {"signal_date": signal_date, "execution_date": execution_date, "ticker": ticker, "reason": "no_observed_execution_quote"}
                )
                continue
            issuer = str(row[issuer_column]) if issuer_column and row[issuer_column] is not None else ticker[:4]
            if issuer in seen_issuers:
                rejected_rows.append(
                    {"signal_date": signal_date, "execution_date": execution_date, "ticker": ticker, "reason": "issuer_share_class_cap"}
                )
                continue
            seen_issuers.add(issuer)
            ranked_tickers.append(ticker)
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
                selection_rows.append(
                    {
                        "signal_date": signal_date,
                        "execution_date": execution_date,
                        "ticker": ticker,
                        "target_weight": 1.0 / len(selected),
                    }
                )
        rebalance_lookup[execution_date] = target
        rebalance_signal_dates[execution_date] = signal_date
        position_counts.append(len(selected))

    if not rebalance_lookup:
        raise DataQualityError("no signal has a future executable market date")
    first_signal_date = min(row["signal_date"] for row in selection_rows) if selection_rows else min(rebalance_lookup)
    start_index = date_to_index[first_signal_date]
    capital = config.initial_capital
    equity = [capital]
    portfolio_daily: list[float] = []
    total_turnover = 0.0
    excluded_transitions: set[tuple[str, object, object]] = set()
    weights = np.zeros(len(tickers))
    cash_weight = 1.0
    contribution_rows: list[dict[str, object]] = []
    stale_age = np.zeros(len(tickers), dtype=int)

    for index in range(start_index + 1, len(dates)):
        previous_date = dates[index - 1]
        current_date = dates[index]
        previous_capital = capital
        active = weights > 0
        asset_returns = daily_ret[index - 1].copy()
        stale_age = np.where(observed[index], 0, stale_age + 1)
        conservatively_delisted = active & (stale_age == config.max_stale_valuation_sessions + 1)
        asset_returns[conservatively_delisted] = -0.999
        unadjusted_action = corporate_action[index - 1] & (
            (price_column == "close") | (np.abs(asset_returns) > max_abs_return)
        )
        active_actions = active & unadjusted_action
        for column in np.flatnonzero(active_actions):
            excluded_transitions.add((tickers[column], previous_date, current_date))
        asset_returns[active_actions] = 0.0
        invalid_active = active & ~np.isfinite(asset_returns)
        if np.any(invalid_active):
            raise DataQualityError("held assets have non-finite returns")
        consecutive_quotes = observed[index - 1] & observed[index]
        suspicious = active & consecutive_quotes & ~active_actions & (np.abs(asset_returns) > max_abs_return)
        if np.any(suspicious):
            basis = "adjusted" if price_column == "adjusted_close" else "unadjusted"
            raise DataQualityError(f"held assets have suspicious {basis} returns requiring investigation")

        if np.any(active):
            for column in np.flatnonzero(active):
                contribution_rows.append(
                    {
                        "trade_date": current_date,
                        "ticker": tickers[column],
                        "weight_at_previous_close": float(weights[column]),
                        "asset_return": float(asset_returns[column]),
                        "return_contribution": float(weights[column] * asset_returns[column]),
                        "stale_age_sessions": int(stale_age[column]),
                    }
                )
        day_return = float(weights[active] @ asset_returns[active]) if np.any(active) else 0.0
        if not np.isfinite(day_return) or day_return <= -1.0:
            raise DataQualityError(f"invalid portfolio return {day_return} for {previous_date} -> {current_date}")
        capital *= 1.0 + day_return
        growth = 1.0 + day_return
        drifted = np.zeros_like(weights)
        drifted[active] = weights[active] * (1.0 + asset_returns[active]) / growth
        weights = drifted
        cash_weight = cash_weight / growth

        # Execute only after reaching the execution close. The interval from the
        # signal close to this close was earned by the old portfolio/cash.
        if current_date in rebalance_lookup:
            target = rebalance_lookup[current_date].copy()
            blocked = active & ~observed[index]
            if np.any(blocked):
                target[blocked] = weights[blocked]
                residual = max(0.0, 1.0 - float(target[blocked].sum()))
                tradable_target = ~blocked
                requested = float(target[tradable_target].sum())
                if requested > residual and requested > 0:
                    target[tradable_target] *= residual / requested
                for column in np.flatnonzero(blocked):
                    rejected_rows.append(
                        {
                            "signal_date": rebalance_signal_dates[current_date],
                            "execution_date": current_date,
                            "ticker": tickers[column],
                            "reason": "no_observed_quote_for_exit",
                        }
                    )
            target_cash = max(0.0, 1.0 - float(target.sum()))
            turnover = _one_way_turnover(weights, target, current_cash=cash_weight, target_cash=target_cash)
            cost_fraction = turnover * config.transaction_cost_bps / 10_000.0
            if cost_fraction >= 1.0:
                raise DataQualityError(f"transaction costs consume all capital on {current_date}")
            total_turnover += turnover
            capital *= 1.0 - cost_fraction
            weights = target
            cash_weight = target_cash

        net_day_return = float(capital / previous_capital - 1.0)
        portfolio_daily.append(net_day_return)
        equity.append(capital)

    equity_array = np.asarray(equity)
    daily_array = np.asarray(portfolio_daily)
    summary = _metrics(
        equity_array,
        daily_array,
        total_turnover,
        rebalance_periods=len(rebalance_lookup),
        average_positions=float(np.mean(position_counts)) if position_counts else 0.0,
        corporate_action_transitions_excluded=len(excluded_transitions),
        execution_delay_bars=config.execution_delay_bars,
    )
    curve = pl.DataFrame({"trade_date": dates[start_index:], "portfolio_value": equity_array}).with_columns(
        pl.lit(price_column).alias("price_basis")
    )
    selections = pl.DataFrame(selection_rows) if selection_rows else pl.DataFrame(
        schema={"signal_date": pl.Date, "execution_date": pl.Date, "ticker": pl.Utf8, "target_weight": pl.Float64}
    )
    contributions = pl.DataFrame(contribution_rows) if contribution_rows else pl.DataFrame(
        schema={"trade_date": pl.Date, "ticker": pl.Utf8, "weight_at_previous_close": pl.Float64, "asset_return": pl.Float64, "return_contribution": pl.Float64, "stale_age_sessions": pl.Int64}
    )
    rejected = pl.DataFrame(rejected_rows) if rejected_rows else pl.DataFrame(
        schema={"signal_date": pl.Date, "execution_date": pl.Date, "ticker": pl.Utf8, "reason": pl.Utf8}
    )
    return DetailedBacktest(curve, summary, selections, contributions, rejected)


def run_monthly_topk_backtest(
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    *,
    score_column: str,
    config: BacktestConfig = _DEFAULT_CONFIG,
) -> tuple[pl.DataFrame, BacktestSummary]:
    result = run_monthly_topk_backtest_detailed(
        prices, signals, score_column=score_column, config=config
    )
    return result.curve, result.summary
