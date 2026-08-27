from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import polars as pl
from portfolio_core.data_quality import DataQualityError, validate_finite_array, validate_prices


@dataclass(frozen=True)
class ExecutionCostModel:
    """One-way execution-cost assumptions, all stated independently.

    ``fee_bps`` is charged on executed one-way notional. A bid/ask observation
    is a ``quoted_full_spread_bps``; crossing once costs half of it. The ADV
    fallback tiers are already one-way half-spreads (not full spreads and not
    all-in costs). ``impact_bps`` is the coefficient of ``sqrt(Q / ADV)`` in
    the registered model and excludes both fees and spread.
    """

    mode: Literal["fixed_bps", "liquidity"] = "fixed_bps"
    fee_bps: float = 0.0
    participation_cap: float = 0.05
    impact_bps: float = 10.0
    impact_model: Literal["sqrt_participation_bps", "volatility_scaled"] = (
        "sqrt_participation_bps"
    )
    volatility_impact_y: float = 0.5
    adv_tier_limits: tuple[float, float] = (1_000_000.0, 10_000_000.0)
    adv_tier_one_way_half_spread_bps: tuple[float, float, float] = (30.0, 15.0, 8.0)
    use_eod_quoted_spread: bool = False


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
    cost_model: ExecutionCostModel = ExecutionCostModel()


@dataclass(frozen=True)
class BacktestSummary:
    start_value: float
    end_value: float
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe: float
    raw_sharpe: float
    excess_cagr: float
    risk_free_cagr: float
    max_drawdown: float
    turnover: float
    annualized_turnover: float
    rebalance_periods: int
    average_positions: float
    corporate_action_transitions_excluded: int
    total_cost: float = 0.0
    fees_cost: float = 0.0
    spread_cost: float = 0.0
    impact_cost: float = 0.0
    execution_delay_bars: int = 1
    gross_end_value: float = 0.0
    gross_total_return: float = 0.0
    gross_cagr: float = 0.0
    gross_annualized_volatility: float = 0.0
    gross_sharpe: float = 0.0
    gross_raw_sharpe: float = 0.0
    relative_cagr: float = 0.0
    requested_notional: float = 0.0
    filled_notional: float = 0.0
    rejected_notional: float = 0.0
    fill_ratio: float = 1.0


@dataclass(frozen=True)
class DetailedBacktest:
    curve: pl.DataFrame
    summary: BacktestSummary
    selections: pl.DataFrame
    contributions: pl.DataFrame
    rejected_orders: pl.DataFrame
    costs: pl.DataFrame


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
    risk_free_returns: np.ndarray | None = None,
    cost_totals: dict[str, float] | None = None,
    gross_equity: np.ndarray | None = None,
    execution_totals: dict[str, float] | None = None,
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
    if risk_free_returns is None:
        risk_free = np.zeros_like(daily_returns)
    else:
        risk_free = np.asarray(risk_free_returns, dtype=float)
        if risk_free.shape != daily_returns.shape:
            raise ValueError("risk_free_returns must align with daily_returns")
        validate_finite_array(risk_free, context="risk-free returns")
    excess_returns = daily_returns - risk_free
    vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0
    excess_vol = float(np.std(excess_returns, ddof=1) * np.sqrt(252)) if len(excess_returns) > 1 else 0.0
    annualized_mean = float(np.mean(daily_returns) * 252) if len(daily_returns) else 0.0
    raw_sharpe = annualized_mean / vol if vol > 1e-12 else 0.0
    annualized_excess_mean = float(np.mean(excess_returns) * 252) if len(excess_returns) else 0.0
    sharpe = annualized_excess_mean / excess_vol if excess_vol > 1e-12 else 0.0
    risk_free_wealth = float(np.prod(1.0 + risk_free))
    risk_free_cagr = risk_free_wealth ** (1.0 / years) - 1.0
    relative_wealth = (equity[-1] / equity[0]) / risk_free_wealth
    relative_cagr = float(relative_wealth ** (1.0 / years) - 1.0)
    costs = cost_totals or {}
    gross = equity if gross_equity is None else np.asarray(gross_equity, dtype=float)
    validate_finite_array(gross, context="gross equity curve")
    gross_total_return = float(gross[-1] / gross[0] - 1.0)
    gross_cagr = float((gross[-1] / gross[0]) ** (1.0 / years) - 1.0)
    gross_returns = gross[1:] / gross[:-1] - 1.0
    gross_vol = (
        float(np.std(gross_returns, ddof=1) * np.sqrt(252))
        if len(gross_returns) > 1
        else 0.0
    )
    gross_raw_sharpe = (
        float(np.mean(gross_returns) * 252) / gross_vol if gross_vol > 1e-12 else 0.0
    )
    gross_excess = gross_returns - risk_free
    gross_excess_vol = (
        float(np.std(gross_excess, ddof=1) * np.sqrt(252))
        if len(gross_excess) > 1
        else 0.0
    )
    gross_sharpe = (
        float(np.mean(gross_excess) * 252) / gross_excess_vol
        if gross_excess_vol > 1e-12
        else 0.0
    )
    execution = execution_totals or {}
    requested_notional = float(execution.get("requested_notional", 0.0))
    filled_notional = float(execution.get("filled_notional", requested_notional))
    rejected_notional = max(0.0, requested_notional - filled_notional)
    fill_ratio = filled_notional / requested_notional if requested_notional > 0 else 1.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    return BacktestSummary(
        start_value=float(equity[0]),
        end_value=float(equity[-1]),
        total_return=total,
        cagr=cagr,
        annualized_volatility=vol,
        sharpe=sharpe,
        raw_sharpe=raw_sharpe,
        excess_cagr=relative_cagr,
        risk_free_cagr=risk_free_cagr,
        max_drawdown=float(np.min(drawdowns)),
        turnover=float(turnover),
        annualized_turnover=float(turnover / years),
        rebalance_periods=rebalance_periods,
        average_positions=average_positions,
        corporate_action_transitions_excluded=corporate_action_transitions_excluded,
        total_cost=float(sum(costs.values())),
        fees_cost=float(costs.get("fees", 0.0)),
        spread_cost=float(costs.get("spread", 0.0)),
        impact_cost=float(costs.get("impact", 0.0)),
        execution_delay_bars=execution_delay_bars,
        gross_end_value=float(gross[-1]),
        gross_total_return=gross_total_return,
        gross_cagr=gross_cagr,
        gross_annualized_volatility=gross_vol,
        gross_sharpe=gross_sharpe,
        gross_raw_sharpe=gross_raw_sharpe,
        relative_cagr=relative_cagr,
        requested_notional=requested_notional,
        filled_notional=filled_notional,
        rejected_notional=rejected_notional,
        fill_ratio=fill_ratio,
    )


def _aligned_daily_returns(dates: list[date], frame: pl.DataFrame | None) -> np.ndarray:
    if frame is None:
        return np.zeros(len(dates) - 1, dtype=float)
    required = {"trade_date", "daily_return"}
    if missing := required - set(frame.columns):
        raise DataQualityError(f"risk-free series missing columns: {sorted(missing)}")
    duplicates = frame.group_by("trade_date").len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise DataQualityError("risk-free series contains duplicate trade dates")
    lookup = dict(frame.select("trade_date", "daily_return").iter_rows())
    missing_dates = [value for value in dates[1:] if value not in lookup]
    if missing_dates:
        raise DataQualityError(
            f"risk-free series lacks {len(missing_dates)} market dates; first missing={missing_dates[0]}"
        )
    values = np.asarray([lookup[value] for value in dates[1:]], dtype=float)
    validate_finite_array(values, context="risk-free series")
    if np.any(values <= -1.0):
        raise DataQualityError("risk-free series contains return <= -100%")
    return values


def _liquidity_cost(
    current: np.ndarray,
    requested: np.ndarray,
    *,
    capital: float,
    adv: np.ndarray,
    model: ExecutionCostModel,
    quoted_full_spread_bps: np.ndarray | None = None,
    sigma_daily: np.ndarray | None = None,
    quoted_spread_bps: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    if quoted_full_spread_bps is None:
        # Compatibility alias; source bid/ask observations are full spreads.
        quoted_full_spread_bps = quoted_spread_bps
    if quoted_full_spread_bps is None:
        quoted_full_spread_bps = np.full_like(adv, np.nan, dtype=float)
    delta = requested - current
    participation = np.full_like(delta, np.nan, dtype=float)
    finite_adv = np.isfinite(adv) & (adv > 0)
    if np.any((np.abs(delta) > 1e-15) & ~finite_adv):
        raise DataQualityError("liquidity cost model requires positive point-in-time 21-day ADV")
    max_delta = np.where(finite_adv, model.participation_cap * adv / capital, 0.0)
    executed_delta = np.sign(delta) * np.minimum(np.abs(delta), max_delta)
    executed = np.maximum(current + executed_delta, 0.0)
    if executed.sum() > 1.0:
        buys = executed_delta > 0
        excess = float(executed.sum() - 1.0)
        buy_total = float(executed_delta[buys].sum())
        if buy_total > 0:
            executed[buys] -= executed_delta[buys] * min(1.0, excess / buy_total)
    order_fraction = np.abs(executed - current)
    participation[finite_adv] = order_fraction[finite_adv] * capital / adv[finite_adv]
    limits = model.adv_tier_limits
    tier_half_spread_bps = np.select(
        [adv <= limits[0], adv <= limits[1]],
        model.adv_tier_one_way_half_spread_bps[:2],
        default=model.adv_tier_one_way_half_spread_bps[2],
    ).astype(float)
    if model.use_eod_quoted_spread:
        one_way_half_spread_bps = np.where(
            np.isfinite(quoted_full_spread_bps),
            quoted_full_spread_bps / 2.0,
            tier_half_spread_bps,
        )
    else:
        one_way_half_spread_bps = tier_half_spread_bps
    fees = float(np.sum(order_fraction) * model.fee_bps / 10_000.0)
    spread = float(np.sum(order_fraction * one_way_half_spread_bps) / 10_000.0)
    root_participation = np.sqrt(np.nan_to_num(participation, nan=0.0))
    if model.impact_model == "volatility_scaled":
        if sigma_daily is None:
            raise DataQualityError("volatility-scaled impact requires point-in-time daily volatility")
        sigma = np.asarray(sigma_daily, dtype=float)
        traded = order_fraction > 1e-15
        if np.any(traded & (~np.isfinite(sigma) | (sigma < 0))):
            raise DataQualityError(
                "volatility-scaled impact requires finite point-in-time daily volatility"
            )
        # Pre-registered challenger: Y * sigma_daily * sqrt(Q / ADV).
        impact_rate = model.volatility_impact_y * np.nan_to_num(sigma, nan=0.0) * root_participation
        impact = float(np.sum(order_fraction * impact_rate))
    else:
        realized_impact_bps = model.impact_bps * root_participation
        impact = float(np.sum(order_fraction * realized_impact_bps) / 10_000.0)
    return executed, {"fees": fees, "spread": spread, "impact": impact}, participation


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
    risk_free_returns: pl.DataFrame | None = None,
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
    if config.cost_model.mode not in {"fixed_bps", "liquidity"}:
        raise ValueError("unsupported cost model")
    if config.cost_model.impact_model not in {
        "sqrt_participation_bps",
        "volatility_scaled",
    }:
        raise ValueError("unsupported impact model")
    if not 0 < config.cost_model.participation_cap <= 1:
        raise ValueError("participation_cap must be in (0, 1]")
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
    adv = np.full_like(close, np.nan)
    quoted_full_spread_bps = np.full_like(close, np.nan)
    sigma_daily = np.full_like(close, np.nan)
    if config.cost_model.mode == "liquidity":
        if "traded_value_brl" in used_prices.columns and "volume" in used_prices.columns:
            traded_value = pl.coalesce("traded_value_brl", "volume")
        elif "traded_value_brl" in used_prices.columns:
            traded_value = pl.col("traded_value_brl")
        elif "volume" in used_prices.columns:
            traded_value = pl.col("volume")
        else:
            raise DataQualityError("liquidity cost model requires B3 traded value in BRL")
        liquidity_long = (
            used_prices.sort(["ticker", "trade_date"])
            .with_columns(
                traded_value
                .rolling_mean(window_size=21, min_samples=5)
                .shift(1)
                .over("ticker")
                .alias("__adv"),
                pl.col(price_column).pct_change().over("ticker").alias("__daily_return"),
            )
            .with_columns(
                pl.col("__daily_return")
                .rolling_std(window_size=21, min_samples=5)
                .shift(1)
                .over("ticker")
                .alias("__sigma_daily")
            )
        )
        if {"best_bid", "best_ask"} <= set(used_prices.columns):
            liquidity_long = liquidity_long.with_columns(
                pl.when(
                    (pl.col("best_bid") > 0)
                    & (pl.col("best_ask") >= pl.col("best_bid"))
                )
                .then(
                    (pl.col("best_ask") - pl.col("best_bid"))
                    / ((pl.col("best_ask") + pl.col("best_bid")) / 2.0)
                    * 10_000.0
                )
                .otherwise(None)
                .alias("__quoted_full_spread_bps")
            )
        else:
            liquidity_long = liquidity_long.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("__quoted_full_spread_bps")
            )
        adv_frame = liquidity_long.select("trade_date", "ticker", "__adv").pivot(
            index="trade_date", on="ticker", values="__adv"
        )
        spread_frame = liquidity_long.select(
            "trade_date", "ticker", "__quoted_full_spread_bps"
        ).pivot(
            index="trade_date", on="ticker", values="__quoted_full_spread_bps"
        )
        sigma_frame = liquidity_long.select(
            "trade_date", "ticker", "__sigma_daily"
        ).pivot(
            index="trade_date", on="ticker", values="__sigma_daily"
        )
        adv = close_frame.select("trade_date").join(
            adv_frame, on="trade_date", how="left", validate="1:1"
        ).select(tickers).to_numpy()
        quoted_full_spread_bps = close_frame.select("trade_date").join(
            spread_frame, on="trade_date", how="left", validate="1:1"
        ).select(tickers).to_numpy()
        sigma_daily = close_frame.select("trade_date").join(
            sigma_frame, on="trade_date", how="left", validate="1:1"
        ).select(tickers).to_numpy()
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
        issuer_by_ticker: dict[str, str] = {}
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
            issuer_by_ticker[ticker] = issuer
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
                        "issuer_id": issuer_by_ticker[ticker],
                        "target_weight": 1.0 / len(selected),
                    }
                )
        rebalance_lookup[execution_date] = target
        rebalance_signal_dates[execution_date] = signal_date
        position_counts.append(len(selected))

    if not rebalance_lookup:
        raise DataQualityError("no signal has a future executable market date")
    first_signal_date = min(rebalance_signal_dates.values())
    start_index = date_to_index[first_signal_date]
    capital = config.initial_capital
    gross_capital = config.initial_capital
    equity = [capital]
    gross_equity = [gross_capital]
    portfolio_daily: list[float] = []
    total_turnover = 0.0
    excluded_transitions: set[tuple[str, object, object]] = set()
    weights = np.zeros(len(tickers))
    cash_weight = 1.0
    contribution_rows: list[dict[str, object]] = []
    stale_age = np.zeros(len(tickers), dtype=int)
    risk_free_daily = _aligned_daily_returns(dates[start_index:], risk_free_returns)
    realized_risk_free: list[float] = []
    cost_rows: list[dict[str, object]] = []
    cost_totals = {"fees": 0.0, "spread": 0.0, "impact": 0.0}
    execution_totals = {"requested_notional": 0.0, "filled_notional": 0.0}

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
        rf_return = float(risk_free_daily[index - start_index - 1])
        day_return = (
            float(weights[active] @ asset_returns[active]) if np.any(active) else 0.0
        ) + cash_weight * rf_return
        if not np.isfinite(day_return) or day_return <= -1.0:
            raise DataQualityError(f"invalid portfolio return {day_return} for {previous_date} -> {current_date}")
        capital *= 1.0 + day_return
        gross_capital *= 1.0 + day_return
        growth = 1.0 + day_return
        drifted = np.zeros_like(weights)
        drifted[active] = weights[active] * (1.0 + asset_returns[active]) / growth
        weights = drifted
        cash_weight = cash_weight * (1.0 + rf_return) / growth

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
            requested_target = target.copy()
            component_costs = {"fees": 0.0, "spread": 0.0, "impact": 0.0}
            participation = np.full(len(tickers), np.nan)
            if config.cost_model.mode == "liquidity":
                target, component_costs, participation = _liquidity_cost(
                    weights,
                    requested_target,
                    capital=capital,
                    adv=adv[index],
                    quoted_full_spread_bps=quoted_full_spread_bps[index],
                    sigma_daily=sigma_daily[index],
                    model=config.cost_model,
                )
                capped = np.abs(target - requested_target) > 1e-12
                for column in np.flatnonzero(capped):
                    rejected_rows.append(
                        {
                            "signal_date": rebalance_signal_dates[current_date],
                            "execution_date": current_date,
                            "ticker": tickers[column],
                            "reason": "participation_cap_partial_fill",
                        }
                    )
            target_cash = max(0.0, 1.0 - float(target.sum()))
            execution_totals["requested_notional"] += float(
                np.abs(requested_target - weights).sum() * capital
            )
            execution_totals["filled_notional"] += float(
                np.abs(target - weights).sum() * capital
            )
            turnover = _one_way_turnover(weights, target, current_cash=cash_weight, target_cash=target_cash)
            if config.cost_model.mode == "fixed_bps":
                component_costs["fees"] = turnover * config.transaction_cost_bps / 10_000.0
            cost_fraction = float(sum(component_costs.values()))
            if cost_fraction >= 1.0:
                raise DataQualityError(f"transaction costs consume all capital on {current_date}")
            total_turnover += turnover
            for component, fraction in component_costs.items():
                amount = capital * fraction
                cost_totals[component] += amount
                if amount > 0:
                    cost_rows.append(
                        {
                            "trade_date": current_date,
                            "component": component,
                            "cost_amount": amount,
                            "cost_fraction": fraction,
                            "capital": capital,
                            "max_participation": float(np.nanmax(participation))
                            if np.any(np.isfinite(participation))
                            else None,
                        }
                    )
            capital *= 1.0 - cost_fraction
            weights = target
            cash_weight = target_cash

        net_day_return = float(capital / previous_capital - 1.0)
        portfolio_daily.append(net_day_return)
        realized_risk_free.append(rf_return)
        equity.append(capital)
        gross_equity.append(gross_capital)

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
        risk_free_returns=np.asarray(realized_risk_free),
        cost_totals=cost_totals,
        gross_equity=np.asarray(gross_equity),
        execution_totals=execution_totals,
    )
    curve = pl.DataFrame(
        {
            "trade_date": dates[start_index:],
            "portfolio_value": equity_array,
            "gross_portfolio_value": np.asarray(gross_equity),
            "net_return": [None, *portfolio_daily],
            "risk_free_return": [None, *realized_risk_free],
        }
    ).with_columns(
        pl.lit(price_column).alias("price_basis"),
        (
            pl.col("gross_portfolio_value")
            / pl.col("gross_portfolio_value").shift(1)
            - 1.0
        ).alias("gross_return"),
        (pl.col("net_return") - pl.col("risk_free_return")).alias("excess_return"),
    )
    selections = pl.DataFrame(selection_rows) if selection_rows else pl.DataFrame(
        schema={
            "signal_date": pl.Date,
            "execution_date": pl.Date,
            "ticker": pl.Utf8,
            "issuer_id": pl.Utf8,
            "target_weight": pl.Float64,
        }
    )
    contributions = pl.DataFrame(contribution_rows) if contribution_rows else pl.DataFrame(
        schema={"trade_date": pl.Date, "ticker": pl.Utf8, "weight_at_previous_close": pl.Float64, "asset_return": pl.Float64, "return_contribution": pl.Float64, "stale_age_sessions": pl.Int64}
    )
    rejected = pl.DataFrame(rejected_rows) if rejected_rows else pl.DataFrame(
        schema={"signal_date": pl.Date, "execution_date": pl.Date, "ticker": pl.Utf8, "reason": pl.Utf8}
    )
    costs = pl.DataFrame(cost_rows) if cost_rows else pl.DataFrame(
        schema={"trade_date": pl.Date, "component": pl.Utf8, "cost_amount": pl.Float64, "cost_fraction": pl.Float64, "capital": pl.Float64, "max_participation": pl.Float64}
    )
    return DetailedBacktest(curve, summary, selections, contributions, rejected, costs)


def run_monthly_topk_backtest(
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    *,
    score_column: str,
    config: BacktestConfig = _DEFAULT_CONFIG,
    risk_free_returns: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, BacktestSummary]:
    result = run_monthly_topk_backtest_detailed(
        prices,
        signals,
        score_column=score_column,
        config=config,
        risk_free_returns=risk_free_returns,
    )
    return result.curve, result.summary
