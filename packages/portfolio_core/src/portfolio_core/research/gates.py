from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class GateResult:
    gate_name: str
    operator: Literal[">=", "<=", ">", "<", "=="]
    threshold: float | bool
    actual_value: float | bool
    passed: bool


def _gate(name: str, actual: float | bool, op: str, threshold: float | bool) -> GateResult:
    if op == ">=":
        passed = float(actual) >= float(threshold)
    elif op == "<=":
        passed = float(actual) <= float(threshold)
    elif op == ">":
        passed = float(actual) > float(threshold)
    elif op == "<":
        passed = float(actual) < float(threshold)
    elif op == "==":
        passed = actual == threshold
    else:
        raise ValueError(f"unsupported operator {op}")
    return GateResult(name, op, threshold, actual, bool(passed))


def evaluate_acceptance_gates(
    *,
    rank_ic_ci_low: float,
    top10_spread_ci_low: float,
    excess_sharpe: float,
    max_drawdown: float,
    annualized_turnover: float,
    excess_cagr_ci_low_vs_cdi: float,
    excess_cagr_ci_low_vs_ibov: float,
    deflated_sharpe_probability: float,
    number_of_trials: int,
    leakage_tests_passed: bool,
    diagnostic_only_acknowledged: bool,
) -> dict[str, object]:
    """Pre-registered gates. No 2026 metric is an input to these thresholds."""
    model_gates = [
        _gate("rank_ic_ci_low", rank_ic_ci_low, ">", 0.0),
        _gate("top10_spread_ci_low", top10_spread_ci_low, ">", 0.0),
    ]
    portfolio_gates = [
        _gate("excess_sharpe_vs_cdi", excess_sharpe, ">=", 0.75),
        _gate("max_drawdown", max_drawdown, ">=", -0.35),
        _gate("annualized_turnover", annualized_turnover, "<=", 5.0),
        _gate("excess_cagr_ci_low_vs_cdi", excess_cagr_ci_low_vs_cdi, ">", 0.0),
        _gate("excess_cagr_ci_low_vs_ibov", excess_cagr_ci_low_vs_ibov, ">", 0.0),
        _gate("deflated_sharpe_probability", deflated_sharpe_probability, ">=", 0.95),
        _gate("number_of_trials_recorded", number_of_trials, ">=", 1),
    ]
    production_gates = [
        _gate("leakage_tests_passed", leakage_tests_passed, "==", True),
        _gate("diagnostic_only_acknowledged", diagnostic_only_acknowledged, "==", True),
    ]
    model_ok = all(item.passed for item in model_gates)
    portfolio_ok = all(item.passed for item in portfolio_gates)
    production_ok = model_ok and portfolio_ok and all(item.passed for item in production_gates)
    return {
        "model_signal_accepted": model_ok,
        "portfolio_strategy_accepted": portfolio_ok,
        "production_ready": production_ok,
        "model_signal_gates": [asdict(item) for item in model_gates],
        "portfolio_strategy_gates": [asdict(item) for item in portfolio_gates],
        "production_gates": [asdict(item) for item in production_gates],
    }
