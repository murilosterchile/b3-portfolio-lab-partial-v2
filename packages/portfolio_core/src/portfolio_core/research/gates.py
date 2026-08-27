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


def evaluate_benchmark_fairness(
    records: list[dict[str, object]],
) -> dict[str, object]:
    """Automated guard against comparing strategies under different economics."""
    if not records:
        return {"passed": False, "reason": "no strategy/cost records"}
    invariant_fields = (
        "initial_capital",
        "evaluation_start",
        "evaluation_end",
        "rebalance_dates",
        "rebalance_periods",
        "execution_delay_bars",
        "participation_cap",
        "fill_policy",
    )
    failures: list[str] = []
    by_cost: dict[object, list[dict[str, object]]] = {}
    by_strategy: dict[object, set[object]] = {}
    for record in records:
        by_cost.setdefault(record["cost_scenario"], []).append(record)
        by_strategy.setdefault(record["strategy"], set()).add(record["cost_scenario"])
    expected_scenarios = set(by_cost)
    for strategy, scenarios in by_strategy.items():
        if scenarios != expected_scenarios:
            failures.append(f"{strategy} does not have every registered cost scenario")
    for scenario, scenario_records in by_cost.items():
        reference = scenario_records[0]
        for record in scenario_records[1:]:
            for field in invariant_fields:
                if record.get(field) != reference.get(field):
                    failures.append(
                        f"{scenario}: {record['strategy']} differs on {field}"
                    )
    return {
        "passed": not failures,
        "strategies": len(by_strategy),
        "cost_scenarios": sorted(str(value) for value in expected_scenarios),
        "failures": failures,
        "rule": (
            "champion and B3 baselines must share initial capital, rebalance dates, "
            "evaluation window, fill policy, participation cap, and execution lag within each cost scenario; "
            "every strategy must be evaluated under every scenario"
        ),
    }


def evaluate_acceptance_gates(
    *,
    rank_ic_ci_low: float,
    top10_spread_ci_low: float,
    excess_sharpe: float,
    max_drawdown: float,
    annualized_turnover: float,
    relative_cagr_ci_low_vs_cdi: float | None = None,
    relative_cagr_ci_low_vs_ibov: float | None = None,
    excess_cagr_ci_low_vs_cdi: float | None = None,
    excess_cagr_ci_low_vs_ibov: float | None = None,
    deflated_sharpe_probability: float,
    number_of_trials: int,
    leakage_tests_passed: bool,
    diagnostic_only_acknowledged: bool,
    benchmark_fairness_passed: bool = True,
) -> dict[str, object]:
    """Pre-registered gates. No 2026 metric is an input to these thresholds."""
    # The excess_* aliases preserve callers predating the wealth-ratio naming;
    # values are expected to be wealth-ratio estimates, never CAGR differences.
    relative_cdi = (
        relative_cagr_ci_low_vs_cdi
        if relative_cagr_ci_low_vs_cdi is not None
        else excess_cagr_ci_low_vs_cdi
    )
    relative_ibov = (
        relative_cagr_ci_low_vs_ibov
        if relative_cagr_ci_low_vs_ibov is not None
        else excess_cagr_ci_low_vs_ibov
    )
    if relative_cdi is None or relative_ibov is None:
        raise ValueError("relative CAGR gate inputs are required")
    model_gates = [
        _gate("rank_ic_ci_low", rank_ic_ci_low, ">", 0.0),
        _gate("top10_spread_ci_low", top10_spread_ci_low, ">", 0.0),
    ]
    portfolio_gates = [
        _gate("excess_sharpe_vs_cdi", excess_sharpe, ">=", 0.75),
        _gate("max_drawdown", max_drawdown, ">=", -0.35),
        _gate("annualized_turnover", annualized_turnover, "<=", 5.0),
        _gate("relative_cagr_ci_low_vs_cdi", relative_cdi, ">", 0.0),
        _gate("relative_cagr_ci_low_vs_ibov", relative_ibov, ">", 0.0),
        _gate("deflated_sharpe_probability", deflated_sharpe_probability, ">=", 0.95),
        _gate("number_of_trials_recorded", number_of_trials, ">=", 1),
    ]
    production_gates = [
        _gate("leakage_tests_passed", leakage_tests_passed, "==", True),
        _gate("diagnostic_only_acknowledged", diagnostic_only_acknowledged, "==", True),
        _gate("benchmark_fairness_passed", benchmark_fairness_passed, "==", True),
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
