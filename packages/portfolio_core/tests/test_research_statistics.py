import numpy as np

from portfolio_core.research.gates import evaluate_acceptance_gates
from portfolio_core.research.statistics import (
    deflated_sharpe_ratio,
    probability_backtest_overfitting,
    probabilistic_sharpe_ratio,
)


def test_gate_cannot_be_production_ready_without_verified_temporal_tests() -> None:
    result = evaluate_acceptance_gates(
        rank_ic_ci_low=0.01,
        top10_spread_ci_low=0.01,
        excess_sharpe=1.0,
        max_drawdown=-0.20,
        annualized_turnover=2.0,
        excess_cagr_ci_low_vs_cdi=0.01,
        excess_cagr_ci_low_vs_ibov=0.01,
        deflated_sharpe_probability=0.99,
        number_of_trials=10,
        leakage_tests_passed=False,
        diagnostic_only_acknowledged=True,
    )
    assert result["production_ready"] is False


def test_probabilistic_and_deflated_sharpe_are_probabilities() -> None:
    returns = np.array([0.01, 0.02, -0.005, 0.015, 0.003, 0.012] * 4)
    psr = probabilistic_sharpe_ratio(returns)
    dsr, hurdle = deflated_sharpe_ratio(
        returns, number_of_trials=20, trials_sharpe_std=0.25
    )
    assert 0.0 <= psr <= 1.0
    assert 0.0 <= dsr <= 1.0
    assert hurdle > 0.0


def test_pbo_accepts_aligned_strategy_matrix() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.normal(0.001, 0.02, size=(48, 4))
    pbo = probability_backtest_overfitting(matrix)
    assert 0.0 <= pbo <= 1.0
