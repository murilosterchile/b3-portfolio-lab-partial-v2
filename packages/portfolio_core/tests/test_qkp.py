import itertools

import numpy as np
import pytest
from portfolio_core.data_quality import DataQualityError
from portfolio_core.optimizer import (
    Candidate,
    QKPInstance,
    build_portfolio_qkp,
    solve_exact_branch_and_bound,
)


def brute_force(instance: QKPInstance) -> tuple[float, tuple[int, ...]]:
    best = -1e100
    best_set: tuple[int, ...] = ()
    n = len(instance.names)
    max_card = instance.max_cardinality or n
    for bits in itertools.product([0, 1], repeat=n):
        selected = tuple(i for i, bit in enumerate(bits) if bit)
        if not (instance.min_cardinality <= len(selected) <= max_card):
            continue
        if instance.costs[list(selected)].sum() > instance.capacity + 1e-9:
            continue
        value = float(instance.linear_values[list(selected)].sum()) if selected else 0.0
        for a, i in enumerate(selected):
            for j in selected[a + 1 :]:
                value += float(instance.pair_values[i, j])
        if value > best:
            best, best_set = value, selected
    return best, best_set


def direct_utility_oracle(
    candidates: list[Candidate],
    correlation: np.ndarray,
    selected: tuple[int, ...],
    *,
    risk_aversion: float,
    uncertainty_penalty: float,
    horizon_bars: int,
) -> float:
    """Economic oracle: U = mu'w - lambda*w'Sigma*w for w=x/k."""
    mu = np.asarray(
        [
            candidate.predicted_excess_return
            - uncertainty_penalty * candidate.uncertainty
            for candidate in candidates
        ]
    )
    annual_volatility = np.asarray([candidate.volatility_annual for candidate in candidates])
    covariance = (
        correlation
        * np.outer(annual_volatility, annual_volatility)
        * horizon_bars
        / 252.0
    )
    weights = np.zeros(len(candidates))
    weights[list(selected)] = 1.0 / len(selected)
    return float(weights @ mu - risk_aversion * weights @ covariance @ weights)


def test_exact_bnb_matches_bruteforce_with_negative_pairs() -> None:
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = 8
        linear = rng.normal(3, 3, n)
        raw = rng.normal(0, 2, (n, n))
        pair = np.triu(raw, 1)
        pair = pair + pair.T
        costs = rng.integers(1, 8, n).astype(float)
        instance = QKPInstance(
            names=tuple(f"A{i}" for i in range(n)),
            linear_values=linear,
            pair_values=pair,
            costs=costs,
            capacity=15,
            min_cardinality=2,
            max_cardinality=5,
        )
        expected, _ = brute_force(instance)
        actual = solve_exact_branch_and_bound(instance)
        assert actual.exact
        assert abs(actual.objective - expected) < 1e-8


def test_portfolio_qkp_rejects_non_finite_inputs() -> None:
    candidates = [Candidate("TEST3", 10.0, np.nan, 0.1)]

    with pytest.raises(DataQualityError, match="expected_return"):
        build_portfolio_qkp(
            candidates,
            correlation=np.eye(1),
            budget=1000.0,
            min_positions=1,
            max_positions=1,
            risk_aversion=1.0,
            expected_return_horizon_bars=21,
            covariance_horizon_bars=21,
        )


def test_portfolio_qkp_rejects_invalid_correlation() -> None:
    candidates = [Candidate("A3", 10.0, 0.1, 0.1), Candidate("B3", 10.0, 0.1, 0.1)]
    correlation = np.array([[1.0, 1.2], [1.2, 1.0]])

    with pytest.raises(DataQualityError, match=r"\[-1, 1\]"):
        build_portfolio_qkp(
            candidates,
            correlation=correlation,
            budget=1000.0,
            min_positions=1,
            max_positions=2,
            risk_aversion=1.0,
            expected_return_horizon_bars=21,
            covariance_horizon_bars=21,
        )


def test_stock_split_does_not_change_qkp_selection() -> None:
    base = [
        Candidate("A3", 100.0, 0.10, 0.01),
        Candidate("B3", 20.0, 0.08, 0.01),
        Candidate("C3", 30.0, 0.06, 0.01),
    ]
    split = [
        Candidate(
            candidate.ticker,
            candidate.price / 10.0,
            candidate.predicted_excess_return,
            candidate.uncertainty,
        )
        for candidate in base
    ]
    kwargs = dict(
        correlation=np.eye(3),
        budget=1_000.0,
        min_positions=2,
        max_positions=2,
        risk_aversion=1.0,
        expected_return_horizon_bars=21,
        covariance_horizon_bars=21,
    )
    before = solve_exact_branch_and_bound(build_portfolio_qkp(base, **kwargs))
    after = solve_exact_branch_and_bound(build_portfolio_qkp(split, **kwargs))
    assert before.selected_names == after.selected_names


def test_qkp_never_selects_two_share_classes_from_same_issuer() -> None:
    candidates = [
        Candidate("AAAA3", 10.0, 0.20, 0.0, issuer_id="issuer-a"),
        Candidate("AAAA4", 11.0, 0.19, 0.0, issuer_id="issuer-a"),
        Candidate("BBBB3", 12.0, 0.10, 0.0, issuer_id="issuer-b"),
    ]
    instance = build_portfolio_qkp(
        candidates,
        correlation=np.eye(3),
        budget=1_000.0,
        min_positions=2,
        max_positions=2,
        risk_aversion=0.0,
        expected_return_horizon_bars=21,
        covariance_horizon_bars=21,
    )
    result = solve_exact_branch_and_bound(instance)
    assert len({instance.issuer_ids[index] for index in result.selected_indices}) == 2


def test_qkp_objective_matches_direct_utility_oracle_across_k() -> None:
    candidates = [
        Candidate("A3", 10.0, 0.045, 0.010, volatility_annual=0.18),
        Candidate("B3", 10.0, 0.040, 0.004, volatility_annual=0.22),
        Candidate("C3", 10.0, 0.031, 0.002, volatility_annual=0.14),
        Candidate("D3", 10.0, 0.025, 0.001, volatility_annual=0.12),
    ]
    correlation = np.array(
        [
            [1.0, 0.55, 0.10, 0.05],
            [0.55, 1.0, 0.15, 0.10],
            [0.10, 0.15, 1.0, 0.30],
            [0.05, 0.10, 0.30, 1.0],
        ]
    )
    risk_aversion = 0.7
    uncertainty_penalty = 0.5
    for k in (1, 2, 3):
        instance = build_portfolio_qkp(
            candidates,
            correlation=correlation,
            budget=1_000.0,
            min_positions=k,
            max_positions=k,
            fixed_k=k,
            risk_aversion=risk_aversion,
            uncertainty_penalty=uncertainty_penalty,
            expected_return_horizon_bars=21,
            covariance_horizon_bars=21,
        )
        result = solve_exact_branch_and_bound(instance)
        utilities = {
            selected: direct_utility_oracle(
                candidates,
                correlation,
                selected,
                risk_aversion=risk_aversion,
                uncertainty_penalty=uncertainty_penalty,
                horizon_bars=21,
            )
            for selected in itertools.combinations(range(len(candidates)), k)
        }
        assert result.objective == pytest.approx(max(utilities.values()))
        assert result.objective == pytest.approx(utilities[result.selected_indices])


def test_qkp_rejects_mixed_horizons() -> None:
    with pytest.raises(DataQualityError, match="horizons must match"):
        build_portfolio_qkp(
            [Candidate("A3", 10.0, 0.03, 0.01)],
            correlation=np.eye(1),
            budget=1_000.0,
            min_positions=1,
            max_positions=1,
            risk_aversion=0.7,
            expected_return_horizon_bars=21,
            covariance_horizon_bars=1,
        )
