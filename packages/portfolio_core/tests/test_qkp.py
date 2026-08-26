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
    )
    before = solve_exact_branch_and_bound(build_portfolio_qkp(base, **kwargs))
    after = solve_exact_branch_and_bound(build_portfolio_qkp(split, **kwargs))
    assert before.selected_names == after.selected_names
