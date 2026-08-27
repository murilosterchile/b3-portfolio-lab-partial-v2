from __future__ import annotations

from collections import Counter

import numpy as np

from .types import OptimizationResult, QKPInstance


def solve_exact_branch_and_bound(instance: QKPInstance, *, max_n: int = 32) -> OptimizationResult:
    """Exact branch-and-bound fallback for small QKP instances.

    The upper bound deliberately ignores budget/cardinality and includes every positive remaining
    linear/pair coefficient, so it is loose but mathematically valid even with negative q_ij.
    This fallback prioritizes correctness and testability; SCIP is the default backend for larger
    production-shaped instances.
    """
    n = len(instance.names)
    if n > max_n:
        raise ValueError(f"Fallback exact B&B supports n <= {max_n}; install/use SCIP for n={n}")

    max_card = instance.max_cardinality if instance.max_cardinality is not None else n
    # Good incumbent earlier: order by positive utility density plus positive interaction potential.
    positive_interactions = np.maximum(instance.pair_values, 0.0).sum(axis=1)
    density = (np.maximum(instance.linear_values, 0.0) + positive_interactions) / np.maximum(
        instance.costs, 1e-9
    )
    order = np.argsort(-density)

    names = tuple(instance.names[i] for i in order)
    linear = instance.linear_values[order]
    pair = instance.pair_values[np.ix_(order, order)]
    costs = instance.costs[order]
    sectors = tuple(instance.sectors[i] for i in order) if instance.sectors is not None else None
    issuer_ids = tuple(instance.issuer_ids[i] for i in order) if instance.issuer_ids is not None else None

    best_value = -float("inf")
    best_selected: tuple[int, ...] = ()

    def upper_bound(index: int, selected: list[int], current: float) -> float:
        remaining = list(range(index, n))
        ub = current
        ub += float(np.maximum(linear[remaining], 0.0).sum()) if remaining else 0.0
        if selected and remaining:
            ub += float(np.maximum(pair[np.ix_(selected, remaining)], 0.0).sum())
        if len(remaining) > 1:
            sub = np.maximum(pair[np.ix_(remaining, remaining)], 0.0)
            ub += float(np.triu(sub, k=1).sum())
        return ub

    def sector_ok(sector_counts: Counter[str], candidate_sector: str | None) -> bool:
        if candidate_sector is None:
            return True
        limit = instance.sector_max_count.get(candidate_sector)
        return limit is None or sector_counts[candidate_sector] + 1 <= limit

    def dfs(
        index: int,
        selected: list[int],
        current: float,
        cost: float,
        sector_counts: Counter[str],
        issuer_counts: Counter[str],
    ) -> None:
        nonlocal best_value, best_selected
        remaining_count = n - index
        if len(selected) > max_card or len(selected) + remaining_count < instance.min_cardinality:
            return
        if cost > instance.capacity + 1e-9:
            return
        if upper_bound(index, selected, current) <= best_value + 1e-12:
            return
        if index == n:
            if instance.min_cardinality <= len(selected) <= max_card and current > best_value:
                best_value = current
                best_selected = tuple(selected)
            return

        # Include branch first to find strong incumbents early.
        candidate_sector = sectors[index] if sectors is not None else None
        candidate_issuer = issuer_ids[index] if issuer_ids is not None else None
        if (
            len(selected) < max_card
            and cost + costs[index] <= instance.capacity + 1e-9
            and sector_ok(sector_counts, candidate_sector)
            and (candidate_issuer is None or issuer_counts[candidate_issuer] == 0)
        ):
            incremental = float(linear[index])
            if selected:
                incremental += float(pair[index, selected].sum())
            selected.append(index)
            if candidate_sector is not None:
                sector_counts[candidate_sector] += 1
            if candidate_issuer is not None:
                issuer_counts[candidate_issuer] += 1
            dfs(index + 1, selected, current + incremental, cost + float(costs[index]), sector_counts, issuer_counts)
            if candidate_issuer is not None:
                issuer_counts[candidate_issuer] -= 1
            if candidate_sector is not None:
                sector_counts[candidate_sector] -= 1
            selected.pop()

        dfs(index + 1, selected, current, cost, sector_counts, issuer_counts)

    dfs(0, [], 0.0, 0.0, Counter(), Counter())
    if best_value == -float("inf"):
        return OptimizationResult((), (), 0.0, 0.0, "infeasible", "exact-bnb", True, 0.0)

    original = tuple(sorted(int(order[i]) for i in best_selected))
    return OptimizationResult(
        selected_indices=original,
        selected_names=tuple(instance.names[i] for i in original),
        objective=float(best_value),
        total_cost=float(instance.costs[list(original)].sum()),
        status="optimal",
        solver="exact-bnb",
        exact=True,
        gap=0.0,
    )
