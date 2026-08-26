from __future__ import annotations

from .types import OptimizationResult, QKPInstance


def solve_exact_scip(instance: QKPInstance, *, time_limit_seconds: float | None = None) -> OptimizationResult:
    """Solve a binary QKP exactly with SCIP via an exact MILP linearization."""
    from pyscipopt import Model, quicksum

    n = len(instance.names)
    model = Model("b3_portfolio_qkp")
    model.hideOutput()
    if time_limit_seconds is not None:
        model.setRealParam("limits/time", float(time_limit_seconds))

    x = [model.addVar(vtype="B", name=f"x_{i}") for i in range(n)]
    y: dict[tuple[int, int], object] = {}
    for i in range(n):
        for j in range(i + 1, n):
            var = model.addVar(vtype="B", name=f"y_{i}_{j}")
            y[i, j] = var
            model.addCons(var <= x[i])
            model.addCons(var <= x[j])
            model.addCons(var >= x[i] + x[j] - 1)

    model.addCons(quicksum(float(instance.costs[i]) * x[i] for i in range(n)) <= instance.capacity)
    model.addCons(quicksum(x) >= instance.min_cardinality)
    if instance.max_cardinality is not None:
        model.addCons(quicksum(x) <= instance.max_cardinality)

    if instance.sectors is not None:
        for sector, limit in instance.sector_max_count.items():
            indices = [i for i, value in enumerate(instance.sectors) if value == sector]
            if indices:
                model.addCons(quicksum(x[i] for i in indices) <= limit)

    objective = quicksum(float(instance.linear_values[i]) * x[i] for i in range(n))
    objective += quicksum(
        float(instance.pair_values[i, j]) * y[i, j]
        for i in range(n)
        for j in range(i + 1, n)
        if abs(float(instance.pair_values[i, j])) > 1e-15
    )
    model.setObjective(objective, sense="maximize")
    model.optimize()

    status = str(model.getStatus()).lower()
    if status in {"infeasible", "inforunbd"}:
        proven = status == "infeasible"
        return OptimizationResult((), (), 0.0, 0.0, status, "SCIP", proven, 0.0 if proven else None)
    if model.getNSols() == 0:
        return OptimizationResult((), (), 0.0, 0.0, status, "SCIP", False, None)

    solution = model.getBestSol()
    selected = tuple(i for i in range(n) if model.getSolVal(solution, x[i]) > 0.5)
    exact = status == "optimal"
    gap = float(model.getGap()) if model.getNSols() else None
    return OptimizationResult(
        selected_indices=selected,
        selected_names=tuple(instance.names[i] for i in selected),
        objective=float(model.getSolObjVal(solution)),
        total_cost=float(instance.costs[list(selected)].sum()) if selected else 0.0,
        status=status,
        solver="SCIP",
        exact=exact,
        gap=gap,
    )
