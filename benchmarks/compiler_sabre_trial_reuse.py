"""Inspect native-CZ selection using only completed production SABRE trials.

This opt-in experiment preserves production search budgets. Equal-cost routes
that the incumbent search prunes are unavailable; compare delivered outputs
against the earlier experiment as well as production before drawing conclusions.
"""

from __future__ import annotations

from collections.abc import Callable

from benchmarks.compiler_sabre_native_cost import _choose_native_route
from flagquantum.compiler.topology import sabre
from flagquantum.compiler.topology.coupling import CouplingMap
from flagquantum.core.ir import CircuitIR

_ORIGINAL = sabre._select_sabre_trials


def _select(
    program: CircuitIR,
    coupling: CouplingMap,
    baseline: sabre.SabrePlan,
    *,
    cost: Callable[[sabre.SabrePlan], int] | None = None,
    native_budget: int = 256,
) -> sabre.SabrePlan:
    if cost is not None or not baseline.swaps:
        return _ORIGINAL(program, coupling, baseline, cost=cost)
    trials = list(sabre._sabre_trial_costs(program, coupling, baseline))
    incumbent, swap_limit = min(trials, key=lambda item: item[1])
    candidates = list(
        dict.fromkeys(
            plan for plan, swaps in trials if swaps <= swap_limit and plan != incumbent
        )
    )
    if not candidates:
        return incumbent
    return _choose_native_route(
        program, coupling, incumbent, candidates, native_budget=native_budget
    )
