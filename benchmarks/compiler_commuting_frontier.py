"""Inspect one additional routing trial using proven commuting frontiers.

This experiment reuses the native structural commutation checker and SABRE
planner. Production selection stays the incumbent. No public routing policy or
compiler default is changed by importing this module.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

from flagquantum.compiler.optimization.commutation import commute
from flagquantum.compiler.topology import sabre
from flagquantum.compiler.topology.coupling import CouplingMap
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import get_operator_schema

_MAX_COMMUTING_RUN = 20
_ORIGINAL = sabre._select_sabre_trials


def _eligible(instruction: Instruction) -> bool:
    schema = get_operator_schema(instruction.name)
    return (
        not instruction.metadata
        and instruction.matrix is None
        and schema is not None
        and schema.unitary
        and len(instruction.wires) in (1, 2)
    )


def _commuting_dependencies(
    instructions: tuple[Instruction, ...],
) -> tuple[list[list[int]], tuple[int, ...]] | None:
    """Relax quantum order only within bounded, pairwise commuting wire runs.

    Consecutive runs are completely ordered on their shared wire. With run size
    at most B and arity at most two, the graph has at most 2 * B * N edges.
    Annotations, custom matrices and nonunitaries retain the production graph.
    """
    if not all(_eligible(instruction) for instruction in instructions):
        return None
    positions_by_wire: dict[int, list[int]] = {}
    for position, instruction in enumerate(instructions):
        for wire in instruction.wires:
            positions_by_wire.setdefault(wire, []).append(position)
    successors: list[list[int]] = [[] for _ in instructions]
    pending = [0] * len(instructions)
    relaxed = False
    for positions in positions_by_wire.values():
        runs: list[list[int]] = []
        for position in positions:
            if (
                runs
                and len(runs[-1]) < _MAX_COMMUTING_RUN
                and all(
                    commute(instructions[member], instructions[position])
                    for member in runs[-1]
                )
            ):
                runs[-1].append(position)
                relaxed = True
            else:
                runs.append([position])
        for previous, current in pairwise(runs):
            for before in previous:
                successors[before].extend(current)
            for after in current:
                pending[after] += len(previous)
    return (successors, tuple(pending)) if relaxed else None


def _select(
    program: CircuitIR,
    coupling: CouplingMap,
    baseline: sabre.SabrePlan,
    *,
    cost: Callable[[sabre.SabrePlan], int] | None = None,
) -> sabre.SabrePlan:
    """Retain the production incumbent unless one relaxed trial costs less."""
    incumbent = _ORIGINAL(program, coupling, baseline, cost=cost)
    if not baseline.swaps:
        return incumbent
    dependencies = _commuting_dependencies(program.instructions)
    if dependencies is None:
        return incumbent
    score = (
        cost
        if cost is not None
        else lambda plan: sabre._restored_swap_count(plan, coupling)
    )
    incumbent_cost = score(incumbent)
    candidate = sabre._plan_sabre_trial(
        program,
        coupling,
        initial_layout=baseline.initial_logical_to_physical,
        committed_swap_limit=incumbent_cost,
        _dependencies=dependencies,
    )
    return (
        candidate
        if candidate is not None and score(candidate) < incumbent_cost
        else incumbent
    )
