"""AO-04 opt-in target selection for a complete native register program."""

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler._construction.constraints import CircuitLimits
from flagquantum.compiler._construction.frontend_program import capture_program
from flagquantum.compiler._construction.selection import (
    Constraints,
    ConstructionBudget,
    Selection,
    select_register_program,
)
from flagquantum.compiler._construction.target import PHASE_BASIS, Target
from flagquantum.compiler.topology.coupling import CouplingMap
from flagquantum.synth import Const, Output, QNum, allocate, hadamard_transform, qfunc

pytestmark = pytest.mark.unit

_BASIS = PHASE_BASIS
_CONSTRAINTS = Constraints()
_BUDGET = ConstructionBudget()
_TARGET = Target()


@qfunc
def energy(x: Const[QNum], y: Const[QNum], result: Output[QNum]) -> None:
    result |= x * x + y


@qfunc
def main(x: Output[QNum], y: Output[QNum], result: Output[QNum]) -> None:
    allocate(2, x)
    allocate(2, y)
    hadamard_transform(x)
    hadamard_transform(y)
    energy(x, y, result)


def _select(
    *,
    constraints: Constraints = _CONSTRAINTS,
    budget: ConstructionBudget = _BUDGET,
    target: Target = _TARGET,
    include_boolean: bool = True,
) -> Selection:
    root, functions = capture_program(main)
    return select_register_program(
        root,
        functions,
        constraints=constraints,
        budget=budget,
        target=target,
        include_boolean=include_boolean,
    )


def test_checked_complete_native_program_meets_sentinel_and_is_deterministic() -> None:
    constraints = Constraints(CircuitLimits(9, 24, 20), 1)
    first = _select(constraints=constraints)
    second = _select(constraints=constraints)
    assert first.complete and first.selected is not None
    assert tuple(row.strategy for row in first.rows) == (
        "retain",
        "recompute",
        "boolean",
    )
    assert first.selected.strategy == "boolean"
    target = first.selected.target
    assert target is not None and second.selected is not None
    assert second.selected.target is not None
    assert target.content_identities == second.selected.target.content_identities
    result = fq.run(target.executable).statevector().reshape(-1)
    expected = torch.zeros(512, dtype=torch.complex128)
    for x in range(4):
        for y in range(4):
            expected[(x << 7) | (y << 5) | ((x * x + y) << 1)] = 0.25
    torch.testing.assert_close(result, expected, rtol=0, atol=1e-12)


def test_target_profile_can_change_the_selected_family() -> None:
    logical = _select()
    assert logical.selected is not None and logical.selected.strategy == "boolean"
    old_widths = {
        row.resources.n_qubits for row in logical.rows[:-1] if row.resources is not None
    }
    profiles = tuple(
        CouplingMap(n, tuple((i, i + 1) for i in range(n - 1)))
        for n in sorted(old_widths)
    )
    routed = _select(target=Target(basis=_BASIS, couplings=profiles))
    assert routed.complete and routed.selected is not None
    assert routed.selected.strategy in {"retain", "recompute"}
    assert routed.rows[-1].diagnostics[0].code == "target.unsupported"
    assert routed.selected.target is not None
    assert {
        gate.name for gate in routed.selected.target.executable.instructions
    } <= set(_BASIS)
    assert dict(routed.selected.stage_resources)["source"].n_operations < (
        routed.selected.resources.n_operations
    )


def test_final_constraints_and_incomplete_search_keep_existing_semantics() -> None:
    strict = Constraints(CircuitLimits(9, 24, 20), 1)
    decomposed = _select(constraints=strict, target=Target(basis=_BASIS))
    assert decomposed.complete and decomposed.selected is None
    assert any(
        d.code == "synthesis.constraints.max_operations"
        for d in decomposed.rows[-1].diagnostics
    )
    partial = _select(budget=ConstructionBudget(candidate_checks=1))
    assert not partial.complete and partial.selected is None
    assert partial.diagnostics[0].code == "selection.budget"
    default = _select(include_boolean=False)
    assert tuple(row.strategy for row in default.rows) == ("retain", "recompute")


@qfunc
def wide(x: Output[QNum], y: Output[QNum], result: Output[QNum]) -> None:
    allocate(4, x)
    allocate(4, y)
    result |= x + y


def test_outside_the_boolean_envelope_preserves_checked_arithmetic() -> None:
    root, functions = capture_program(wide)
    result = select_register_program(root, functions, include_boolean=True)
    assert result.complete and result.selected is not None
    assert result.selected.strategy in {"retain", "recompute"}
    refusal = result.rows[-1].diagnostics[0]
    assert refusal.code == "selection.boolean.refused"
    assert "input or output width" in refusal.message
