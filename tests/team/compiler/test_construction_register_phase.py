"""Initialized basis-constant phase borrows preserve linear effects and storage."""

from collections.abc import Callable
from dataclasses import replace

import pytest

from examples.synthesis_register_phase import phase_composition
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import Schedule
from flagquantum.compiler._construction.register_lowering import (
    schedule_register_program,
)
from flagquantum.compiler._construction.register_profile import verify_register_program
from flagquantum.compiler._construction.selection import (
    ConstructionBudget,
    select_register_program,
)
from flagquantum.compiler._construction.target import PHASE_BASIS, Target
from flagquantum.compiler._hybrid.model import (
    INDEX,
    Block,
    HybridProgram,
    Operation,
    Region,
    Value,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.compiler.topology.coupling import CouplingMap
from tests.register_program_cases import Registers

pytestmark = pytest.mark.unit


def predicate(x: int, y: int) -> bool:
    return x == y


def program() -> tuple[HybridProgram, tuple[HybridProgram, ...]]:
    graph = capture(predicate, {"x": 1, "y": 1})
    p = Registers()
    x, y = p.declare("x", 1), p.declare("y", 1)
    p.emit("qreg.allocate", x)
    p.emit("qreg.allocate", y)
    p.emit("qreg.phase", x, y, callee=graph.name)
    return p.finish(), (graph,)


def altered(
    change: Callable[[Operation], Operation],
) -> tuple[HybridProgram, tuple[HybridProgram, ...]]:
    model, functions = program()
    block = model.body.blocks[0]
    ops = tuple(
        change(op) if op.name == "qreg.phase" else op for op in block.operations
    )
    return replace(model, body=Region((replace(block, operations=ops),))), functions


@pytest.mark.parametrize("scheduling", ["retain", "recompute"])
def test_phase_borrows_return_initialized_and_have_actual_cleanup_segments(
    scheduling: Schedule,
) -> None:
    model, functions = phase_composition()
    assert verify_register_program(model, functions) is model
    row = schedule_register_program(
        model, functions, n_qubits=24, scheduling=scheduling
    )
    assert tuple(row.registers) == ("x", "y", "product", "reference")
    assert len(row.segments) == 4
    assert row.peak_workspace == max(s.peak_workspace for s in row.segments)
    assert row.peak_workspace < sum(s.peak_workspace for s in row.segments)
    assert row.recomputations == sum(s.recomputations for s in row.segments)
    assert all(s.circuit.n_wires == 4 + s.peak_workspace for s in row.segments)
    assert all(
        any(g.name in {"z", "cz"} for g in row.segments[i].circuit.instructions)
        for i in (1, 3)
    )


@pytest.mark.parametrize(
    "change,code",
    [
        (lambda op: replace(op, attributes={}), "operation"),
        (lambda op: replace(op, attributes={"callee": "missing"}), "callee"),
        (lambda op: replace(op, attributes={"callee": 42}), "callee"),
        (
            lambda op: replace(op, attributes={**op.attributes, "const": True}),
            "operation",
        ),
        (lambda op: replace(op, regions=(Region((Block(),)),)), "operation"),
        (lambda op: replace(op, operands=op.operands[:1]), "signature"),
        (lambda op: replace(op, operands=op.operands[:2]), "signature"),
        (
            lambda op: replace(
                op, operands=(op.operands[0], *op.operands[1:], op.operands[1])
            ),
            "signature",
        ),
        (lambda op: replace(op, results=()), "effect"),
        (lambda op: replace(op, results=(Value(op.results[0].id, INDEX),)), "effect"),
        (lambda op: replace(op, results=(op.operands[0],)), "ssa"),
        (lambda op: replace(op, operands=(op.results[0], *op.operands[1:])), "use"),
    ],
)
def test_invalid_claims_signatures_and_effects_fail_before_lowering(
    change: Callable[[Operation], Operation], code: str
) -> None:
    model, functions = altered(change)
    with pytest.raises(HybridVerificationError, match=f"register.{code}"):
        verify_register_program(model, functions)


def test_phase_does_not_initialize_or_consume_its_argument() -> None:
    def bit(x: int) -> int:
        return x

    graph = capture(bit, {"x": 1})
    p = Registers()
    x = p.declare("x", 1)
    p.emit("qreg.phase", x, callee=graph.name)
    with pytest.raises(HybridVerificationError, match="uninitialized"):
        verify_register_program(p.finish(), (graph,))
    p = Registers()
    x = p.declare("x", 1)
    p.emit("qreg.allocate", x)
    p.emit("qreg.phase", x, callee=graph.name)
    p.emit("qreg.gate", x, gate="h", index=0)
    assert verify_register_program(p.finish(), (graph,))


def test_non_one_bit_predicate_and_wrong_width_are_refused() -> None:
    def wide(x: int, y: int) -> int:
        return x + y

    model, _ = program()
    for graph, code in [
        (capture(wide, {"x": 1, "y": 1}), "phase"),
        (capture(predicate, {"x": 2, "y": 1}), "signature"),
    ]:
        block = model.body.blocks[0]
        ops = tuple(
            (
                replace(op, attributes={"callee": graph.name})
                if op.name == "qreg.phase"
                else op
            )
            for op in block.operations
        )
        with pytest.raises(HybridVerificationError, match=f"register.{code}"):
            verify_register_program(
                replace(model, body=Region((replace(block, operations=ops),))), (graph,)
            )


def test_repeated_actuals_are_borrowed_once_and_cannot_hide_a_consumed_return() -> None:
    p = Registers()
    x = p.declare("x", 1)
    p.emit("qreg.allocate", x)
    p.emit("qreg.phase", x, x, callee="predicate")
    graph = capture(predicate, {"x": 1, "y": 1})
    model = p.finish()
    assert verify_register_program(model, (graph,)) is model
    row = schedule_register_program(model, (graph,), n_qubits=12)
    assert len(row.segments[0].lifetimes.program.body.blocks[0].arguments) == 1
    block = model.body.blocks[0]
    dropped = replace(block.operations[-1], operands=block.operations[-1].operands[:1])
    with pytest.raises(HybridVerificationError, match="register.return"):
        verify_register_program(
            replace(
                model,
                body=Region(
                    (replace(block, operations=(*block.operations[:-1], dropped)),)
                ),
            ),
            (graph,),
        )


@pytest.mark.parametrize(
    "target",
    [
        Target(couplings=(CouplingMap.line(24),)),
        Target(basis=PHASE_BASIS, optimize=True),
        Target(basis=("x", "cx", "u3"), optimize=False),
    ],
)
def test_register_phase_uses_existing_exact_phase_target_admission(
    target: Target,
) -> None:
    model, functions = program()
    with pytest.raises(ValueError, match="phase"):
        select_register_program(model, functions, target=target)


def test_finite_search_and_operation_limits_remain_fail_closed() -> None:
    model, functions = program()
    selection = select_register_program(
        model, functions, budget=ConstructionBudget(candidate_checks=1)
    )
    assert not selection.complete and selection.selected is None
    with pytest.raises(HybridVerificationError, match="operations"):
        schedule_register_program(model, functions, n_qubits=12, max_operations=1)
