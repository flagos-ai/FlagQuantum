"""Cleanup authority follows unchanged computed storage, never a zero assertion."""

from collections.abc import Callable
from dataclasses import replace

import pytest

from examples.synthesis.tutorials.composition.register_cleanup.main import (
    cleanup_composition,
    copy_reference,
    flag_value,
    less,
)
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import (
    Schedule,
    adjoint_value_candidate,
)
from flagquantum.compiler._construction.register_lowering import (
    schedule_register_program,
)
from flagquantum.compiler._construction.register_profile import verify_register_program
from flagquantum.compiler._hybrid.model import (
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    Operation,
    Region,
    Value,
    ValueId,
)
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.core.ir import Instruction
from tests.register_program_cases import Registers, pressure

pytestmark = pytest.mark.unit


def pending() -> tuple[Registers, Value, Value, Value, HybridProgram]:
    graph = capture(less, {"x": 2, "y": 2})
    p = Registers()
    x, y, flag = p.declare("x", 2), p.declare("y", 2), p.declare("flag", 1)
    p.emit("qreg.allocate", x)
    p.emit("qreg.allocate", y)
    p.emit("qreg.compute", flag, x, y, callee=graph.name)
    return p, x, y, flag, graph


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_cleanup_reverses_selected_gates_events_and_reuses_result_ports(
    strategy: Schedule,
) -> None:
    model, functions = cleanup_composition()
    assert verify_register_program(model, functions) is model
    row = schedule_register_program(model, functions, n_qubits=24, scheduling=strategy)
    assert len(row.segments) == 7
    for forward, inverse in (
        (row.segments[1], row.segments[3]),
        (row.segments[4], row.segments[6]),
    ):
        assert inverse.circuit.instructions == forward.circuit.instructions[::-1]
        assert inverse.events == tuple(
            replace(e, action="uncompute" if e.action == "compute" else "compute")
            for e in reversed(forward.events)
        )
        assert inverse.peak_workspace == forward.peak_workspace
        assert inverse.recomputations == forward.recomputations
    assert row.registers["flag"] == (4,)
    assert row.peak_workspace == max(s.peak_workspace for s in row.segments)
    assert row.recomputations == sum(s.recomputations for s in row.segments)


@pytest.mark.parametrize("gate", ["x", "h", "s", "z"])
@pytest.mark.parametrize("target", ["source", "result"])
def test_writes_invalidate_cleanup_even_if_basis_diagonal(
    gate: str, target: str
) -> None:
    p, x, _, flag, graph = pending()
    p.emit("qreg.gate", x if target == "source" else flag, gate=gate, index=0)
    p.emit("qreg.uncompute", flag)
    p.emit("qreg.allocate", flag)
    with pytest.raises(HybridVerificationError, match="register.uncompute"):
        verify_register_program(p.finish(), (graph,))


def test_cancelling_writes_do_not_reinstate_cleanup_authority() -> None:
    p, x, _, flag, graph = pending()
    for _ in range(2):
        p.emit("qreg.gate", x, gate="x", index=0)
    p.emit("qreg.uncompute", flag)
    with pytest.raises(HybridVerificationError, match="register.uncompute"):
        verify_register_program(p.finish(), (graph,))


@pytest.mark.parametrize("action", ["return", "read", "twice"])
def test_cleaned_storage_is_fresh_until_reinitialized(action: str) -> None:
    p, _, _, flag, graph = pending()
    functions = (graph, capture(flag_value, {"flag": 1}))
    p.emit("qreg.uncompute", flag)
    code = "return"
    if action == "read":
        p.emit("qreg.phase", flag, callee="flag_value")
        code = "uninitialized"
    elif action == "twice":
        p.emit("qreg.uncompute", flag)
        code = "uncompute"
    with pytest.raises(HybridVerificationError, match=f"register.{code}"):
        verify_register_program(p.finish(), functions)


@pytest.mark.parametrize("initializer", ["qreg.allocate", "qreg.initialize"])
def test_prepared_or_asserted_zero_is_not_a_compute_proof(initializer: str) -> None:
    p = Registers()
    flag = p.declare("flag", 1)
    p.emit(
        initializer,
        flag,
        **({"value": 0} if initializer.endswith("initialize") else {}),
    )
    p.emit("qreg.uncompute", flag)
    with pytest.raises(HybridVerificationError, match="register.uncompute"):
        verify_register_program(p.finish())


@pytest.mark.parametrize(
    "change,code",
    [
        (lambda op: replace(op, attributes={"trusted_zero": True}), "operation"),
        (lambda op: replace(op, regions=(Region((Block(),)),)), "operation"),
        (lambda op: replace(op, results=()), "effect"),
        (lambda op: replace(op, operands=op.operands[:1]), "signature"),
        (lambda op: replace(op, operands=(*op.operands, op.operands[1])), "uncompute"),
        (lambda op: replace(op, operands=(op.results[0], *op.operands[1:])), "use"),
        (
            lambda op: replace(
                op,
                operands=(
                    Value(ValueId("cleanup_composition", 0), QUANTUM_EFFECT),
                    *op.operands[1:],
                ),
            ),
            "effect",
        ),
    ],
)
def test_cleanup_keeps_closed_operation_effect_and_operand_contracts(
    change: Callable[[Operation], Operation], code: str
) -> None:
    model, functions = cleanup_composition()
    block = model.body.blocks[0]
    operations = tuple(
        change(op) if op.name == "qreg.uncompute" else op for op in block.operations
    )
    malformed = replace(model, body=Region((replace(block, operations=operations),)))
    with pytest.raises(HybridVerificationError, match=f"register.{code}"):
        verify_register_program(malformed, functions)


def test_even_unused_helpers_cannot_uncompute_const_arguments() -> None:
    entry = Value(ValueId("bad", 0), QUANTUM_EFFECT)
    p, x, _, _, graph = pending()
    effect = Value(ValueId("bad", 1), QUANTUM_EFFECT)
    helper = HybridProgram(
        "bad",
        Region(
            (
                Block(
                    (entry, x),
                    (
                        Operation("qreg.uncompute", (entry, x), (effect,)),
                        Operation("qreg.return", (effect, x)),
                    ),
                ),
            )
        ),
    )
    with pytest.raises(HybridVerificationError, match="register.const"):
        verify_register_program(p.finish(), (graph, helper))


@pytest.mark.parametrize("reverse_dependencies", [False, True])
def test_dependencies_must_be_available_in_their_original_encoding(
    reverse_dependencies: bool,
) -> None:
    functions = (capture(copy_reference, {"x": 2}), capture(less, {"x": 2, "y": 2}))
    p = Registers()
    x, y = p.declare("x", 2), p.declare("y", 2)
    mid, flag = p.declare("mid", 2), p.declare("flag", 1)
    for value in (x, y):
        p.emit("qreg.allocate", value)
    p.emit("qreg.compute", mid, x, callee="copy_reference")
    p.emit("qreg.compute", flag, mid, y, callee="less")
    order = (flag, mid) if reverse_dependencies else (mid, flag)
    for value in order:
        p.emit("qreg.uncompute", value)
        p.emit("qreg.allocate", value)
    if reverse_dependencies:
        verify_register_program(p.finish(), functions)
    else:
        with pytest.raises(HybridVerificationError, match="register.uncompute"):
            verify_register_program(p.finish(), functions)


def test_caller_does_not_guess_cleanup_authority_from_helper_output() -> None:
    p, x, y, flag, graph = pending()
    entry = Value(ValueId("compute_helper", 0), QUANTUM_EFFECT)
    following = Value(ValueId("compute_helper", 1), QUANTUM_EFFECT)
    helper = HybridProgram(
        "compute_helper",
        Region(
            (
                Block(
                    (entry, x, y),
                    (
                        next(op for op in p.operations if op.results == (flag,)),
                        Operation(
                            "qreg.compute",
                            (entry, flag, x, y),
                            (following,),
                            {"callee": graph.name},
                        ),
                        Operation("qreg.return", (following, x, y, flag)),
                    ),
                ),
            )
        ),
    )
    old = p.operations[-1]
    p.operations[-1] = replace(
        old,
        name="qreg.call",
        operands=(old.operands[0], x, y, flag),
        attributes={"callee": helper.name},
    )
    p.emit("qreg.uncompute", flag)
    with pytest.raises(HybridVerificationError, match="register.uncompute"):
        verify_register_program(p.finish(), (graph, helper))


def test_const_helper_borrow_preserves_the_callers_cleanup_proof() -> None:
    p, _, _, flag, graph = pending()
    phase = capture(flag_value, {"flag": 1})
    entry = Value(ValueId("borrow_flag", 0), QUANTUM_EFFECT)
    following = Value(ValueId("borrow_flag", 1), QUANTUM_EFFECT)
    helper = HybridProgram(
        "borrow_flag",
        Region(
            (
                Block(
                    (entry, flag),
                    (
                        Operation(
                            "qreg.phase",
                            (entry, flag),
                            (following,),
                            {"callee": phase.name},
                        ),
                        Operation("qreg.return", (following, flag)),
                    ),
                ),
            )
        ),
    )
    p.emit("qreg.call", flag, callee=helper.name)
    p.emit("qreg.uncompute", flag)
    p.emit("qreg.allocate", flag)
    model = p.finish()
    assert verify_register_program(model, (graph, phase, helper)) is model
    row = schedule_register_program(model, (graph, phase, helper), n_qubits=24)
    assert (
        row.segments[-1].circuit.instructions
        == row.segments[0].circuit.instructions[::-1]
    )


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_inverse_preserves_rematerialization_events_and_whole_operation_budget(
    strategy: Schedule,
) -> None:
    graph = capture(pressure, {"x": 1, "y": 1})
    p = Registers()
    x, y, out = p.declare("x", 1), p.declare("y", 1), p.declare("out", 2)
    p.emit("qreg.allocate", x)
    p.emit("qreg.allocate", y)
    p.emit("qreg.compute", out, x, y, callee=graph.name)
    p.emit("qreg.uncompute", out)
    p.emit("qreg.allocate", out)
    model = p.finish()
    row = schedule_register_program(model, (graph,), n_qubits=24, scheduling=strategy)
    forward, inverse = row.segments
    assert inverse.circuit.instructions == forward.circuit.instructions[::-1]
    assert inverse.recomputations == forward.recomputations
    if strategy == "recompute":
        assert inverse.recomputations > 0
    assert len(row.circuit.instructions) == 2 * len(forward.circuit.instructions)
    with pytest.raises(HybridVerificationError, match="lowering.operations"):
        schedule_register_program(
            model,
            (graph,),
            n_qubits=24,
            scheduling=strategy,
            max_operations=len(row.circuit.instructions) - 1,
        )


@pytest.mark.parametrize("damage", ["events", "gate", "payload"])
def test_inverse_rejects_non_value_or_non_exact_candidates(damage: str) -> None:
    model, functions = cleanup_composition()
    original = schedule_register_program(model, functions, n_qubits=24).segments[1]
    if damage == "events":
        candidate = replace(original, events=original.events[:-1])
    else:
        gate = (
            Instruction("h", (0,))
            if damage == "gate"
            else replace(
                original.circuit.instructions[0], metadata={"unverified": True}
            )
        )
        candidate = replace(
            original, circuit=replace(original.circuit, instructions=(gate,))
        )
    with pytest.raises(HybridVerificationError, match="register.adjoint"):
        adjoint_value_candidate(candidate)
