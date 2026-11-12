"""Owned locals cannot escape, alias live storage or hide allocated width."""

from collections.abc import Callable
from dataclasses import replace

import pytest

from flagquantum.compiler._construction.lowering import Schedule
from flagquantum.compiler._construction.register_expansion import expand_register_calls
from flagquantum.compiler._construction.register_lowering import (
    schedule_register_program,
)
from flagquantum.compiler._construction.register_profile import (
    register_ports,
    register_type,
    verify_register_program,
)
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
from tests.register_program_cases import Registers
from tools.synthesis_compiler_examples.local_registers.main import (
    local_composition,
    local_marker,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize("overlapping", [False, True])
def test_local_lifetimes_determine_real_reserved_band_and_full_workspace(
    strategy: Schedule,
    overlapping: bool,
) -> None:
    model, functions = local_composition(overlapping=overlapping)
    assert verify_register_program(model, functions) is model
    names, values, outputs = register_ports(model)
    locals_ = [
        op.results[0]
        for op in model.body.blocks[0].operations
        if op.name == "qreg.local"
    ]
    assert names == {"x": (0, 1), "y": (2, 3), "reference": (4, 5)}
    assert outputs == 6
    assert values[locals_[0].id] == (6,)
    assert values[locals_[1].id] == ((7,) if overlapping else (6,))
    row = schedule_register_program(model, functions, n_qubits=24, scheduling=strategy)
    band = 2 if overlapping else 1
    assert row.peak_workspace == band + max(s.peak_workspace for s in row.segments)
    assert all(
        s.circuit.n_wires == outputs + band + s.peak_workspace for s in row.segments
    )
    assert row.recomputations == sum(s.recomputations for s in row.segments)


def test_many_sequential_local_declarations_do_not_sum_into_live_capacity() -> None:
    p = Registers()
    x = p.declare("x", 1)
    p.emit("qreg.allocate", x)
    for index in range(257):
        value = p.declare(f"tmp{index}", 1, local=True)
        p.emit("qreg.release", value)
    model = p.finish()
    verify_register_program(model, max_qubits=2)
    names, values, outputs = register_ports(model)
    assert outputs == 1 and names == {"x": (0,)}
    assert len(values) == 258 and set(values.values()) == {(0,), (1,)}
    row = schedule_register_program(model, n_qubits=2)
    assert row.peak_workspace == 1 and not row.circuit.instructions


def test_future_outputs_remain_reserved_even_after_earlier_local_release() -> None:
    p = Registers()
    x = p.declare("x", 1)
    p.emit("qreg.allocate", x)
    tmp = p.declare("tmp", 2, local=True)
    p.emit("qreg.release", tmp)
    out = p.declare("out", 2)
    p.emit("qreg.allocate", out)
    model = p.finish()
    verify_register_program(model, max_qubits=5)
    with pytest.raises(HybridVerificationError, match="register.budget"):
        verify_register_program(model, max_qubits=4)
    names, values, outputs = register_ports(model)
    assert names == {"x": (0,), "out": (1, 2)} and outputs == 3
    assert values[tmp.id] == (3, 4)


def test_simultaneously_live_locals_cannot_exceed_construction_capacity() -> None:
    p = Registers()
    out = p.declare("out", 1)
    p.emit("qreg.allocate", out)
    a, b = p.declare("a", 2, local=True), p.declare("b", 2, local=True)
    p.emit("qreg.release", a)
    p.emit("qreg.release", b)
    model = p.finish()
    verify_register_program(model, max_qubits=5)
    with pytest.raises(HybridVerificationError, match="register.budget"):
        verify_register_program(model, max_qubits=4)


@pytest.mark.parametrize("initialized", [False, True])
def test_unreleased_locals_cannot_disappear_at_return(initialized: bool) -> None:
    p = Registers()
    x = p.declare("x", 1)
    tmp = p.declare("tmp", 1, local=True)
    p.emit("qreg.allocate", x)
    if initialized:
        p.emit("qreg.allocate", tmp)
    with pytest.raises(HybridVerificationError, match="register.local"):
        verify_register_program(p.finish())


@pytest.mark.parametrize("operation", ["qreg.allocate", "qreg.gate", "qreg.release"])
def test_a_released_reference_cannot_be_used_again(operation: str) -> None:
    p = Registers()
    x = p.declare("x", 1)
    tmp = p.declare("tmp", 1, local=True)
    p.emit("qreg.allocate", x)
    p.emit("qreg.release", tmp)
    p.emit(
        operation,
        tmp,
        **({"gate": "x", "index": 0} if operation == "qreg.gate" else {}),
    )
    with pytest.raises(HybridVerificationError, match="register.operand"):
        verify_register_program(p.finish())


@pytest.mark.parametrize("initializer", ["qreg.allocate", "qreg.initialize"])
def test_initialized_even_known_zero_local_is_not_releasable(initializer: str) -> None:
    p = Registers()
    x = p.declare("x", 1)
    tmp = p.declare("tmp", 1, local=True)
    p.emit("qreg.allocate", x)
    p.emit(
        initializer, tmp, **({"value": 0} if initializer.endswith("initialize") else {})
    )
    p.emit("qreg.release", tmp)
    with pytest.raises(HybridVerificationError, match="register.initialized"):
        verify_register_program(p.finish())


def test_a_named_output_is_not_a_local_pool_allocation() -> None:
    p = Registers()
    out = p.declare("out", 1)
    p.emit("qreg.release", out)
    with pytest.raises(HybridVerificationError, match="register.release"):
        verify_register_program(p.finish())


@pytest.mark.parametrize(
    "opcode,change,code",
    [
        (
            "qreg.local",
            lambda op: replace(
                op, operands=(Value(ValueId("local_composition", 0), QUANTUM_EFFECT),)
            ),
            "declare",
        ),
        ("qreg.local", lambda op: replace(op, results=()), "declare"),
        ("qreg.local", lambda op: replace(op, attributes={"name": ""}), "name"),
        ("qreg.local", lambda op: replace(op, attributes={"name": "x"}), "name"),
        (
            "qreg.local",
            lambda op: replace(op, attributes={"name": "tmp", "zero": True}),
            "operation",
        ),
        (
            "qreg.release",
            lambda op: replace(op, attributes={"zero": True}),
            "operation",
        ),
        (
            "qreg.release",
            lambda op: replace(op, regions=(Region((Block(),)),)),
            "operation",
        ),
        ("qreg.release", lambda op: replace(op, operands=op.operands[:1]), "signature"),
        (
            "qreg.release",
            lambda op: replace(op, operands=(*op.operands, op.operands[1])),
            "release",
        ),
        ("qreg.release", lambda op: replace(op, results=()), "effect"),
        (
            "qreg.release",
            lambda op: replace(
                op,
                operands=(
                    Value(ValueId("local_composition", 0), QUANTUM_EFFECT),
                    *op.operands[1:],
                ),
            ),
            "effect",
        ),
    ],
)
def test_local_operations_keep_closed_shapes_and_effects(
    opcode: str,
    change: Callable[[Operation], Operation],
    code: str,
) -> None:
    model, functions = local_composition()
    block = model.body.blocks[0]
    changed = False
    operations = []
    for op in block.operations:
        if op.name == opcode and not changed:
            op = change(op)
            changed = True
        operations.append(op)
    malformed = replace(
        model, body=Region((replace(block, operations=tuple(operations)),))
    )
    with pytest.raises(HybridVerificationError, match=f"register.{code}"):
        verify_register_program(malformed, functions)


def test_const_borrows_cannot_be_released_even_in_unused_helpers() -> None:
    p = Registers()
    x = p.declare("x", 1)
    p.emit("qreg.allocate", x)
    e = Value(ValueId("bad", 0), QUANTUM_EFFECT)
    after = Value(ValueId("bad", 1), QUANTUM_EFFECT)
    helper = HybridProgram(
        "bad",
        Region(
            (
                Block(
                    (e, x),
                    (
                        Operation("qreg.release", (e, x), (after,)),
                        Operation("qreg.return", (after, x)),
                    ),
                ),
            )
        ),
    )
    with pytest.raises(HybridVerificationError, match="register.const"):
        verify_register_program(p.finish(), (helper,))


def test_helper_local_escape_is_checked_independently_even_when_unused() -> None:
    model, functions = local_composition(helpers=True)
    helper = next(f for f in functions if f.name == "mark_less")
    block = helper.body.blocks[0]
    release = next(op for op in block.operations if op.name == "qreg.release")
    body = tuple(
        (
            replace(op, operands=(release.operands[0], *op.operands[1:]))
            if op.name == "qreg.return"
            else op
        )
        for op in block.operations
        if op.name != "qreg.release"
    )
    bad = replace(
        helper, name="unused_bad", body=Region((replace(block, operations=body),))
    )
    with pytest.raises(HybridVerificationError, match="register.local"):
        verify_register_program(model, (*functions, bad))


def test_expansion_clones_local_identities_and_avoids_root_name_collisions() -> None:
    _, functions = local_composition()
    helper = local_marker("mark_less", "less")
    p = Registers()
    x, y = p.declare("x", 2), p.declare("y", 2)
    for value in (x, y):
        p.emit("qreg.allocate", value)
    collision = Value(ValueId("register_probe.calls", 0), register_type(1))
    p.operations.append(
        Operation(
            "qreg.local",
            results=(collision,),
            attributes={"name": "register_probe.calls.local1"},
        )
    )
    p.emit("qreg.release", collision)
    for _ in range(3):
        p.emit("qreg.call", x, y, callee=helper.name)
    model = p.finish()
    probe = expand_register_calls(model, (*functions, helper))
    wanted = [op for op in probe.body.blocks[0].operations if op.name == "qreg.local"][
        1
    ].attributes["name"]
    block = model.body.blocks[0]
    operations = tuple(
        replace(op, attributes={"name": wanted}) if op.results == (collision,) else op
        for op in block.operations
    )
    model = replace(model, body=Region((replace(block, operations=operations),)))
    verify_register_program(model, (*functions, helper))
    flat = expand_register_calls(model, (*functions, helper))
    locals_ = [op for op in flat.body.blocks[0].operations if op.name == "qreg.local"]
    assert len({op.results[0].id for op in locals_}) == 4
    assert len({op.attributes["name"] for op in locals_}) == 4
    assert locals_[0].attributes["name"] == wanted
    assert locals_[1].attributes["name"] != wanted
    assert len({register_ports(flat)[1][op.results[0].id] for op in locals_}) == 1


def test_fresh_local_is_not_an_initialized_const_actual() -> None:
    _, functions = local_composition()
    helper = local_marker("mark_less", "less")
    p = Registers()
    x = p.declare("x", 2)
    tmp = p.declare("tmp", 2, local=True)
    p.emit("qreg.allocate", x)
    p.emit("qreg.call", x, tmp, callee=helper.name)
    p.emit("qreg.release", tmp)
    with pytest.raises(HybridVerificationError, match="register.uninitialized"):
        verify_register_program(p.finish(), (*functions, helper))
