"""Const/fresh-output calls execute through the unchanged checked consumer."""

from dataclasses import replace

import pytest

import flagquantum as fq
from flagquantum import _api, synth
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.register_profile import register_type
from flagquantum.compiler._construction.selection import ConstructionBudget
from flagquantum.compiler._construction.target import Target
from flagquantum.compiler._hybrid.model import (
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    Operation,
    Region,
    Value,
    ValueId,
)
from flagquantum.errors import PlanningError
from flagquantum.services._synthesis_registers import synthesize_registers
from tests.integration.test_synthesis_register_workflow import check_state
from tests.register_program_cases import Registers
from tools.synthesis_compiler_examples.const_calls.main import const_composition

pytestmark = pytest.mark.integration


def identity(x: int) -> int:
    return x


def pressure(x: int, y: int) -> int:
    return (x * y) * (x * y)


def always(x: int, y: int) -> int:
    return x * 0 + y * 0 + 1


def never(x: int, y: int) -> int:
    return x * 0 + y * 0


def model(*, basis=None, calls=True, reference=False, alias=False):
    _, functions = const_composition()
    p = Registers()
    x, y, out = p.declare("x", 2), p.declare("y", 2), p.declare("energy", 4)
    ref = p.declare("reference", 2) if reference else None
    for i, reg in enumerate((x, y)):
        if basis is None:
            p.emit("qreg.allocate", reg)
            for bit in range(2):
                p.emit("qreg.gate", reg, gate="h", index=bit)
            if i == 0:
                p.emit("qreg.gate", reg, gate="s", index=0)
        else:
            p.emit("qreg.initialize", reg, value=basis[i])
    if ref is not None:
        graph = capture(identity, {"x": 2})
        functions = (*functions, graph)
        p.emit("qreg.compute", ref, x, callee=graph.name)
    borrowed = (x, x if alias else y)
    if calls:
        p.emit("qreg.call", *borrowed, out, callee="combined_call")
    else:
        p.emit("qreg.compute", out, *borrowed, callee="energy")
        p.emit("qreg.phase", *borrowed, callee="marked")
    p.emit("qreg.gate", x, gate="x", index=0)
    return p.finish(), functions


def execute(root, functions, **options):
    return synthesize_registers(
        root, functions, memory_limit_bytes=256 * 1024**2, **options
    )


@pytest.mark.parametrize("x", range(4))
@pytest.mark.parametrize("y", range(4))
def test_all_sixteen_basis_inputs_return_output_and_preserve_const_until_caller_mutates(
    x, y
):
    root, functions = model(basis=(x, y))
    program = execute(root, functions)
    check_state(program, fq.run(program), {(x ^ 1, y, x * x + y): (-1) ** (x < y)})


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("calls", [False, True])
@pytest.mark.parametrize("reference", [False, True])
def test_call_and_direct_producers_share_full_complex_state_consumer(
    objective, calls, reference
):
    root, functions = model(calls=calls, reference=reference)
    program = execute(root, functions, objective=objective)
    expected = {
        ((x ^ 1, y, x * x + y, x) if reference else (x ^ 1, y, x * x + y)): 1j
        ** (x & 1)
        * (-1) ** (x < y)
        / 4
        for x in range(4)
        for y in range(4)
    }
    result = fq.run(program)
    check_state(program, result, expected)
    assert dict(
        program.probabilities(result, registers=("y", "x", "energy"))
    ) == pytest.approx(
        {(y, x ^ 1, x * x + y): 1 / 16 for x in range(4) for y in range(4)}
    )
    assert (
        program.report["source_identities"]
        == program.selection.selected.source_identities
    )
    assert program.report["resources"] == program._checked.verify_result(result)


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_repeated_const_actual_aliases_borrow_one_encoding(objective):
    root, functions = model(alias=True)
    program = execute(root, functions, objective=objective)
    check_state(
        program,
        fq.run(program),
        {(x ^ 1, y, x * x + x): 1j ** (x & 1) / 4 for x in range(4) for y in range(4)},
    )


@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_nested_constant_phase_preserves_absolute_sign(value, objective):
    root, functions = model()
    graph = capture(always if value else never, {"x": 2, "y": 2})
    helper = next(f for f in functions if f.name == "phase_call")
    block = helper.body.blocks[0]
    helper = replace(
        helper,
        body=Region(
            (
                replace(
                    block,
                    operations=tuple(
                        (
                            replace(op, attributes={"callee": graph.name})
                            if op.name == "qreg.phase"
                            else op
                        )
                        for op in block.operations
                    ),
                ),
            )
        ),
    )
    functions = (*[f for f in functions if f.name != "phase_call"], graph, helper)
    program = execute(root, functions, objective=objective)
    check_state(
        program,
        fq.run(program),
        {
            (x ^ 1, y, x * x + y): (-1) ** value * 1j ** (x & 1) / 4
            for x in range(4)
            for y in range(4)
        },
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_repeated_calls_reuse_clean_internal_scratch(objective):
    graph = capture(pressure, {"x": 1, "y": 1})
    e = Value(ValueId("repeat", 0), QUANTUM_EFFECT)
    x, y, out = (Value(ValueId("repeat", i), register_type(1)) for i in range(1, 4))
    after = Value(ValueId("repeat", 4), QUANTUM_EFFECT)
    helper = HybridProgram(
        "repeat",
        Region(
            (
                Block(
                    (e, x, y),
                    (
                        Operation(
                            "qreg.declare", results=(out,), attributes={"name": "out"}
                        ),
                        Operation(
                            "qreg.compute",
                            (e, out, x, y),
                            (after,),
                            {"callee": graph.name},
                        ),
                        Operation("qreg.return", (after, x, y, out)),
                    ),
                ),
            )
        ),
    )
    p = Registers()
    a, b = p.declare("a", 1), p.declare("b", 1)
    first, second = p.declare("first", 1), p.declare("second", 1)
    for reg in (a, b):
        p.emit("qreg.allocate", reg)
        p.emit("qreg.gate", reg, gate="h", index=0)
    p.emit("qreg.gate", a, gate="s", index=0)
    p.emit("qreg.call", a, b, first, callee="repeat")
    p.emit("qreg.call", a, b, second, callee="repeat")
    program = execute(p.finish(), (graph, helper), objective=objective)
    check_state(
        program,
        fq.run(program),
        {(a, b, a * b, a * b): 1j**a / 2 for a in range(2) for b in range(2)},
    )
    row = program.selection.selected.arithmetic
    assert row.peak_workspace > 0
    assert row.peak_workspace == max(s.peak_workspace for s in row.segments)
    assert row.peak_workspace < sum(s.peak_workspace for s in row.segments)


def test_nested_phase_cannot_bypass_target_guard_and_unused_phase_does_not_restrict_value(
    monkeypatch,
):
    root, functions = model()
    target = Target(basis=("x", "h", "cx", "ccx", "u3"))
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("unsupported phase target planned")
    )
    with pytest.raises(ValueError, match="phase output admits only"):
        execute(root, functions, target=target)
    p = Registers()
    x, y, out = p.declare("x", 2), p.declare("y", 2), p.declare("out", 4)
    p.emit("qreg.allocate", x)
    p.emit("qreg.allocate", y)
    p.emit("qreg.call", x, y, out, callee="value_call")
    from flagquantum.compiler._construction.selection import select_register_program

    assert select_register_program(
        p.finish(), functions, target=target
    ).selected.feasible


def test_call_budget_refuses_before_planning(monkeypatch):
    root, functions = model()
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("capacity exhausted model planned")
    )
    with pytest.raises(synth.SynthesisError):
        execute(root, functions, budget=ConstructionBudget(qubits=7))


def test_const_call_synthesis_never_executes_and_memory_uses_existing_plan_guard(
    monkeypatch,
):
    root, functions = model()
    monkeypatch.setattr(_api, "run", lambda *a, **k: pytest.fail("synthesis executed"))
    assert execute(root, functions).execution is not None
    with pytest.raises(PlanningError, match="memory budget"):
        synthesize_registers(root, functions, memory_limit_bytes=1)
