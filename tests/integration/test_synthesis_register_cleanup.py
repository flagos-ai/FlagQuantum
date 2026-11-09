"""Exact cleanup preserves phases and correlations through existing execution."""

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.composition.register_cleanup.main import (
    cleanup_composition,
    flag_value,
    less,
)
from flagquantum import _api, synth
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import Schedule
from flagquantum.compiler._construction.selection import select_register_program
from flagquantum.compiler._construction.target import Target
from flagquantum.ecosystem import InteropConversionError
from flagquantum.errors import PlanningError
from flagquantum.services._synthesis_registers import synthesize_registers
from tests.integration.test_construction_register_execution import _check
from tests.integration.test_synthesis_register_workflow import check_state
from tests.register_program_cases import Registers, pressure

pytestmark = pytest.mark.integration
MEMORY = 64 * 1024**2


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize("x", range(4))
@pytest.mark.parametrize("y", range(4))
def test_all_sixteen_basis_inputs_reuse_flag_and_keep_the_absolute_sign(
    strategy: Schedule, x: int, y: int
) -> None:
    model, functions = cleanup_composition(basis=(x, y))
    selection = select_register_program(model, functions)
    row = next(r for r in selection.rows if r.strategy == strategy)
    _check(replace(selection, selected=row), {(x, y, 0, x): (-1) ** int(x <= y)})


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize("direct", [False, True])
def test_both_phase_producers_preserve_complex_external_entanglement(
    strategy: Schedule, direct: bool
) -> None:
    model, functions = cleanup_composition(direct=direct)
    selection = select_register_program(model, functions)
    row = next(r for r in selection.rows if r.strategy == strategy)
    _check(
        replace(selection, selected=row),
        {
            (x, y, 0, x): 1j ** (x & 1) * (-1) ** int(x <= y) / 4
            for x in range(4)
            for y in range(4)
        },
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("direct", [False, True])
def test_producer_replacement_keeps_the_same_program_run_and_named_consumer(
    objective: Literal["width", "depth"], direct: bool
) -> None:
    model, functions = cleanup_composition(direct=direct)
    program = synthesize_registers(
        model, functions, objective=objective, memory_limit_bytes=MEMORY
    )
    result = fq.run(program)
    check_state(
        program,
        result,
        {
            (x, y, 0, x): 1j ** (x & 1) * (-1) ** int(x <= y) / 4
            for x in range(4)
            for y in range(4)
        },
    )
    assert dict(
        program.probabilities(result, registers=("x", "y", "flag", "reference"))
    ) == pytest.approx({(x, y, 0, x): 1 / 16 for x in range(4) for y in range(4)})
    assert program._checked.verify_result(result) == program.report["resources"]
    assert tuple(program.registers["flag"]) == (4,)


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_cleanup_phase_is_observable_by_interference(strategy: Schedule) -> None:
    graph = capture(flag_value, {"flag": 1})
    p = Registers()
    x, flag = p.declare("x", 1), p.declare("flag", 1)
    p.emit("qreg.allocate", x)
    p.emit("qreg.gate", x, gate="h", index=0)
    p.emit("qreg.compute", flag, x, callee=graph.name)
    p.emit("qreg.phase", flag, callee=graph.name)
    p.emit("qreg.uncompute", flag)
    p.emit("qreg.allocate", flag)
    p.emit("qreg.gate", x, gate="h", index=0)
    selection = select_register_program(p.finish(), (graph,))
    row = next(r for r in selection.rows if r.strategy == strategy)
    _check(replace(selection, selected=row), {(1, 0): 1})


def is_two(out: int) -> bool:
    return out == 2


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_rematerialized_result_cleanup_uses_the_selected_physical_history(
    strategy: Schedule,
) -> None:
    graph = capture(pressure, {"x": 1, "y": 1})
    marker = capture(is_two, {"out": 2})
    p = Registers()
    x, y, out = p.declare("x", 1), p.declare("y", 1), p.declare("out", 2)
    for value in (x, y):
        p.emit("qreg.allocate", value)
        p.emit("qreg.gate", value, gate="h", index=0)
    p.emit("qreg.gate", x, gate="s", index=0)
    p.emit("qreg.compute", out, x, y, callee=graph.name)
    p.emit("qreg.phase", out, callee=marker.name)
    p.emit("qreg.uncompute", out)
    p.emit("qreg.allocate", out)
    selection = select_register_program(p.finish(), (graph, marker))
    row = next(r for r in selection.rows if r.strategy == strategy)
    _check(
        replace(selection, selected=row),
        {(a, b, 0): 1j**a * (-1) ** (a * b) / 2 for a in range(2) for b in range(2)},
    )
    # Execute the selected source instructions as well as the checked target;
    # a nontrivial middle phase prevents U followed by its inverse cancelling.
    assert row.arithmetic is not None
    work = row.arithmetic.peak_workspace
    raw = replace(row.arithmetic.circuit, n_wires=4 + work)
    observed = (
        fq.Circuit.from_ir(raw, dtype=torch.complex128, device="cpu")
        .state()
        .reshape(-1)
    )
    expected = torch.zeros_like(observed)
    for a in range(2):
        for b in range(2):
            expected[((a << 3) | (b << 2)) << work] = 1j**a * (-1) ** (a * b) / 2
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)
    assert (
        selection.rows[0].arithmetic.recomputations
        != selection.rows[1].arithmetic.recomputations
    )


def three(x: int) -> int:
    return x * 0 + 3


def zero(x: int) -> int:
    return x * 0


@pytest.mark.parametrize("function", [three, zero])
def test_fixed_results_and_zero_valued_cleanup_keep_widened_storage(
    function: Callable[[int], int],
) -> None:
    graph = capture(function, {"x": 1})
    p = Registers()
    x = p.declare("x", 1)
    out = p.declare("out", 3)
    p.emit("qreg.allocate", x)
    p.emit("qreg.gate", x, gate="h", index=0)
    p.emit("qreg.compute", out, x, callee=graph.name)
    p.emit("qreg.uncompute", out)
    p.emit("qreg.initialize", out, value=5)
    program = synthesize_registers(p.finish(), (graph,), memory_limit_bytes=MEMORY)
    check_state(program, fq.run(program), {(x, 5): 1 / math.sqrt(2) for x in range(2)})
    assert dict(program.report["register_widths"]) == {"x": 1, "out": 3}


def test_alias_inputs_and_caller_mutation_after_cleanup_are_valid() -> None:
    graph = capture(less, {"x": 2, "y": 2})
    p = Registers()
    x, flag = p.declare("x", 2), p.declare("flag", 2)
    p.emit("qreg.allocate", x)
    for bit in range(2):
        p.emit("qreg.gate", x, gate="h", index=bit)
    p.emit("qreg.compute", flag, x, x, callee=graph.name)
    p.emit("qreg.uncompute", flag)
    p.emit("qreg.initialize", flag, value=2)
    p.emit("qreg.gate", x, gate="x", index=0)
    program = synthesize_registers(p.finish(), (graph,), memory_limit_bytes=MEMORY)
    check_state(program, fq.run(program), {(x, 2): 0.5 for x in range(4)})


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_live_derived_output_survives_cleaning_its_computed_source(
    objective: Literal["width", "depth"],
) -> None:
    functions = (capture(less, {"x": 2, "y": 2}), capture(flag_value, {"flag": 1}))
    p = Registers()
    x, y = p.declare("x", 2), p.declare("y", 2)
    flag, out = p.declare("flag", 1), p.declare("result", 1)
    for value in (x, y):
        p.emit("qreg.allocate", value)
        for bit in range(2):
            p.emit("qreg.gate", value, gate="h", index=bit)
    p.emit("qreg.compute", flag, x, y, callee="less")
    p.emit("qreg.compute", out, flag, callee="flag_value")
    p.emit("qreg.uncompute", flag)
    p.emit("qreg.allocate", flag)
    program = synthesize_registers(
        p.finish(), functions, objective=objective, memory_limit_bytes=MEMORY
    )
    check_state(
        program,
        fq.run(program),
        {(x, y, 0, int(x < y)): 0.25 for x in range(4) for y in range(4)},
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_nested_helper_cleanup_expands_through_the_existing_call_boundary(
    objective: Literal["width", "depth"],
) -> None:
    helper, functions = cleanup_composition()
    p = Registers()
    actuals = tuple(
        p.declare(name, width)
        for name, width in (("x", 2), ("y", 2), ("flag", 1), ("reference", 2))
    )
    p.emit("qreg.call", *actuals, callee=helper.name)
    program = synthesize_registers(
        p.finish(), (*functions, helper), objective=objective, memory_limit_bytes=MEMORY
    )
    check_state(
        program,
        fq.run(program),
        {
            (x, y, 0, x): 1j ** (x & 1) * (-1) ** int(x <= y) / 4
            for x in range(4)
            for y in range(4)
        },
    )


def test_existing_planning_memory_and_exact_phase_target_guards_remain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, functions = cleanup_composition()
    monkeypatch.setattr(_api, "run", lambda *a, **k: pytest.fail("synthesis executed"))
    assert (
        synthesize_registers(model, functions, memory_limit_bytes=MEMORY).execution
        is not None
    )
    with pytest.raises(PlanningError, match="memory budget"):
        synthesize_registers(model, functions, memory_limit_bytes=1)
    with pytest.raises(ValueError, match="phase output admits only"):
        synthesize_registers(
            model,
            functions,
            memory_limit_bytes=MEMORY,
            target=Target(basis=("x", "cx")),
        )


def test_native_uncompute_syntax_is_not_accidentally_admitted() -> None:
    with pytest.raises(InteropConversionError):
        synth.synthesize(
            "qfunc main(output q:qbit) { allocate(q); uncompute(q); }",
            source_format="qmod",
        )
