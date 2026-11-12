"""Hidden local storage preserves coherent outputs, phases and actual budgets."""

from dataclasses import replace
from typing import Literal

import pytest
import torch

import flagquantum as fq
from flagquantum import _api, synth
from flagquantum.compiler._construction.constraints import CircuitLimits
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import Schedule
from flagquantum.compiler._construction.register_lowering import (
    schedule_register_program,
)
from flagquantum.compiler._construction.selection import (
    Constraints,
    ConstructionBudget,
    select_register_program,
)
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
from flagquantum.ecosystem import InteropConversionError
from flagquantum.errors import PlanningError
from flagquantum.services._synthesis_registers import synthesize_registers
from flagquantum.services.synthesis import SynthesisError
from tests.integration.test_construction_register_execution import _check
from tests.register_program_cases import Registers
from tests.synthesis_workflow_helpers import check_state
from tools.synthesis_compiler_examples.local_registers.main import (
    local_composition,
)
from tools.synthesis_compiler_examples.register_cleanup.main import (
    copy_reference,
    flag_value,
    less,
)

pytestmark = pytest.mark.integration
MEMORY = 64 * 1024**2


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize("overlapping", [False, True])
@pytest.mark.parametrize("x", range(4))
@pytest.mark.parametrize("y", range(4))
def test_all_basis_inputs_clear_locals_even_across_later_local_allocation(
    strategy: Schedule,
    overlapping: bool,
    x: int,
    y: int,
) -> None:
    model, functions = local_composition(basis=(x, y), overlapping=overlapping)
    selection = select_register_program(model, functions)
    row = next(r for r in selection.rows if r.strategy == strategy)
    _check(replace(selection, selected=row), {(x, y, x): (-1) ** int(x <= y)})


@pytest.mark.parametrize("producer", ["reused", "overlapping", "helpers", "direct"])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_replacement_preserves_the_public_program_and_named_consumer(
    producer: str,
    objective: Literal["width", "depth"],
) -> None:
    model, functions = local_composition(
        overlapping=producer == "overlapping",
        helpers=producer == "helpers",
        direct=producer == "direct",
    )
    program = synthesize_registers(
        model, functions, objective=objective, memory_limit_bytes=MEMORY
    )
    result = fq.run(program)
    check_state(
        program,
        result,
        {
            (x, y, x): 1j ** (x & 1) * (-1) ** int(x <= y) / 4
            for x in range(4)
            for y in range(4)
        },
    )
    assert tuple(program.registers) == ("x", "y", "reference")
    assert dict(program.report["register_widths"]) == {"x": 2, "y": 2, "reference": 2}
    assert dict(
        program.probabilities(result, registers=("x", "y", "reference"))
    ) == pytest.approx({(x, y, x): 1 / 16 for x in range(4) for y in range(4)})
    assert program._checked.verify_result(result) == program.report["resources"]


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize("overlapping", [False, True])
def test_raw_source_executes_selected_inverses_with_a_nontrivial_middle_phase(
    strategy: Schedule,
    overlapping: bool,
) -> None:
    model, functions = local_composition(overlapping=overlapping)
    row = schedule_register_program(model, functions, n_qubits=24, scheduling=strategy)
    work = row.peak_workspace
    raw = replace(row.circuit, n_wires=6 + work)
    observed = (
        fq.Circuit.from_ir(raw, dtype=torch.complex128, device="cpu")
        .state()
        .reshape(-1)
    )
    expected = torch.zeros_like(observed)
    for x in range(4):
        for y in range(4):
            expected[((x << 4) | (y << 2) | x) << work] = (
                1j ** (x & 1) * (-1) ** int(x <= y) / 4
            )
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)
    # The first inverse must stay valid while the later local is still live.
    first = row.segments[1]
    inverse = row.segments[4] if overlapping else row.segments[3]
    assert inverse.circuit.instructions == first.circuit.instructions[::-1]


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_fragmented_zero_slots_keep_the_numeric_significance_of_a_wide_local(
    strategy: Schedule,
) -> None:
    functions = (capture(copy_reference, {"x": 2}), capture(less, {"x": 2, "y": 2}))
    p = Registers()
    x, y = p.declare("x", 2), p.declare("y", 2)
    for value in (x, y):
        p.emit("qreg.allocate", value)
        for bit in range(2):
            p.emit("qreg.gate", value, gate="h", index=bit)
    p.emit("qreg.gate", x, gate="s", index=0)
    a, b, c = (p.declare(name, 1, local=True) for name in ("a", "b", "c"))
    p.emit("qreg.release", a)
    p.emit("qreg.release", c)
    word = p.declare("word", 2, local=True)
    p.emit("qreg.compute", word, x, callee="copy_reference")
    p.emit("qreg.phase", word, y, callee="less")
    p.emit("qreg.uncompute", word)
    p.emit("qreg.release", word)
    p.emit("qreg.release", b)
    selection = select_register_program(p.finish(), functions)
    row = next(r for r in selection.rows if r.strategy == strategy)
    _check(
        replace(selection, selected=row),
        {
            (x, y): 1j ** (x & 1) * (-1) ** int(x < y) / 4
            for x in range(4)
            for y in range(4)
        },
    )
    assert row.arithmetic.peak_workspace == 8


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_repeated_nested_helpers_keep_local_state_private_and_phase_exact(
    strategy: Schedule,
) -> None:
    _, functions = local_composition(helpers=True)
    entry = Value(ValueId("pair", 0), QUANTUM_EFFECT)
    x, y = (
        Value(ValueId("pair", i), functions[-1].body.blocks[0].arguments[1].type)
        for i in (1, 2)
    )
    first = Value(ValueId("pair", 3), QUANTUM_EFFECT)
    second = Value(ValueId("pair", 4), QUANTUM_EFFECT)
    pair = HybridProgram(
        "pair",
        Region(
            (
                Block(
                    (entry, x, y),
                    (
                        Operation(
                            "qreg.call",
                            (entry, x, y),
                            (first,),
                            {"callee": "mark_less"},
                        ),
                        Operation(
                            "qreg.call",
                            (first, x, y),
                            (second,),
                            {"callee": "mark_equal"},
                        ),
                        Operation("qreg.return", (second, x, y)),
                    ),
                ),
            )
        ),
    )
    p = Registers()
    x, y, reference = (p.declare(name, 2) for name in ("x", "y", "reference"))
    for value in (x, y):
        p.emit("qreg.allocate", value)
        for bit in range(2):
            p.emit("qreg.gate", value, gate="h", index=bit)
    p.emit("qreg.gate", x, gate="s", index=0)
    p.emit("qreg.compute", reference, x, callee="copy_reference")
    for _ in range(3):
        p.emit("qreg.call", x, y, callee="pair")
    selection = select_register_program(p.finish(), (*functions, pair))
    row = next(r for r in selection.rows if r.strategy == strategy)
    _check(
        replace(selection, selected=row),
        {
            (x, y, x): 1j ** (x & 1) * (-1) ** int(x <= y) / 4
            for x in range(4)
            for y in range(4)
        },
    )
    assert row.resources.n_qubits == 12 and row.arithmetic.peak_workspace == 6


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_local_release_preserves_observable_phase_interference(
    strategy: Schedule,
) -> None:
    graph = capture(flag_value, {"flag": 1})
    p = Registers()
    x, tmp = p.declare("x", 1), p.declare("tmp", 1, local=True)
    p.emit("qreg.allocate", x)
    p.emit("qreg.gate", x, gate="h", index=0)
    p.emit("qreg.compute", tmp, x, callee=graph.name)
    p.emit("qreg.phase", tmp, callee=graph.name)
    p.emit("qreg.uncompute", tmp)
    p.emit("qreg.release", tmp)
    p.emit("qreg.gate", x, gate="h", index=0)
    selection = select_register_program(p.finish(), (graph,))
    row = next(r for r in selection.rows if r.strategy == strategy)
    _check(replace(selection, selected=row), {(1,): 1})


def test_hidden_locals_count_toward_whole_qubit_and_workspace_constraints() -> None:
    constraints = Constraints(CircuitLimits(max_qubits=12), max_workspace=6)
    for overlapping in (False, True):
        model, functions = local_composition(overlapping=overlapping)
        if overlapping:
            with pytest.raises(SynthesisError) as error:
                synthesize_registers(
                    model, functions, constraints=constraints, memory_limit_bytes=MEMORY
                )
            assert error.value.selection.complete
            assert all(
                row.arithmetic.peak_workspace == 7 for row in error.value.selection.rows
            )
        else:
            program = synthesize_registers(
                model, functions, constraints=constraints, memory_limit_bytes=MEMORY
            )
            assert program.report["workspace_qubits"] == 6
            assert program.report["resources"].n_qubits == 12
    with pytest.raises(SynthesisError):
        synthesize_registers(
            model,
            functions,
            budget=ConstructionBudget(qubits=7),
            memory_limit_bytes=MEMORY,
        )


def test_existing_no_execution_memory_and_exact_target_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, functions = local_composition(helpers=True)
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


def test_aliased_const_actuals_stay_one_register_across_local_helpers() -> None:
    _, functions = local_composition(helpers=True)
    p = Registers()
    x = p.declare("x", 2)
    p.emit("qreg.allocate", x)
    for bit in range(2):
        p.emit("qreg.gate", x, gate="h", index=bit)
    p.emit("qreg.gate", x, gate="s", index=0)
    for helper in ("mark_less", "mark_equal"):
        p.emit("qreg.call", x, x, callee=helper)
    program = synthesize_registers(p.finish(), functions, memory_limit_bytes=MEMORY)
    check_state(
        program, fq.run(program), {(x,): -(1j ** (x & 1)) / 2 for x in range(4)}
    )


@pytest.mark.parametrize(
    "statement", ["tmp:qbit; allocate(tmp);", "release(q);", "free(q);", "drop(q);"]
)
def test_manual_local_lifecycle_stays_outside_native_source(statement: str) -> None:
    with pytest.raises(InteropConversionError):
        synth.synthesize(
            f"qfunc main(output q:qbit) {{ allocate(q); {statement} }}",
            source_format="qmod",
        )
