"""Ordered basis-constant phase borrows through existing exact CPU workflows."""

import ast
import math
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis_register_phase import phase_composition
from flagquantum import _api
from flagquantum.compiler._construction.constraints import CircuitLimits
from flagquantum.compiler._construction.frontend_python import Definition, capture
from flagquantum.compiler._construction.lowering import Schedule
from flagquantum.compiler._construction.register_lowering import (
    RegisterCandidate,
    schedule_register_program,
)
from flagquantum.compiler._construction.selection import Constraints, ConstructionBudget
from flagquantum.compiler._construction.target import PHASE_BASIS, Target
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.compiler.topology.coupling import CouplingMap
from flagquantum.errors import PlanningError
from flagquantum.services._synthesis_registers import synthesize_registers
from flagquantum.services.synthesis import SynthesisError
from tests.integration.test_synthesis_register_workflow import check_state
from tests.register_program_cases import Registers

pytestmark = pytest.mark.integration
MEMORY = 256 * 1024**2


def predicate(x: int, y: int) -> bool:
    return x < y or x == 2


def same(a: int, b: int) -> bool:
    return a == b


def different(a: int, b: int) -> bool:
    return a != b


def both(a: int, b: int) -> int:
    return a * b


def borrowed_segment(strategy: Schedule) -> RegisterCandidate:
    graph = capture(predicate, {"x": 2, "y": 1})
    p = Registers()
    x, y = p.declare("x", 2), p.declare("y", 1)
    p.emit("qreg.allocate", x)
    p.emit("qreg.allocate", y)
    p.emit("qreg.phase", x, y, callee=graph.name)
    return schedule_register_program(
        p.finish(), (graph,), n_qubits=24, scheduling=strategy
    )


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_borrowed_phase_segment_all_eight_basis_values_keep_data_and_clear_workspace(
    strategy: Schedule,
) -> None:
    # Numerical proof of the borrowed segment, not an external-state claim for
    # the enclosing fresh-allocation program, whose entry requires zero storage.
    row = borrowed_segment(strategy)
    segment = row.segments[0]
    n = segment.circuit.n_wires
    assert n <= 20
    for x in range(4):
        for y in range(2):
            index = ((x << 1) | y) << segment.peak_workspace
            initial = torch.zeros(1 << n, dtype=torch.complex128)
            initial[index] = 1
            expected = initial * ((-1) ** int(x < y or x == 2))
            observed = (
                fq.Circuit.from_ir(
                    segment.circuit,
                    inputs=initial,
                    dtype=torch.complex128,
                    device="cpu",
                )
                .state()
                .reshape(-1)
            )
            torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
def test_segment_preserves_arbitrary_complex_data_entangled_with_external_reference(
    strategy: Schedule,
) -> None:
    segment = borrowed_segment(strategy).segments[0]
    n = segment.circuit.n_wires
    amplitudes = torch.randn(
        16, dtype=torch.complex128, generator=torch.Generator().manual_seed(984)
    )
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << (n + 1), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for prefix in range(8):
        x, y = prefix >> 1, prefix & 1
        for reference in range(2):
            index = (prefix << (segment.peak_workspace + 1)) | reference
            initial[index] = amplitudes[2 * prefix + reference]
            expected[index] = initial[index] * ((-1) ** int(x < y or x == 2))
    circuit = fq.Circuit(n + 1, dtype=torch.complex128).compose(
        segment.circuit, qubits=range(n)
    )
    observed = (
        fq.Circuit.from_ir(
            circuit.to_ir(), inputs=initial, dtype=torch.complex128, device="cpu"
        )
        .state()
        .reshape(-1)
    )
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)
    assert (
        float(
            observed.reshape(8, 1 << segment.peak_workspace, 2)[:, 1:, :]
            .abs()
            .square()
            .sum()
        )
        <= 1e-12
    )


@pytest.mark.parametrize("strategy", ["retain", "recompute"])
@pytest.mark.parametrize("function", [same, different, both])
def test_repeated_aliases_preserve_true_false_and_data_dependent_absolute_phase(
    strategy: Schedule, function: Definition
) -> None:
    graph = capture(function, {"a": 1, "b": 1})
    p = Registers()
    x = p.declare("x", 1)
    p.emit("qreg.allocate", x)
    p.emit("qreg.phase", x, x, callee=graph.name)
    row = schedule_register_program(
        p.finish(), (graph,), n_qubits=12, scheduling=strategy
    )
    segment = row.segments[0]
    n = segment.circuit.n_wires
    amplitudes = torch.tensor(
        [math.sqrt(0.3) * 1j, math.sqrt(0.7)], dtype=torch.complex128
    )
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for xvalue in range(2):
        index = xvalue << segment.peak_workspace
        initial[index] = amplitudes[xvalue]
        flag = 1 if function is same else 0 if function is different else xvalue
        expected[index] = initial[index] * ((-1) ** flag)
    observed = (
        fq.Circuit.from_ir(
            segment.circuit, inputs=initial, dtype=torch.complex128, device="cpu"
        )
        .state()
        .reshape(-1)
    )
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)
    if function is both:
        assert segment.peak_workspace == 0 and n == 1


def reference_state() -> dict[tuple[int, ...], complex]:
    # Independent closed expression for the whole zero-entry program: S/H
    # preparation, a copied external reference, two phase actions around X and
    # product computation, then a final H causing real interference.
    expected: dict[tuple[int, ...], complex] = {}
    for a in range(2):
        for b in range(2):
            x = a ^ 1
            z = x * b
            first = int(a + b < 2)
            second = int(z + b < 2)
            for y in range(2):
                key = (x, y, z, a)
                amplitude = (
                    (1j if a else 1) * ((-1) ** (first + second + b * y)) / math.sqrt(8)
                )
                expected[key] = expected.get(key, 0j) + amplitude
    return {key: value for key, value in expected.items() if abs(value) > 1e-14}


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("profile", ["logical", "line_keep"])
def test_whole_program_calls_mutations_reference_and_workspace_reuse(
    objective: Literal["width", "depth"], profile: str
) -> None:
    model, functions = phase_composition()
    target = (
        Target()
        if profile == "logical"
        else Target(
            basis=PHASE_BASIS,
            optimize=False,
            restore_layout=False,
            couplings=tuple(CouplingMap.line(n) for n in range(4, 25)),
        )
    )
    program = synthesize_registers(
        model,
        functions,
        objective=objective,
        target=target,
        budget=ConstructionBudget(qubits=24),
        memory_limit_bytes=MEMORY,
    )
    result = fq.run(program)
    expected = reference_state()
    check_state(program, result, expected)
    assert dict(
        program.probabilities(result, registers=tuple(program.registers))
    ) == pytest.approx(
        {key: abs(value) ** 2 for key, value in expected.items()}, abs=1e-12
    )
    row = program.selection.selected
    assert row is not None and row.arithmetic is not None
    assert len(row.arithmetic.segments) == 4
    assert program.report["workspace_qubits"] == max(
        s.peak_workspace for s in row.arithmetic.segments
    )
    assert program.report["workspace_qubits"] < sum(
        s.peak_workspace for s in row.arithmetic.segments
    )
    assert result.plan.identity == program.execution.identity


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_direct_phase_alias_fits_zero_workspace_and_interferes_to_one(
    objective: Literal["width", "depth"],
) -> None:
    graph = capture(both, {"a": 1, "b": 1})
    p = Registers()
    x = p.declare("x", 1)
    p.emit("qreg.allocate", x)
    p.emit("qreg.gate", x, gate="h", index=0)
    p.emit("qreg.phase", x, x, callee=graph.name)
    p.emit("qreg.gate", x, gate="h", index=0)
    program = synthesize_registers(
        p.finish(),
        (graph,),
        objective=objective,
        constraints=Constraints(CircuitLimits(max_qubits=1), max_workspace=0),
        memory_limit_bytes=1024,
    )
    result = fq.run(program)
    check_state(program, result, {(1,): 1})
    assert program.report["workspace_qubits"] == 0
    assert program.report["resources"].n_qubits == 1
    assert dict(program.probabilities(result, registers=("x",))) == {(1,): 1.0}


def test_uninitialized_phase_reads_and_projective_targets_fail_before_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("planned invalid phase borrow")
    )
    graph = capture(both, {"a": 1, "b": 1})
    p = Registers()
    x = p.declare("x", 1)
    p.emit("qreg.phase", x, x, callee=graph.name)
    with pytest.raises(HybridVerificationError, match="uninitialized"):
        synthesize_registers(p.finish(), (graph,), memory_limit_bytes=MEMORY)
    model, functions = phase_composition()
    with pytest.raises(ValueError, match="phase"):
        synthesize_registers(
            model,
            functions,
            target=Target(basis=("x", "cx", "u3"), optimize=False),
            memory_limit_bytes=MEMORY,
        )


def test_constraints_and_memory_never_execute_an_invalid_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fq, "run", lambda *a, **k: pytest.fail("executed during synthesis")
    )
    model, functions = phase_composition()
    for constraints in [
        Constraints(max_workspace=0),
        Constraints(CircuitLimits(max_operations=1)),
        Constraints(CircuitLimits(max_qubits=4)),
    ]:
        with pytest.raises(SynthesisError):
            synthesize_registers(
                model, functions, constraints=constraints, memory_limit_bytes=MEMORY
            )
    with pytest.raises(PlanningError, match="memory"):
        synthesize_registers(model, functions, memory_limit_bytes=1)


def test_private_contributor_example_runs_through_existing_program_and_named_reading() -> (
    None
):
    completed = subprocess.run(
        [sys.executable, "-m", "examples.synthesis_register_phase"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    rows = dict(
        line.split(": ", 1) for line in completed.stdout.splitlines() if ": " in line
    )
    probabilities = {key: abs(value) ** 2 for key, value in reference_state().items()}
    for objective in ["width", "depth"]:
        assert ast.literal_eval(rows[f"{objective} probabilities"]) == pytest.approx(
            probabilities, abs=1e-12
        )


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_restored_interleaving_route_without_swap_witnesses_refuses_before_plan(
    objective: Literal["width", "depth"], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The existing normalizer cancels restore SWAPs in this large interleaving
    # fixture. Preserve the target's refusal rather than accepting lost evidence.
    monkeypatch.setattr(
        _api, "plan", lambda *a, **k: pytest.fail("planned unproven layout")
    )
    model, functions = phase_composition()
    target = Target(
        basis=PHASE_BASIS,
        optimize=False,
        couplings=tuple(CouplingMap.line(n) for n in range(4, 25)),
    )
    with pytest.raises(SynthesisError, match="lost routed SWAP evidence") as caught:
        synthesize_registers(
            model,
            functions,
            objective=objective,
            target=target,
            budget=ConstructionBudget(qubits=24),
            memory_limit_bytes=MEMORY,
        )
    assert caught.value.selection.selected is None
    assert all(
        row.diagnostics[0].code == "target.layout"
        for row in caught.value.selection.rows
    )


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("function", [same, different])
def test_checked_program_preserves_constant_alias_global_phase(
    objective: Literal["width", "depth"], function: Definition
) -> None:
    graph = capture(function, {"a": 1, "b": 1})
    p = Registers()
    x = p.declare("x", 1)
    p.emit("qreg.allocate", x)
    p.emit("qreg.gate", x, gate="h", index=0)
    p.emit("qreg.gate", x, gate="s", index=0)
    p.emit("qreg.phase", x, x, callee=graph.name)
    program = synthesize_registers(
        p.finish(), (graph,), objective=objective, memory_limit_bytes=MEMORY
    )
    sign = -1 if function is same else 1
    check_state(
        program,
        fq.run(program),
        {(0,): sign / math.sqrt(2), (1,): sign * 1j / math.sqrt(2)},
    )
