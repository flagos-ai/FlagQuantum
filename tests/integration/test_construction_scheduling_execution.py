"""Both workspace schedules on exact CPU amplitudes and controlled phase paths."""

import cmath
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import compile as compile_program
from flagquantum.compiler._construction.frontend_python import capture
from flagquantum.compiler._construction.lowering import (
    ArithmeticCandidate,
    Schedule,
    schedule_arithmetic,
)

pytestmark = pytest.mark.integration


def test_workspace_example_runs_the_fixed_cost_table_and_both_cpu_paths() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "examples.synthesis_workspace"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert "1 retain 6 10 40 35 0" in result.stdout
    assert "1 recompute 5 9 58 54 3" in result.stdout
    assert "2 retain 28 39 739 545 0" in result.stdout
    assert "2 recompute 23 34 1091 860 3" in result.stdout
    assert "3 retain 47 65 2714 2035 0" in result.stdout
    assert "3 recompute 39 57 4010 3115 3" in result.stdout
    assert "both one-bit schedules passed" in result.stdout


def pressure(x: int, y: int) -> int:
    t = x * y
    a = t * x
    u = a * y
    return t + u


@pytest.fixture(params=("retain", "recompute"))
def candidate(request: pytest.FixtureRequest) -> ArithmeticCandidate:
    strategy: Schedule = request.param
    graph = capture(pressure, {"x": 1, "y": 1})
    bindings = {v.id: (i,) for i, v in enumerate(graph.body.blocks[0].arguments)}
    options = dict(entry=graph.name, workspace_state="fresh_zero", scheduling=strategy)
    probe = schedule_arithmetic(
        (graph,),
        bindings,
        (2, 3),
        n_qubits=32,
        workspace=tuple(range(4, 32)),
        **options,
    )
    n = 4 + probe.peak_workspace
    return schedule_arithmetic(
        (graph,), bindings, (2, 3), n_qubits=n, workspace=tuple(range(4, n)), **options
    )


def _options() -> fq.ExecutionOptions:
    return fq.ExecutionOptions(
        mode="statevector",
        backend="torch",
        device="cpu",
        precision="complex128",
        require_gradients=False,
        allow_approximate=False,
        allow_backend_fallback=False,
    )


def test_all_basis_inputs_and_every_xor_target_through_existing_run(
    candidate: ArithmeticCandidate,
) -> None:
    program = candidate.circuit
    work = candidate.peak_workspace
    for x in range(2):
        for y in range(2):
            for target in range(4):
                initial = (((x << 1) | y) << 2 | target) << work
                expected = (((x << 1) | y) << 2 | (target ^ pressure(x, y))) << work
                circuit = fq.Circuit(program.n_wires, dtype=torch.complex128)
                for q in range(program.n_wires):
                    if initial & (1 << (program.n_wires - q - 1)):
                        circuit.x(q)
                observed = (
                    fq.run(circuit.compose(program), options=_options())
                    .statevector()
                    .reshape(-1)
                )
                reference = torch.zeros_like(observed)
                reference[expected] = 1
                torch.testing.assert_close(observed, reference, atol=1e-12, rtol=0)


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_complex_reference_entanglement_and_controlled_relative_phase(
    candidate: ArithmeticCandidate,
    controlled: bool,
    compiled: bool,
) -> None:
    original = candidate.circuit
    n = original.n_wires
    if controlled:
        circuit = fq.Circuit.from_ir(original, dtype=torch.complex128).control(
            1, ctrl_qubits=(n,)
        )
    else:
        circuit = fq.Circuit(n + 1, dtype=torch.complex128).compose(original)
    program = compile_program(circuit.to_ir()) if compiled else circuit.to_ir()
    amplitudes = torch.randn(
        32, dtype=torch.complex128, generator=torch.Generator().manual_seed(505)
    )
    amplitudes /= torch.linalg.vector_norm(amplitudes)
    initial = torch.zeros(1 << (n + 1), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    row = 0
    for x in range(2):
        for y in range(2):
            for target in range(4):
                for control in range(2):
                    prefix = (x << 1) | y
                    source = (
                        ((prefix << 2) | target) << (candidate.peak_workspace + 1)
                    ) | control
                    value = pressure(x, y) if not controlled or control else 0
                    destination = (
                        ((prefix << 2) | (target ^ value))
                        << (candidate.peak_workspace + 1)
                    ) | control
                    initial[source] = amplitudes[row]
                    expected[destination] = amplitudes[row]
                    row += 1
    observed = (
        fq.Circuit.from_ir(program, inputs=initial, device="cpu").state().reshape(-1)
    )
    # No quotient by a global phase: the control-zero branch supplies an external
    # phase reference, and all nonzero workspace amplitudes must also vanish.
    torch.testing.assert_close(observed, expected, atol=1e-10, rtol=0)
    dirty = observed.reshape(16, 1 << candidate.peak_workspace, 2)[:, 1:, :]
    assert float(dirty.abs().square().sum()) <= 1e-12


def test_same_generated_block_compute_mark_uncompute_has_exact_phase(
    candidate: ArithmeticCandidate,
) -> None:
    program = candidate.circuit
    block = fq.Circuit.from_ir(program, dtype=torch.complex128)
    circuit = fq.Circuit(program.n_wires, dtype=torch.complex128)
    circuit.h(0).h(1).s(0).t(1).compose(block).z(2).compose(block.adjoint())
    observed = fq.run(circuit, options=_options()).statevector().reshape(-1)
    expected = torch.zeros_like(observed)
    for x in range(2):
        for y in range(2):
            sign = (-1) ** ((pressure(x, y) >> 1) & 1)
            expected[((x << 1) | y) << (program.n_wires - 2)] = (
                sign * cmath.exp(1j * (x * math.pi / 2 + y * math.pi / 4)) / 2
            )
    torch.testing.assert_close(observed, expected, atol=1e-12, rtol=0)
    assert float(observed.reshape(4, -1)[:, 1:].abs().square().sum()) <= 1e-12
