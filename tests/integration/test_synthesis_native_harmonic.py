"""Harmonic oscillator translation retains both branches and every input column."""

import ast
import re
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.quantum_algo_for_solving_linear_differential_equations.harmonic_oscillator.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.community.paper_implementation_project.quantum_algo_for_solving_linear_differential_equations.harmonic_oscillator.main import (
    AMPLITUDES,
    build_program,
    main,
)
from examples.synthesis.community.paper_implementation_project.quantum_algo_for_solving_linear_differential_equations.harmonic_oscillator.physics import (
    ENCODING,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "tests/fixtures/classiq_qmod/corpus/original/community/paper_implementation_project/quantum_algo_for_solving_linear_differential_equations/harmonic_oscillator.qmod"
)


def complete_columns(program: synth.Program, reference: fq.Circuit) -> None:
    logical = tuple(
        wire for name in ("x", "ancilla", "y") for wire in program.registers[name]
    )
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    width = resources.n_qubits
    hidden = tuple(w for w in range(width) if w not in logical)
    basis = torch.eye(32, dtype=torch.complex128)
    inputs = torch.zeros((32, 1 << width), dtype=torch.complex128)
    for word in range(32):
        packed = sum(
            ((word >> (4 - bit)) & 1) << (width - wire - 1)
            for bit, wire in enumerate(logical)
        )
        inputs[word, packed] = 1
    state = fq.Circuit.from_ir(
        program.circuit, dtype=torch.complex128, inputs=inputs, bsz=32
    ).state()
    assert isinstance(state, torch.Tensor)
    actual = (
        state.reshape((32,) + (2,) * width)
        .permute((0, *(w + 1 for w in logical), *(w + 1 for w in hidden)))
        .reshape(32, 32, -1)
    )
    expected = fq.Circuit.from_ir(
        reference.to_ir(), dtype=torch.complex128, inputs=basis, bsz=32
    ).state()
    assert isinstance(expected, torch.Tensor)
    torch.testing.assert_close(
        actual[:, :, 0], expected.reshape(32, 32), atol=1e-12, rtol=0
    )
    assert float(actual[:, :, 1:].abs().square().sum()) < 1e-24


def test_untouched_source_native_reference_and_all_matrix_entries() -> None:
    matrices = [
        ast.literal_eval(m)
        for m in re.findall(
            r"unitary\(\s*(\[.*?\])\s*,\s*\w+\s*\)", SOURCE.read_text(), re.S
        )
    ]
    assert len(matrices) == 14
    assert tuple(tuple(row) for row in matrices[1]) == ENCODING
    assert matrices[1] == matrices[-3]
    reference = reference_circuit()
    original = synth.synthesize(SOURCE)
    native = build_program()
    for program in (original, native):
        compare_states(program, reference, ("x", "ancilla", "y"))
        complete_columns(program, reference)
    assert native.report["workspace_qubits"] == 2
    resources = native.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert (resources.n_qubits, resources.n_operations, resources.depth) == (
        7,
        387,
        335,
    )


@pytest.mark.parametrize("scale,amplitudes", [(0.73, (0.6, -0.8)), (1.21, (-0.8, 0.6))])
def test_two_nonzero_bindings_all_columns_relative_global_phases_and_work(
    scale: float, amplitudes: tuple[float, ...]
) -> None:
    native = build_program(scale, amplitudes)
    reference = reference_circuit(scale, amplitudes)
    compare_states(native, reference, ("x", "ancilla", "y"))
    complete_columns(native, reference)


@pytest.mark.parametrize(
    "constraint", ["workspace", "width", "depth", "operations", "memory"]
)
def test_actual_resource_budget_refusal(constraint: str) -> None:
    bindings: dict[str, float | list[float]] = {
        "amplitudes": list(AMPLITUDES),
        "scale": 1.0,
    }
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "workspace":
            synth.synthesize(main, classical_inputs=bindings, max_workspace=1)
        elif constraint == "width":
            synth.synthesize(main, classical_inputs=bindings, max_qubits=4)
        elif constraint == "depth":
            synth.synthesize(main, classical_inputs=bindings, max_depth=1)
        elif constraint == "operations":
            synth.synthesize(main, classical_inputs=bindings, max_operations=10)
        else:
            synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=16)


def test_invalid_bindings_are_not_normalized_or_coerced_silently() -> None:
    with pytest.raises(HybridVerificationError):
        build_program(amplitudes=(0.2, 0.2))
    with pytest.raises(ValueError):
        build_program(amplitudes=(1.0,))
    with pytest.raises(HybridVerificationError):
        build_program(scale=float("nan"))
