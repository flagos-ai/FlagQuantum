"""Toeplitz native translation preserves all columns and signed snapshot flags."""

from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.block_encoding.toeplitz.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.community.paper_implementation_project.block_encoding.toeplitz.main import (
    ANGLES,
    VALUES,
    build_program,
    main,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def complete_columns(program: synth.Program) -> torch.Tensor:
    assert program.report["workspace_qubits"] == 0
    assert tuple(program.registers) == ("data", "s", "j", "dlt")
    basis = torch.eye(256, dtype=torch.complex128)
    state = fq.Circuit.from_ir(
        program.circuit, dtype=torch.complex128, inputs=basis, bsz=256
    ).state()
    assert isinstance(state, torch.Tensor)
    return state.reshape(256, 256)


def reference_columns(angles: tuple[float, ...]) -> torch.Tensor:
    basis = torch.eye(256, dtype=torch.complex128)
    state = fq.Circuit.from_ir(
        reference_circuit(angles).to_ir(),
        dtype=torch.complex128,
        inputs=basis,
        bsz=256,
    ).state()
    assert isinstance(state, torch.Tensor)
    return state.reshape(256, 256)


def test_untouched_source_native_and_independent_complete_state() -> None:
    source = (
        ROOT
        / "tests/fixtures/classiq_qmod/corpus/original/community/paper_implementation_project/block_encoding/toeplitz_BE_N16.qmod"
    )
    original = synth.synthesize(source)
    compare_states(original, reference_circuit(), ("data", "s", "j", "dlt"))
    native = build_program()
    compare_states(native, reference_circuit(), ("data", "s", "j", "dlt"))
    assert tuple(map(len, native.registers.values())) == (1, 2, 4, 1)
    assert len(VALUES) == 4
    distribution = dict(native.probabilities(fq.run(native), registers=("dlt",)))
    assert distribution == pytest.approx({(0,): 15 / 16, (1,): 1 / 16})


@pytest.mark.parametrize("offset", [0.0, 0.29, -0.43])
def test_all_256_columns_include_nonzero_xor_underflow_overflow_and_coherence(
    offset: float,
) -> None:
    angles = tuple(angle + offset * (i + 1) for i, angle in enumerate(ANGLES))
    actual = complete_columns(build_program(angles))
    expected = reference_columns(angles)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        actual @ actual.conj().T,
        torch.eye(256, dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )
    coherent = torch.exp(0.13j * torch.arange(256, dtype=torch.float64)) / 16
    torch.testing.assert_close(
        coherent @ actual, coherent @ expected, atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("constraint", ["width", "depth", "operations", "memory"])
def test_budget_refusal(constraint: str) -> None:
    bindings = {"angles": list(ANGLES)}
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "width":
            synth.synthesize(main, classical_inputs=bindings, max_qubits=7)
        elif constraint == "depth":
            synth.synthesize(main, classical_inputs=bindings, max_depth=1)
        elif constraint == "operations":
            synth.synthesize(main, classical_inputs=bindings, max_operations=10)
        else:
            synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=16)


def test_all_coefficients_are_required() -> None:
    with pytest.raises(ValueError):
        build_program(ANGLES[:-1])
