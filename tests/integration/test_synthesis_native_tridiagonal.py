"""Tridiagonal encoding retains the original32 table entries and signed preparation."""

from pathlib import Path

import pytest

from examples.synthesis.community.paper_implementation_project.block_encoding.tridiagonal.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.community.paper_implementation_project.block_encoding.tridiagonal.main import (
    AMPLITUDES,
    ANGLES,
    VALUES,
    build_program,
    main,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem.contracts import InteropConversionError
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def test_approved_normalization_and_native_match_complete_reference_state() -> None:
    original = synth.synthesize(
        ROOT
        / "tests/fixtures/classiq_qmod/corpus/approved/community/paper_implementation_project/block_encoding/tridiagonal_BE_size16.qmod"
    )
    compare_states(original, reference_circuit(), ("s", "j", "data"))
    native = build_program()
    compare_states(native, reference_circuit(), ("s", "j", "data"))
    assert len(VALUES) == len(ANGLES) == 32
    assert VALUES[-1] == 0
    assert tuple(map(len, native.registers.values())) == (2, 4, 1)
    assert native.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    "offset,amplitudes",
    [(0.17, (0.5, -0.5, 0.5, 0.5)), (-0.31, (-0.5, 0.5, 0.5, -0.5))],
)
def test_two_nonzero_table_bindings_cover_all_selector_branches(
    offset: float, amplitudes: tuple[float, ...]
) -> None:
    angles = tuple(
        value + offset * (index + 1) / 32 for index, value in enumerate(ANGLES)
    )
    compare_states(
        build_program(angles, amplitudes),
        reference_circuit(angles, amplitudes),
        ("s", "j", "data"),
    )


@pytest.mark.parametrize("constraint", ["operations", "qubits", "depth", "memory"])
def test_tridiagonal_budget_refusals(constraint: str) -> None:
    bindings = {"angles": list(ANGLES), "amplitudes": list(AMPLITUDES)}
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "operations":
            synth.synthesize(main, classical_inputs=bindings, max_operations=10)
        elif constraint == "qubits":
            synth.synthesize(main, classical_inputs=bindings, max_qubits=6)
        elif constraint == "depth":
            synth.synthesize(main, classical_inputs=bindings, max_depth=1)
        else:
            synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=16)


def test_table_and_selector_width_cannot_be_silently_changed() -> None:
    with pytest.raises(ValueError, match="32 table angles"):
        build_program(ANGLES[:-1])
    with pytest.raises(ValueError, match="four normalized"):
        build_program(amplitudes=AMPLITUDES[:-1])


def test_original_rounded_amplitudes_remain_a_reported_semantic_refusal() -> None:
    with pytest.raises(InteropConversionError, match="normalized finite real vector"):
        synth.synthesize(
            ROOT
            / "tests/fixtures/classiq_qmod/corpus/original/community/paper_implementation_project/block_encoding/tridiagonal_BE_size16.qmod"
        )
