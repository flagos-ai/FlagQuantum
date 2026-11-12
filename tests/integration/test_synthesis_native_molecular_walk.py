"""Complete molecular source/native/Circuit states, phase and environment."""

import hashlib
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_phase_estimation.qpe_with_qubitization.qpe_for_molecule_with_qubitization import (
    main as entry,
)
from examples.synthesis.algorithms.quantum_phase_estimation.qpe_with_qubitization.qpe_for_molecule_with_qubitization.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.quantum_phase_estimation.qpe_with_qubitization.qpe_for_molecule_with_qubitization.physics import (
    PAULI_WORDS,
    PROBABILITIES,
    selection_members,
)
from flagquantum import synth
from flagquantum.errors import PlanningError
from flagquantum.runtime.result import ExecutionResult

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / (
    "tests/fixtures/classiq_qmod/corpus/original/algorithms/quantum_phase_estimation/"
    "qpe_with_qubitization/qpe_for_molecule_with_qubitization.qmod"
)


@pytest.fixture(scope="module")
def native_program() -> synth.Program:
    return entry.build_program()


@pytest.fixture(scope="module")
def native_result(native_program: synth.Program) -> ExecutionResult:
    result: ExecutionResult = fq.run(native_program)
    return result


@pytest.fixture(scope="module")
def expected_state() -> torch.Tensor:
    result: ExecutionResult = fq.run(reference_circuit())
    return result.statevector().flatten()


def test_frozen_source_identity() -> None:
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == (
        "3bc5118a82cd3bc66df0c01f31ee489e8acecdfa4ce8ea96344972fb86988943"
    )


def test_complete_native_state_phase_and_workspace(
    native_program: synth.Program,
    native_result: ExecutionResult,
    expected_state: torch.Tensor,
) -> None:
    assert native_program.registers == {
        "block": (0, 1, 2, 3),
        "phase_var": (4, 5, 6, 7, 8),
    }
    state = native_result.statevector().reshape(8192, -1)
    torch.testing.assert_close(state[:, 0], expected_state, atol=1e-10, rtol=0)
    assert float(state[:, 1:].abs().square().sum()) < 1e-20


def test_unchanged_original_matches_complete_independent_state(
    expected_state: torch.Tensor,
) -> None:
    program = synth.synthesize(
        ORIGINAL, max_qubits=15, max_workspace=2, max_operations=60_000
    )
    state = fq.run(program).statevector().reshape(8192, -1)
    torch.testing.assert_close(state[:, 0], expected_state, atol=1e-10, rtol=0)
    assert float(state[:, 1:].abs().square().sum()) < 1e-20


def test_private_data_is_retained_in_public_density(
    native_result: ExecutionResult, expected_state: torch.Tensor
) -> None:
    actual = native_result.statevector().reshape(8192, -1)[:, 0].reshape(512, 16)
    expected = expected_state.reshape(512, 16)
    torch.testing.assert_close(
        actual @ actual.mH, expected @ expected.mH, atol=1e-10, rtol=0
    )
    assert float(expected[:, 1:].abs().square().sum()) > 0.1


def test_named_phase_readout_preserves_raw_and_signed_fractional_values(
    native_program: synth.Program,
    native_result: ExecutionResult,
    expected_state: torch.Tensor,
) -> None:
    expected = expected_state.reshape(16, 32, 16).abs().square().sum(dim=(0, 2))
    raw = dict(native_program.probabilities(native_result, registers=("phase_var",)))
    numeric = dict(
        native_program.numeric_probabilities(native_result, registers=("phase_var",))
    )
    for word in range(32):
        value = Fraction(word if word < 16 else word - 32, 32)
        assert raw.get((word,), 0.0) == pytest.approx(float(expected[word]), abs=1e-10)
        assert numeric.get((value,), 0.0) == pytest.approx(
            float(expected[word]), abs=1e-10
        )


def test_grouping_retains_every_selector_and_pauli_axis() -> None:
    assert len(PAULI_WORDS) == 16
    assert PAULI_WORDS[15] == PAULI_WORDS[14]
    for axis in ("X", "Y", "Z"):
        members = selection_members(axis)
        for bit in range(4):
            for index, word in enumerate(PAULI_WORDS):
                assert (index in members[5 * bit : 5 * bit + 5]) == (word[bit] == axis)


@pytest.mark.parametrize("axis", ["I", "A", "", "XY"])
def test_invalid_axis_is_rejected(axis: str) -> None:
    with pytest.raises(ValueError, match="Pauli selection"):
        selection_members(axis)


@pytest.mark.parametrize(
    ("limits", "error_type"),
    [
        ({"max_qubits": 14}, synth.SynthesisError),
        ({"max_workspace": 1}, synth.SynthesisError),
        ({"max_operations": 18_933}, synth.SynthesisError),
        ({"memory_limit_bytes": 1024}, PlanningError),
    ],
)
def test_actual_full_program_budgets_are_enforced(
    limits: dict[str, int], error_type: type[Exception]
) -> None:
    parameters: dict[str, list[int] | list[float]] = {
        "probabilities": list(PROBABILITIES),
        "x_members": list(selection_members("X")),
        "y_members": list(selection_members("Y")),
        "z_members": list(selection_members("Z")),
        "powers": [2**index for index in range(4)],
    }
    with pytest.raises(error_type, match="qubits|workspace|operations|memory"):
        synth.synthesize(
            entry.main,
            classical_inputs=parameters,
            max_qubits=limits.get("max_qubits", 24),
            max_workspace=limits.get("max_workspace", 256),
            max_operations=limits.get("max_operations", 100_000),
            memory_limit_bytes=limits.get("memory_limit_bytes", 1024**3),
        )
