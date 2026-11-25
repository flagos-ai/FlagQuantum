"""Split coordinates preserve the original two-dimensional Laplacian unitary."""

from pathlib import Path

import pytest
import torch

from examples.synthesis.community.paper_implementation_project.block_encoding.laplacian_4x4.comparisons.circuit import (
    block_matrix,
    reference_circuit,
)
from examples.synthesis.community.paper_implementation_project.block_encoding.laplacian_4x4.main import (
    ANGLES,
    PROBABILITIES,
    build_program,
    main,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def test_untouched_source_and_native_match_independent_complete_state() -> None:
    source = (
        ROOT
        / "tests/fixtures/classiq_qmod/corpus/original/community/paper_implementation_project/block_encoding/laplacian_BE_Nx4_Ny4.qmod"
    )
    original = synth.synthesize(source)
    compare_states(original, reference_circuit(), ("s", "j", "data", "dlt"))
    native = build_program()
    compare_states(native, reference_circuit(), ("s", "j1", "j0", "data", "dlt"))
    assert tuple(map(len, native.registers.values())) == (3, 2, 2, 1, 1)
    assert native.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    "offset,probabilities",
    [(0.23, (0.125,) * 8), (-0.37, tuple(i / 36 for i in range(1, 9)))],
)
@pytest.mark.parametrize("initial_data,initial_flag", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_two_nonzero_bindings_all_selector_branches_and_nonzero_data_and_flag(
    offset: float,
    probabilities: tuple[float, ...],
    initial_data: int,
    initial_flag: int,
) -> None:
    angles = tuple(angle + offset * (i + 1) for i, angle in enumerate(ANGLES))
    compare_states(
        build_program(angles, probabilities, initial_data, initial_flag),
        reference_circuit(angles, probabilities, initial_data, initial_flag),
        ("s", "j1", "j0", "data", "dlt"),
    )


def test_independent_block_has_no_colliding_columns_or_missing_phase() -> None:
    matrix = block_matrix(ANGLES)
    torch.testing.assert_close(
        matrix.conj().T @ matrix,
        torch.eye(512, dtype=torch.complex128),
        atol=1e-14,
        rtol=0,
    )


@pytest.mark.parametrize("constraint", ["width", "depth", "operations", "memory"])
def test_budget_refusal_before_execution(constraint: str) -> None:
    bindings: dict[str, float | list[float]] = {
        "probabilities": list(PROBABILITIES),
        "angles": list(ANGLES),
        "initial_data": 0,
        "initial_flag": 0,
    }
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "width":
            synth.synthesize(main, classical_inputs=bindings, max_qubits=8)
        elif constraint == "depth":
            synth.synthesize(main, classical_inputs=bindings, max_depth=1)
        elif constraint == "operations":
            synth.synthesize(main, classical_inputs=bindings, max_operations=10)
        else:
            synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=16)


def test_invalid_selector_probabilities_and_initial_flags_refuse() -> None:
    with pytest.raises(HybridVerificationError):
        build_program(probabilities=(0.2,) * 8)
    with pytest.raises(ValueError):
        build_program(initial_data=2)
    with pytest.raises(ValueError):
        build_program(initial_flag=-1)
    with pytest.raises(ValueError):
        build_program(angles=ANGLES[:-1])
