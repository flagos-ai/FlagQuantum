"""Original-sized HHL fidelity/cost sweep and explicit rounded RHS admission."""

import json
import math
from pathlib import Path

import pytest
import torch

from examples.synthesis.tutorials.technology_demonstrations.hhl.hhl_example.overview.ansatz import (
    build_program,
    precision_2,
)
from examples.synthesis.tutorials.technology_demonstrations.hhl.hhl_example.overview.physics import (
    AMPLITUDES,
    CLASSICAL,
    HISTORICAL_CX,
    HISTORICAL_DEPTHS,
    HISTORICAL_FIDELITIES,
    MATRIX,
    PRECISIONS,
    RAW_B,
    RAW_B_NORM,
    B,
    reciprocal_angles,
)
from examples.synthesis.tutorials.technology_demonstrations.hhl.hhl_example.overview.workflow import (
    verify_program,
)
from flagquantum import synth

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("precision", PRECISIONS)
def test_every_source_precision_absolute_native_and_converted_state(
    precision: int,
) -> None:
    point = verify_program(build_program(precision), precision)
    assert point.maximum_state_error < 2e-11
    assert point.converted_state_error < 2e-10
    assert point.dirty_workspace_probability < 1e-22
    assert point.resources.n_qubits == precision + 4
    assert set(point.resources.operation_counts) == {"cx", "u3"}
    assert point.resources.operation_counts["cx"] > 0 and point.resources.depth > 0
    assert 0 < point.fidelity <= 1 and 0 < point.success_probability <= 1
    assert point.retained_phase_leakage > 0
    qsol = torch.tensor(point.quantum_solution, dtype=torch.complex128)
    expected = float(
        torch.dot(CLASSICAL.to(torch.complex128) / CLASSICAL.norm(), qsol / qsol.norm())
        .abs()
        .square()
    )
    assert point.fidelity == pytest.approx(expected, abs=1e-14)
    if precision == 8:
        assert 0.998 < point.fidelity < 0.999


def test_original_rounded_rhs_and_explicit_preparation_normalization() -> None:
    assert MATRIX.shape == (8, 8) and RAW_B.shape == (8,)
    assert pytest.approx(0.9999999994356988, abs=1e-15) == RAW_B_NORM
    assert abs(RAW_B_NORM - 1) > 1e-10
    torch.testing.assert_close(B, RAW_B / RAW_B.norm(), atol=1e-15, rtol=0)
    torch.testing.assert_close(MATRIX @ CLASSICAL, RAW_B, atol=1e-14, rtol=0)
    assert len(AMPLITUDES) == 8 and abs(sum(a * a for a in AMPLITUDES) - 1) < 1e-14


@pytest.mark.parametrize("precision", PRECISIONS)
def test_zero_inverse_and_physical_fractional_rotation_table(precision: int) -> None:
    angles = reciprocal_angles(precision)
    assert len(angles) == 1 << precision and angles[0] == 0
    assert [math.sin(angle / 2) for angle in angles[1:]] == pytest.approx(
        [1 / word for word in range(1, 1 << precision)]
    )


@pytest.mark.parametrize("precision", (True, 1, 9))
def test_invalid_precision_is_refused(precision: int) -> None:
    with pytest.raises(ValueError, match="two through eight"):
        build_program(precision)


def test_source_problem_cannot_be_shrunk_to_fit_a_width_budget() -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            precision_2,
            classical_inputs={
                "amplitudes": list(AMPLITUDES),
                "angles": list(reciprocal_angles(2)),
            },
            max_qubits=5,
        )


def test_historical_arrays_keep_all_seven_source_points() -> None:
    assert (
        len(HISTORICAL_CX) == len(HISTORICAL_DEPTHS) == len(HISTORICAL_FIDELITIES) == 7
    )
    assert HISTORICAL_DEPTHS[-1] == 159291 and HISTORICAL_CX[-1] == 81019
    assert HISTORICAL_FIDELITIES[-1] == 1


def test_all_original_code_stages_are_mapped() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/technology_demonstrations/hhl/hhl_example/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        2,
        3,
        6,
        7,
        8,
        11,
        12,
        13,
        14,
        16,
    ]
    for row in proof["reviewed_code_cells"]:
        assert notebook["cells"][row["native_code_cell"]]["cell_type"] == "code"
