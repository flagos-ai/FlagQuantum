"""Full source matrix, all precision points and retained environments stay explicit."""

import json
import math
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq

pytest.importorskip("numpy")

from examples.synthesis.tutorials.technology_demonstrations.QPE.qpe_for_unitary_matrix.overview.ansatz import (
    build_program,
)
from examples.synthesis.tutorials.technology_demonstrations.QPE.qpe_for_unitary_matrix.overview.physics import (
    EIGENVALUES,
    HISTORICAL_QISKIT_CX,
    HISTORICAL_QISKIT_DEPTHS,
    MATRIX,
    PRECISIONS,
    UNITARY,
    UNUSED_SHIFTED_MATRIX,
)
from examples.synthesis.tutorials.technology_demonstrations.QPE.qpe_for_unitary_matrix.overview.workflow import (
    verify_program,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("precision", range(1, 9))
def test_every_original_precision_preserves_absolute_states_and_target_cost(
    precision: int,
) -> None:
    point = verify_program(build_program(precision), precision)
    assert point.maximum_state_error < 2e-11 and point.converted_state_error < 2e-10
    assert point.named_probability_error < 2e-11
    assert point.resources.n_qubits == precision + 3
    assert set(point.resources.operation_counts) == {"cx", "u3"}
    assert 0 < point.retained_environment_probability < 1


def test_actual_source_matrix_is_not_silently_replaced_by_its_unused_shift() -> None:
    actual = torch.tensor(UNITARY, dtype=torch.complex128)
    torch.testing.assert_close(
        actual, torch.matrix_exp(2j * math.pi * MATRIX), atol=1e-14, rtol=0
    )
    torch.testing.assert_close(
        actual.mH @ actual,
        torch.eye(8, dtype=torch.complex128),
        atol=32 * torch.finfo(torch.float64).eps,
        rtol=0,
    )
    assert (
        MATRIX.shape == (8, 8)
        and float(EIGENVALUES.min()) < 0
        and float(EIGENVALUES.max()) > 1
    )
    assert not torch.allclose(
        actual, torch.matrix_exp(2j * math.pi * UNUSED_SHIFTED_MATRIX)
    )


def test_named_fractional_readout_marginalizes_retained_data_without_reset() -> None:
    program = build_program(3)
    result = fq.run(program)
    state = result.statevector()
    assert isinstance(state, torch.Tensor)
    named = program.numeric_probabilities(result, registers=("phase",))
    physical = state.reshape(8, 8).abs().square().sum(dim=1)
    for word in range(8):
        assert named.get((Fraction(word, 8),), 0.0) == pytest.approx(
            float(physical[word]), abs=2e-11
        )
    assert sum(named.values()) == pytest.approx(1.0, abs=2e-11)
    assert float(state.reshape(8, 8)[:, 1:].abs().square().sum()) > 1e-3


@pytest.mark.parametrize("precision", (0, 9, -1, True, False))
def test_invalid_precisions_do_not_change_source_sizes(precision: int) -> None:
    with pytest.raises(ValueError, match="precision"):
        build_program(precision)


def test_original_historical_reference_is_kept_as_data() -> None:
    assert tuple(range(1, 9)) == PRECISIONS
    assert HISTORICAL_QISKIT_DEPTHS == (320, 950, 2206, 4706, 9694, 19658, 39574, 79394)
    assert HISTORICAL_QISKIT_CX == (162, 484, 1124, 2398, 4938, 10008, 20136, 40378)


def test_all_seven_source_stages_include_the_commented_reference_cells() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/technology_demonstrations/QPE/qpe_for_unitary_matrix/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        2,
        3,
        5,
        7,
        8,
        9,
        11,
    ]
    for row in proof["reviewed_code_cells"]:
        assert notebook["cells"][row["native_code_cell"]]["cell_type"] == "code"
