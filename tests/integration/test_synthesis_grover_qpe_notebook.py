"""Retain original Grover interference, policy bounds and physical environment."""

import json
from pathlib import Path
from typing import Literal

import pytest
import torch

from examples.synthesis.tutorials.technology_demonstrations.QPE.qpe_for_grover_operator.overview.ansatz import (
    build_program,
    precision_1,
)
from examples.synthesis.tutorials.technology_demonstrations.QPE.qpe_for_grover_operator.overview.workflow import (
    verify_program,
)
from flagquantum import synth

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("precision", range(1, 6))
@pytest.mark.parametrize("objective", ("width", "depth"))
def test_all_original_precision_objectives_match_complete_states(
    precision: int, objective: Literal["width", "depth"]
) -> None:
    program = build_program(precision, objective)
    point = verify_program(program, precision, objective)
    assert point.maximum_state_error < 2e-11
    assert point.converted_state_error < 2e-10
    assert point.named_probability_error < 2e-11
    assert point.resources.n_qubits == precision + 5
    assert point.resources.n_qubits <= 15 + 3 * (precision - 1)
    assert point.retained_data_nonzero > 0.01
    assert program.report["workspace_qubits"] == 0


def test_policy_equality_is_measured_without_claiming_an_optimizer_gain() -> None:
    width = build_program(1, "width")
    depth = build_program(1, "depth")
    assert width.report["resources"] == depth.report["resources"]


def test_global_minus_sign_changes_controlled_phase_readout() -> None:
    import flagquantum as fq
    from examples.synthesis.qpe_reference import dense_qpe
    from examples.synthesis.tutorials.technology_demonstrations.QPE.qpe_for_grover_operator.comparisons.circuit import (
        operators,
    )

    matrix, preparation, _ = operators()
    state = fq.run(dense_qpe(matrix, preparation, 1)).statevector()
    wrong = fq.run(dense_qpe(-matrix, preparation, 1)).statevector()
    assert isinstance(state, torch.Tensor) and isinstance(wrong, torch.Tensor)
    expected = state.reshape(2, 32).abs().square().sum(dim=1)
    incorrect = wrong.reshape(2, 32).abs().square().sum(dim=1)
    assert not torch.allclose(expected, incorrect, atol=1e-8, rtol=0)
    torch.testing.assert_close(expected, incorrect.flip(0), atol=1e-12, rtol=0)


@pytest.mark.parametrize("precision", (True, 0, 6))
def test_precision_boundaries_refuse_invalid_source_sizes(precision: int) -> None:
    with pytest.raises(ValueError, match="one through five"):
        build_program(precision)


def test_width_budget_cannot_shrink_the_original_data() -> None:
    with pytest.raises(ValueError):
        synth.synthesize(precision_1, max_qubits=5, max_workspace=0)


def test_all_eleven_source_stages_are_explicit() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/technology_demonstrations/QPE/qpe_for_grover_operator/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        3,
        4,
        5,
        7,
        8,
        9,
        11,
        14,
        15,
        16,
        18,
    ]
    for row in proof["reviewed_code_cells"]:
        assert notebook["cells"][row["native_code_cell"]]["cell_type"] == "code"
