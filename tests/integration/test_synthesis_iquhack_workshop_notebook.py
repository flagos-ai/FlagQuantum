"""Verify the workshop's complete native arithmetic and state preparation."""

import json
import math
from pathlib import Path

import pytest

from examples.synthesis.community.hackathons.iquhack_2025.workshop.overview import (
    workflow,
)

pytestmark = pytest.mark.integration


def test_unsigned_and_signed_full_complex_states() -> None:
    arithmetic = workflow.arithmetic()
    assert (arithmetic.qubits, arithmetic.operations) == (14, 69)
    assert arithmetic.state_error < 3e-12 and arithmetic.workspace_probability == 0
    values, signed = workflow.fixed_point()
    assert values == (5, -7.5, -2.5)
    assert signed.qubits == 14 and signed.state_error < 3e-12


def test_seeded_random_distribution_and_complete_qft() -> None:
    pytest.importorskip("numpy")
    probabilities = workflow.random_probabilities()
    assert len(probabilities) == 32 and math.isclose(math.fsum(probabilities), 1)
    prepared, distance = workflow.prepare_random(probabilities)
    assert prepared.qubits == 5 and prepared.state_error < 3e-12
    assert distance <= 0.01
    checked, output, sample = workflow.exponential_qft()
    assert len(output) == 32 and math.isclose(math.fsum(output), 1)
    assert checked.qubits == 5 and checked.state_error < 3e-12
    assert 0 <= sample < 32


def test_fractional_square_and_joint_counts() -> None:
    checked, sample = workflow.amplitude_assignment()
    assert checked.qubits == 6 and checked.state_error < 3e-12
    assert sum(count for _, count in sample.joint) == 2048
    assert sum(count for _, count in sample.marginal) == 2048
    marginal = dict(sample.marginal)
    assert len(marginal) == 32
    for word in range(32):
        x = word / 32
        assert marginal[x] == sum(
            count for (value, _), count in sample.joint if value == x
        )
    assert math.isclose(sample.expected_indicator_probability, 0.18470048904418945)


def test_all_original_workflow_stages_unexecuted() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/community/hackathons/iquhack_2025/workshop/overview/walkthrough.ipynb"
    )
    cells = json.loads(path.read_text())["cells"]
    code = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(code) == 16
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in code
    )
