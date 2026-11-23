"""Verify the original starter sizes and precise local Gaussian comparison."""

import json
from pathlib import Path

import pytest

from examples.synthesis.community.hackathons.iquhack_2025.challenge_template.overview import (
    workflow,
)

pytestmark = pytest.mark.integration


def test_original_eight_bit_starter_and_exact_reference() -> None:
    stage = workflow.build(1)
    observed = workflow.compare(stage)
    assert workflow.resources(stage.starter) == (8, 8, 1, 0)
    # Host math can make one symmetry-cancelled Walsh angle exactly zero or
    # a roundoff-sized nonzero value. Both retain all 254 entanglers; verify the
    # emitted sequence's exact count and its independent absolute state below.
    width, operations, depth, entanglers = workflow.resources(stage.reference)
    assert (width, entanglers) == (8, 254)
    assert (operations, depth) in ((508, 501), (509, 502))
    assert len(observed.grid) == len(observed.starter) == len(observed.reference) == 256
    assert observed.starter_mse > 1e-6
    assert observed.reference_mse < 1e-25
    assert observed.absolute_starter_error < 3e-12
    assert observed.absolute_reference_error < 3e-12
    assert observed.starter_sample in observed.grid


def test_source_stage_two_width_and_native_export(tmp_path: Path) -> None:
    stage = workflow.build(2)
    assert workflow.resources(stage.starter) == (4, 4, 1, 0)
    assert workflow.resources(stage.reference) == (4, 29, 26, 14)
    with pytest.raises(ValueError, match="team name"):
        workflow.save_program(stage.starter, tmp_path, 2, None)
    path = workflow.save_program(stage.starter, tmp_path, 2, "Test Team")
    assert path.name == "Test_Team__stage_2__resolution_4.json"
    assert json.loads(path.read_text())["n_wires"] == 4
    with pytest.raises(ValueError, match="only the eight-qubit"):
        workflow.compare(stage)


def test_all_original_code_stages_and_disabled_team_export() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/community/hackathons/iquhack_2025/challenge_template/overview/walkthrough.ipynb"
    )
    data = json.loads(path.read_text())
    cells = [cell for cell in data["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 17
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in cells
    )
    assert "team_name = None" in "".join(cells[-1]["source"])
