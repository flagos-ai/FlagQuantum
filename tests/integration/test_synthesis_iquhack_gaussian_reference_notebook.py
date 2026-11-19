"""Verify both source widths and the measured workspace mismatch."""

import json
from pathlib import Path

import pytest

from examples.synthesis.community.hackathons.iquhack_2025.gaussian_reference.overview import (
    main,
    workflow,
)

pytestmark = pytest.mark.integration


def test_original_direct_preparation() -> None:
    program, error = workflow.exact_grid()
    assert workflow.resource_row(program)[0] == 6
    assert error < 3e-12


@pytest.mark.parametrize("width", [4, 8])
def test_measured_original_mismatch_and_checked_amplification(
    width: int, tmp_path: Path
) -> None:
    archived = workflow.run_amplification(width, preserve_source_overlap=True)
    checked = workflow.run_amplification(width, preserve_source_overlap=False)
    assert archived.overlap != checked.overlap == archived.actual_overlap
    assert archived.workspace_probability > 0.01
    assert (
        abs(archived.workspace_probability - archived.predicted_workspace_probability)
        < 2e-13
    )
    assert checked.workspace_probability < 1e-22 and checked.state_error < 3e-12
    with pytest.raises(ValueError, match="clean work qubits"):
        main.save_checked_program(archived, tmp_path, team="Test Team")
    assert main.save_checked_program(checked, tmp_path, team="Test Team").is_file()


def test_original_stages_and_disabled_team_export() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/community/hackathons/iquhack_2025/gaussian_reference/overview/walkthrough.ipynb"
    )
    data = json.loads(path.read_text())
    cells = [cell for cell in data["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 19
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in cells
    )
    assert "team_name = None" in "".join(cells[-1]["source"])
