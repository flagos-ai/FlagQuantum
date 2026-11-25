"""Check the source-size Gaussian states, compilation and native artifact export."""

import json
import math
from pathlib import Path

import pytest

from examples.synthesis.community.hackathons.iquhack_2025.winning_gaussian.overview import (
    workflow,
)

pytestmark = pytest.mark.integration


def test_generative_window_and_full_signed_state() -> None:
    stage = workflow.build(8, ceiling=18)
    actual = workflow.simulate(stage)
    assert len(actual.grid) == 256 and actual.grid[0] == -2
    assert actual.grid[-1] == 2 - 1 / 64
    assert math.isclose(math.fsum(actual.probabilities), 1.0)
    assert actual.mse < 3e-8
    assert actual.source_state_error < 3e-12 and actual.converted_state_error < 3e-12
    assert actual.sample in actual.grid


def test_127_wire_full_schedule_and_export(tmp_path: Path) -> None:
    stage = workflow.build(127, ceiling=127)
    raw = stage.raw.to_ir()
    assert stage.resources.n_qubits == raw.n_wires == 127
    assert set(stage.resources.operation_counts) == {"u3", "cx"}
    assert sum(op.name == "swap" for op in raw.instructions) == 63
    assert sum(op.name == "h" for op in raw.instructions) == 127
    actual_phases = [
        (op.wires, op.params["theta"]) for op in raw.instructions if op.name == "cphase"
    ]
    expected_phases = [
        ((126 - (j + distance), 126 - j), math.tau / (2 ** (distance + 1)))
        for j in range(127)
        for distance in range(1, min(8, 126 - j) + 1)
    ]
    assert actual_phases == expected_phases
    assert raw.instructions[-1].wires == (126,) and raw.instructions[-1].name == "x"
    path = workflow.save_program(stage, tmp_path, 2)
    assert path.name == "Quantum_Tree__stage_2__resolution_127.json"
    with pytest.raises(ValueError, match="declared stage cap"):
        workflow.build(128, ceiling=127)


def test_original_workflow_stages_are_present_and_unexecuted() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/community/hackathons/iquhack_2025/winning_gaussian/overview/walkthrough.ipynb"
    )
    data = json.loads(path.read_text())
    cells = [cell for cell in data["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 17
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in cells
    )
