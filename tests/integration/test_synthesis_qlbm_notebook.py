"""Verify the fresh source schedule, full absolute state, and actual draws."""

import json
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.cfd.qlbm.overview.model import (
    SHOTS,
    active_magnitudes,
    schedule,
)
from examples.synthesis.applications.cfd.qlbm.overview.reference import source_state
from examples.synthesis.applications.cfd.qlbm.overview.workflow import (
    build_program,
    run_workflow,
)
from examples.synthesis.applications.cfd.qlbm.physics import ACTIVE_VELOCITIES

pytestmark = pytest.mark.integration


def test_fresh_time_schedule_retains_every_simultaneous_magnitude() -> None:
    events = schedule()
    assert len(events) == 9 and events[-1][0] == Fraction(7, 3)
    assert events[3] == (Fraction(1), (1, 2, 3))
    assert events[7] == (Fraction(2), (1, 2, 3))
    assert active_magnitudes() == ACTIVE_VELOCITIES


def test_absolute_phase_all_words_and_work_qubits() -> None:
    program = build_program()
    actual = fq.run(program).statevector().reshape(4096, -1)
    expected = source_state()
    torch.testing.assert_close(actual[:, 0], expected, atol=1e-12, rtol=0)
    assert actual.shape[1] == 128
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22
    assert int(expected.abs().count_nonzero()) == 32


def test_joint_sampling_and_both_marginals() -> None:
    sampled = run_workflow()
    assert sum(sampled.counts.values()) == SHOTS == 2048
    assert len(sampled.probabilities) == 32
    assert all(
        value == pytest.approx(1 / 32, abs=1e-12)
        for value in sampled.probabilities.values()
    )
    assert sampled.position.shape == (8, 8)
    assert sampled.magnitude.shape == (4, 4)
    assert float(sampled.position.sum()) == float(sampled.magnitude.sum()) == 1.0
    assert float(sampled.position[3, 1]) > 0
    for y in range(8):
        for x in range(8):
            assert (
                float(sampled.position[y, x])
                == sum(
                    count for key, count in sampled.counts.items() if key[:2] == (x, y)
                )
                / SHOTS
            )


def test_source_stage_count_and_clean_notebook() -> None:
    from examples.synthesis.applications.cfd.qlbm.overview import (
        __file__ as overview_path,
    )

    assert overview_path is not None
    source = Path(overview_path).parent
    book = json.loads((source / "walkthrough.ipynb").read_text())
    proof = json.loads((source / "provenance.json").read_text())
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == len(proof["reviewed_code_cells"]) == 19
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
