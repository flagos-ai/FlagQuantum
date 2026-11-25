"""Check the regenerated hydrogen sector and full controlled-state readout."""

import json
import math
from pathlib import Path

import pytest
import torch

from examples.synthesis.community.paper_implementation_project.bpde.overview import (
    model,
    quantum,
)

pytestmark = pytest.mark.integration


def test_fresh_original_curve_and_displaced_energy_difference() -> None:
    rows = model.curve()
    positions, derivatives = model.differences(rows)
    assert len(rows) == 30 and len(positions) == len(derivatives) == 29
    assert (rows[0].distance, rows[-1].distance) == (0.4, 2.1)
    assert min(row.fci for row in rows) < -1.13
    left, right = model.displaced()
    assert (left.distance, right.distance) == pytest.approx(
        (0.99875, 1.00125), abs=1e-15
    )
    assert right.fci - left.fci == pytest.approx(0.000527501038270, abs=1e-11)
    assert (left.scalar, left.z, left.x) == pytest.approx(
        (-0.530427259238194, 0.536032051970536, 0.196711389481122), abs=1e-9
    )


@pytest.mark.parametrize(("time", "repetitions"), ((1.0, 2), (20.0, 40), (200.0, 400)))
def test_full_states_reproduce_every_trotter_factor(
    time: float, repetitions: int
) -> None:
    models = model.displaced()
    experiment = quantum.build(models, time)
    assert experiment.repetitions == repetitions
    assert experiment.resources.n_qubits == 2 and experiment.resources.n_operations == 5
    assert tuple(experiment.template.parameter_names) == ("delta_phase",)
    for trial in (0.00035, 0.0007):
        observed = quantum.execute(experiment, trial)
        reference = quantum.reference(models, time, trial)
        torch.testing.assert_close(observed, reference, atol=1e-11, rtol=0)
        assert math.isclose(float(observed.abs().square().sum()), 1, abs_tol=1e-11)


def test_seeded_finite_counts_from_the_executed_state() -> None:
    pytest.importorskip("numpy")
    from examples.synthesis.community.paper_implementation_project.bpde.overview import (
        workflow,
    )

    scan = workflow.scan(model.displaced(), 20.0, workflow.TRIALS)
    assert len(scan.zero_counts) == 5
    assert scan.zero_counts == (999950, 999966, 999958, 999955, 999960)
    for count, estimate, probability in zip(
        scan.zero_counts, scan.z_estimates, scan.exact_zero, strict=True
    ):
        assert estimate == 2 * count / workflow.SHOTS - 1
        assert abs(count / workflow.SHOTS - probability) < 0.001


def test_source_stage_count_and_invalid_time() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        quantum.build(model.displaced(), 0)
    book = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "examples/synthesis/community/paper_implementation_project/bpde/overview/walkthrough.ipynb"
        ).read_text()
    )
    code = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 14
    assert all(cell["execution_count"] is None and not cell["outputs"] for cell in code)
