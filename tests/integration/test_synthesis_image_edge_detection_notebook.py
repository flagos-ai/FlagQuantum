"""Verify image preparation, full-state stencil, and original finite sampling."""

import json
import math
from pathlib import Path

import pytest
import torch

pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

from examples.synthesis.applications.image_processing.quantum_hadamard_edge_detection.overview.quantum import (
    prepare_image,
    walsh_angles,
)
from examples.synthesis.applications.image_processing.quantum_hadamard_edge_detection.overview.workflow import (
    SHOTS,
    THRESHOLD,
    load_photo,
    pixel_amplitudes,
    run,
)

pytestmark = pytest.mark.integration


def test_full_image_full_state_and_draws() -> None:
    result = run()
    assert result.photo.shape == (128, 128, 3)
    assert float(pixel_amplitudes(result.photo).norm()) == pytest.approx(1)
    assert torch.equal(result.preview[0], result.photo[0])
    assert not torch.equal(result.preview, result.photo)
    assert result.preparation_error < 1e-12 and result.state_error < 1e-12
    assert result.work_probability < 1e-24
    assert result.preparation_resources.n_qubits == 14
    assert result.preparation_resources.operation_counts == {"ry": 16383, "cx": 16382}
    assert result.resources.n_qubits == 20
    assert result.resources.n_operations <= 35_000
    assert result.edge_probability == pytest.approx(0.056223561978731934, abs=1e-12)
    assert (
        abs(
            float(result.sampled_edge.square().sum())
            - result.observed_edge_shots / SHOTS
        )
        < 1e-12
    )
    assert abs(result.observed_edge_shots / SHOTS - result.edge_probability) < 0.003
    assert torch.equal(
        result.thresholded_edge,
        torch.where(result.sampled_edge > THRESHOLD, result.sampled_edge, 0.0),
    )


def test_walsh_convention_and_preparation_refusals() -> None:
    angles = torch.tensor([0, 0.25, -0.5, 1], dtype=torch.float64)
    expected = torch.tensor(
        [
            sum(float(a) * (-1) ** ((i & j).bit_count()) for j, a in enumerate(angles))
            / 4
            for i in range(4)
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(walsh_angles(angles), expected)
    with pytest.raises(ValueError, match="power-of-two"):
        walsh_angles(torch.ones(3))
    with pytest.raises(ValueError, match="16384"):
        prepare_image(angles)
    pixels = pixel_amplitudes(load_photo()).flatten()
    with pytest.raises(ValueError, match="unit norm"):
        prepare_image(pixels * math.sqrt(2))
    with pytest.raises(ValueError, match="nonnegative"):
        prepare_image(-pixels)


def test_all_original_notebook_cells_are_unexecuted() -> None:
    overview = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/applications/image_processing/quantum_hadamard_edge_detection/overview"
    )
    book = json.loads((overview / "walkthrough.ipynb").read_text())
    proof = json.loads((overview / "provenance.json").read_text())
    stages = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(stages) == len(proof["reviewed_code_cells"]) == 17
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in stages
    )
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        1,
        2,
        3,
        6,
        7,
        8,
        10,
        13,
        14,
        15,
        16,
        17,
        20,
        22,
        24,
        26,
        28,
    ]
