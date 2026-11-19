"""Check the original-scale native bars-and-stripes training workflow."""

import json
from pathlib import Path

import pytest
import torch

from examples.synthesis.algorithms.machine_learning.qgan.qgan_bars_and_strips.overview import (
    neural,
    workflow,
)

pytestmark = pytest.mark.integration


def test_batched_native_states_and_source_quantum_shape() -> None:
    weights = torch.linspace(-0.2, 0.4, 8, dtype=torch.float64)
    assert neural.check_states(weights) < 3e-12
    with pytest.raises(ValueError, match="four features and eight weights"):
        neural.batched_state(torch.zeros((3, 4)), weights[:7])
    sampler = neural.SampledLayer(seed=106)
    inputs = torch.tensor([[0.1, 0.2, 0.3, 0.4], [-0.1, 0.7, -0.4, 0.12]])
    values = sampler.forward(inputs)
    assert sampler.query_rows == 2 and values.shape == (2, 4)
    assert bool(((values >= 0) & (values <= 1)).all())
    scaled = values * neural.SHOTS
    torch.testing.assert_close(scaled, scaled.round(), atol=0, rtol=0)


def test_original_sizes_and_rounded_generator_training() -> None:
    np = pytest.importorskip("numpy")
    rng = np.random.RandomState(42)
    classical = workflow.dataset(1000, rng)
    hybrid = workflow.dataset(250, rng)
    assert classical.shape == (1000, 2, 2) and hybrid.shape == (250, 2, 2)
    assert workflow.score(classical).bars_and_stripes == 1
    assert workflow.score(hybrid).distinct_patterns == 6
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        baseline = workflow.train(
            neural.Generator(),
            neural.Discriminator(),
            classical,
            workflow.gaussian_noise,
            10,
        )
        generator = neural.QuantumGenerator(seed=107)
        quantum = workflow.train(
            generator, neural.Discriminator(), hybrid, workflow.binary_noise, 1
        )
    assert len(baseline.losses) == 160 and len(quantum.losses) == 4
    assert generator.qlayer.query_rows == 500
    for metrics in (baseline, quantum):
        assert metrics.maximum_generator_gradient == 0
        assert metrics.changed_generator_tensors == 0
        assert metrics.changed_discriminator_tensors == 6


def test_original_checkpoint_guard_and_complementary_readouts(tmp_path: Path) -> None:
    path = tmp_path / neural.CLASSICAL_ASSET
    path.write_bytes(b"unexpected weights")
    with pytest.raises(ValueError, match="pinned original"):
        neural.load_checkpoint(neural.Generator(), tmp_path, neural.CLASSICAL_ASSET)
    all_ones = torch.ones((4, 4))
    collapsed = workflow.score(all_ones)
    assert collapsed == workflow.Scores(1.0, 1.0, 1)
    single_one = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert workflow.score(single_one) == workflow.Scores(1.0, 0.0, 1)


def test_all_original_workflow_stages_unexecuted() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/algorithms/machine_learning/qgan/qgan_bars_and_strips/overview/walkthrough.ipynb"
    )
    cells = json.loads(path.read_text())["cells"]
    code = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(code) == 23
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in code
    )
