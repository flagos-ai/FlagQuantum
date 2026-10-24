"""Geometry, mixed-decoder composition and resumed CNN training behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flagquantum.qec import get_decoder
from flagquantum.qec.decoders._ai.assessment import evaluate
from flagquantum.qec.decoders._ai.spacetime import SpacetimePredecoder
from flagquantum.qec.decoders._ai.training import Trainer
from tests.qec.test_neural_training_data import _experiment

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("basis", ["x", "z"])
@pytest.mark.parametrize("rounds", [1, 3])
def test_detector_roundtrip_and_boundary_masks(basis: str, rounds: int) -> None:
    memory, noise, dem = _experiment(basis, rounds, 0.02)
    model = SpacetimePredecoder(memory, noise)
    detectors = torch.eye(dem.num_detectors)
    encoded = model.encode_input(detectors)
    assert encoded.shape == (dem.num_detectors, 4, rounds + 1, 5, 5)
    assert torch.equal(encoded[:, 0].flatten(1)[:, model.sites], detectors)
    assert int(model.masks[0].sum()) == dem.num_detectors
    assert torch.equal(model.masks[0], model.masks[1] + model.masks[2])
    assert bool((encoded[:, 0] * (1 - model.masks[0]) == 0).all())
    # Initial and terminal slices have fewer valid checks than a bulk slice.
    assert int(model.masks[0, 0].sum()) == 4
    assert int(model.masks[0, -1].sum()) == 4
    if rounds > 1:
        assert int(model.masks[0, 1].sum()) == 8
    for detector, site in zip(
        memory.detectors.detectors, model.sites.tolist(), strict=True
    ):
        expected_time = (
            memory.rounds
            if any(ref.round_index is None for ref in detector.parity)
            else max(
                ref.round_index
                for ref in detector.parity
                if ref.round_index is not None
            )
        )
        assert site // 25 == expected_time
    assert model(detectors).shape == (dem.num_detectors, dem.num_errors)


def test_cnn_parameters_train_and_resume_exactly(tmp_path: Path) -> None:
    memory, noise, dem = _experiment("z", 1, 0.02)
    torch.manual_seed(9)
    model = SpacetimePredecoder(memory, noise)
    trainer = Trainer(model, seed=3)
    initial = model.encoder[0].weight.detach().clone()
    trainer.train(2, batch_size=8)
    assert not torch.equal(initial, model.encoder[0].weight)
    checkpoint = tmp_path / "cnn.pt"
    trainer.save(checkpoint)
    expected = trainer.train(2, batch_size=8)
    restored = Trainer(SpacetimePredecoder(memory, noise), seed=99)
    restored.restore(checkpoint)
    assert restored.train(2, batch_size=8) == expected
    for left, right in zip(
        model.parameters(), restored.model.parameters(), strict=True
    ):
        assert torch.equal(left, right)
    assert (
        evaluate(
            restored.model,
            get_decoder("minimum_weight_matching", dem),
            shots=16,
            seed=1001,
        )["shots"]
        == 16
    )


def test_nonzero_cnn_corrections_preserve_logical_composition() -> None:
    memory, noise, dem = _experiment("z", 1, 0.02)
    model = SpacetimePredecoder(memory, noise)
    for parameter in model.parameters():
        parameter.data.zero_()
    model.network[-1].bias.data.fill_(10)
    sample = model.sample(16, torch.Generator().manual_seed(5)).sample
    backend = get_decoder("minimum_weight_matching", dem)
    predicted = model.decode(sample.detectors, backend)
    # Independently XOR all selected mechanism signatures, including Lc.
    selected_detectors: set[int] = set()
    selected_logicals: set[int] = set()
    for error in dem.errors:
        selected_detectors.symmetric_difference_update(error.detectors)
        selected_logicals.symmetric_difference_update(error.observables)
    for row, actual in zip(sample.detectors, predicted, strict=True):
        residual = set(row.nonzero().flatten().tolist()) ^ selected_detectors
        expected = set(backend.decode(sorted(residual)).observables) ^ selected_logicals
        assert set(actual.nonzero().flatten().tolist()) == expected


def test_invalid_inputs_refused() -> None:
    memory, noise, _ = _experiment("z", 1, 0.02)
    model = SpacetimePredecoder(memory, noise)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(1, 2))
    with pytest.raises(ValueError, match="binary"):
        model(torch.full((1, model.h.shape[0]), 2))


def test_local_embeddings_pool_in_original_mechanism_order() -> None:
    from torch import nn

    class FirstFeature(nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return features[..., :1]

    memory, noise, dem = _experiment("z", 1, 0.02)
    model = SpacetimePredecoder(memory, noise)
    pointwise = nn.Conv3d(4, 8, kernel_size=1, bias=False)
    pointwise.weight.data.zero_()
    pointwise.weight.data[0, 0, 0, 0, 0] = 1
    model.encoder = nn.Sequential(pointwise)
    model.network = nn.Sequential(FirstFeature())
    detectors = torch.eye(dem.num_detectors)
    logits = model(detectors)
    # Each column's response is 1/degree at exactly its incident detectors.
    for column, error in enumerate(dem.errors):
        expected = torch.zeros(dem.num_detectors)
        expected[list(error.detectors)] = 1 / len(error.detectors)
        assert torch.equal(logits[:, column], expected)
