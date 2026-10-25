"""Synthetic DEM/grid behavior; these layouts are not Ising measurement evidence."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import torch
from torch import nn

from flagquantum.qec.decoders._ai.comparison import decode
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.spacetime import SpacetimePredecoder
from flagquantum.qec.decoders._ai.training import Trainer
from flagquantum.qec.dem.model import DemError, DetectorErrorModel
from tests.qec.test_neural_training_data import _experiment

pytestmark = pytest.mark.integration


def _dem() -> DetectorErrorModel:
    return DetectorErrorModel(
        4,
        2,
        (
            DemError(0.01, (0, 1, 2), (0,)),
            DemError(0.02, (1, 2, 3), (1,)),
            DemError(0.03, (0, 1, 2, 3), (0, 1)),
            DemError(0.04, (2,), ()),
        ),
    )


def _model() -> SpacetimePredecoder:
    return SpacetimePredecoder.from_dem(
        _dem(), grid_shape=(2, 3, 3), sites=(13, 1, 17, 6), kinds=(1, 2, 1, 2)
    )


def _rows() -> torch.Tensor:
    return torch.tensor(list(itertools.product((0, 1), repeat=4)), dtype=torch.int8)


def test_nonuniform_detector_order_roundtrips_and_keeps_original_dem() -> None:
    model = _model()
    bits = torch.cat((torch.zeros(1, 4), torch.ones(1, 4), torch.eye(4)))
    encoded = model.encode_input(bits)
    assert encoded.shape == (6, 4, 2, 3, 3)
    for detector, site in enumerate((13, 1, 17, 6)):
        assert torch.equal(encoded[:, 0].flatten(1)[:, site], bits[:, detector])
        assert model.masks.flatten(1)[1 + detector % 2, site] == 1
    assert int(model.masks[0].sum()) == 4
    assert torch.equal(model.masks[0], model.masks[1] + model.masks[2])
    assert not bool((encoded[:, 0] * (1 - model.masks[0])).any())
    assert model.dem_text == _dem().to_stim_text()
    assert torch.equal(model.h, _dem().detector_error_matrix().float())
    assert torch.equal(model.logical, _dem().observables_flips_matrix().float())
    assert sum(parameter.numel() for parameter in model.parameters()) == 2817
    with pytest.raises(ValueError, match="graphlike"):
        Predecoder(_dem())


def test_hyperedge_pooling_has_independent_scalar_incidence_oracle() -> None:
    class FirstFeature(nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return features[..., :1]

    model = _model()
    pointwise = nn.Conv3d(4, 8, 1, bias=False)
    with torch.no_grad():
        pointwise.weight.zero_()
        pointwise.weight[0, 0, 0, 0, 0] = 1
    model.encoder = nn.Sequential(pointwise)
    model.network = nn.Sequential(FirstFeature())
    bits = _rows()
    expected = torch.tensor(
        [
            [
                sum(int(row[d]) for d in error.detectors) / len(error.detectors)
                for error in _dem().errors
            ]
            for row in bits
        ]
    )
    torch.testing.assert_close(model(bits), expected)


def test_memory_construction_and_explicit_layout_have_identical_state_and_rng() -> None:
    memory, noise, dem = _experiment("x", 3, 0.02)
    torch.manual_seed(9)
    original = SpacetimePredecoder(memory, noise)
    after_original = torch.get_rng_state()
    kinds = tuple(
        1 if original.masks.flatten(1)[1, site] else 2
        for site in original.sites.tolist()
    )
    torch.manual_seed(9)
    explicit = SpacetimePredecoder.from_dem(
        dem,
        grid_shape=original.grid_shape,
        sites=tuple(original.sites.tolist()),
        kinds=kinds,
    )
    assert torch.equal(after_original, torch.get_rng_state())
    assert original.state_dict().keys() == explicit.state_dict().keys()
    for key, value in original.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])
    bits = torch.eye(dem.num_detectors)
    assert torch.equal(original(bits), explicit(bits))


def test_consistent_detector_permutation_preserves_encoded_input_and_logits() -> None:
    model = _model()
    permutation = (2, 0, 3, 1)
    sites, kinds = [0] * 4, [0] * 4
    for old, new in enumerate(permutation):
        sites[new], kinds[new] = (13, 1, 17, 6)[old], (1, 2, 1, 2)[old]
    dem = DetectorErrorModel(
        4,
        2,
        tuple(
            DemError(
                error.probability,
                tuple(permutation[d] for d in error.detectors),
                error.observables,
            )
            for error in _dem().errors
        ),
    )
    permuted = SpacetimePredecoder.from_dem(
        dem, grid_shape=(2, 3, 3), sites=tuple(sites), kinds=tuple(kinds)
    )
    permuted.encoder.load_state_dict(model.encoder.state_dict())
    permuted.network.load_state_dict(model.network.state_dict())
    bits = _rows()
    reordered = torch.empty_like(bits)
    reordered[:, permutation] = bits
    assert torch.equal(model.encode_input(bits), permuted.encode_input(reordered))
    # The canonical DEM sorts mechanisms after detector IDs change. Align by
    # each fixture mechanism's unique prior, rather than bypassing that order.
    columns = [
        next(
            i
            for i, changed in enumerate(dem.errors)
            if changed.probability == e.probability
        )
        for e in _dem().errors
    ]
    torch.testing.assert_close(model(bits), permuted(reordered)[:, columns])


def test_complete_hyperedge_corrections_compose_with_global_outputs() -> None:
    class FixedScores(nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return features.new_tensor([-10, 10, 10, -10])[None, :, None].expand(
                features.shape[0], -1, -1
            )

    model = _model()
    model.network = nn.Sequential(FixedScores())
    bits = _rows()
    residual, frame = model.correct(bits, threshold=0.5)
    expected_residual, expected_frame, expected_final = [], [], []
    for row in bits.tolist():
        remaining, logical = list(row), [0, 0]
        for error in _dem().errors[1:3]:
            for detector in error.detectors:
                remaining[detector] ^= 1
            for observable in error.observables:
                logical[observable] ^= 1
        expected_residual.append(remaining)
        expected_frame.append(logical)
        expected_final.append(
            [logical[0] ^ remaining[0] ^ remaining[2], logical[1] ^ remaining[3]]
        )
    assert torch.equal(residual, torch.tensor(expected_residual))
    assert torch.equal(frame, torch.tensor(expected_frame))

    def correction(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return model.correct(rows, threshold=0.5)

    def global_decode(rows: torch.Tensor) -> torch.Tensor:
        return torch.stack((rows[:, 0] ^ rows[:, 2], rows[:, 3]), dim=1)

    assert torch.equal(
        decode(bits, correction, global_decode), torch.tensor(expected_final)
    )
    inactive, zero = model.correct(bits, enabled=False)
    assert torch.equal(inactive, bits)
    assert not bool(zero.any())


def test_hyperedge_training_and_version_one_resume_remain_exact(tmp_path: Path) -> None:
    torch.manual_seed(17)
    model = _model()
    trainer = Trainer(model, seed=5)
    initial = next(model.encoder.parameters()).detach().clone()
    trainer.train(2, batch_size=8)
    assert not torch.equal(initial, next(model.encoder.parameters()))
    path = tmp_path / "synthetic.pt"
    trainer.save(path)
    state = torch.load(path, weights_only=True)
    assert set(state) == {"version", "dem", "model", "optimizer", "generator", "step"}
    assert state["version"] == 1
    expected = trainer.train(2, batch_size=8)
    restored = Trainer(_model(), seed=99)
    restored.restore(path)
    assert restored.train(2, batch_size=8) == expected
    assert restored.step == trainer.step == 4
    assert torch.equal(restored.generator.get_state(), trainer.generator.get_state())
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[key])
    assert torch.equal(trainer.model(_rows()), restored.model(_rows()))


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (2, 3),
        (2, 3, 3, 1),
        [2, 3, 3],
        (0, 3, 3),
        (2, -1, 3),
        (True, 3, 3),
        (2.0, 3, 3),
    ],
)
def test_invalid_grid_refused(shape: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError, match="grid shape"):
        SpacetimePredecoder.from_dem(
            _dem(), grid_shape=shape, sites=(13, 1, 17, 6), kinds=(1, 2, 1, 2)
        )


@pytest.mark.parametrize(
    "sites", [(1, 1, 2, 3), (1, 2, 3), (-1, 1, 2, 3), (18, 1, 2, 3), (True, 1, 2, 3)]
)
def test_missing_colliding_or_out_of_grid_sites_refused(sites: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="sites"):
        SpacetimePredecoder.from_dem(
            _dem(), grid_shape=(2, 3, 3), sites=sites, kinds=(1, 2, 1, 2)
        )


@pytest.mark.parametrize(
    "kinds", [(1, 2, 1), (0, 2, 1, 2), (3, 2, 1, 2), (True, 2, 1, 2)]
)
def test_incomplete_or_invalid_check_kinds_refused(kinds: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="kinds"):
        SpacetimePredecoder.from_dem(
            _dem(), grid_shape=(2, 3, 3), sites=(13, 1, 17, 6), kinds=kinds
        )


@pytest.mark.parametrize(
    "errors",
    [
        (),
        (DemError(0.01, (), (0,)),),
        (DemError(0.5, (0,), ()),),
        (DemError(0.01, (0,), ()), DemError(0.02, (0,), ())),
        (DemError(0.01, (0,), (), 1), DemError(0.02, (1,), (), 1)),
    ],
)
def test_existing_dem_refusals_are_reused(errors: tuple[DemError, ...]) -> None:
    with pytest.raises(ValueError):
        SpacetimePredecoder.from_dem(
            DetectorErrorModel(4, 2, errors),
            grid_shape=(2, 3, 3),
            sites=(13, 1, 17, 6),
            kinds=(1, 2, 1, 2),
        )
