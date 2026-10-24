"""Exact XOR and feature-label oracles for private CPU diagnostics."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import pytest
import torch

from flagquantum.qec.decoders._ai.diagnostics import (
    FeatureTable,
    diagnose_equivalence,
)
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.training import Trainer
from flagquantum.qec.dem.model import DetectorErrorModel

pytestmark = pytest.mark.integration

CYCLE = "error(0.1) D0 D1\nerror(0.1) D1 D2\nerror(0.1) D0 D2\n"


def fixture_dem(text: str) -> DetectorErrorModel:
    declarations = "detector D0\ndetector D1\ndetector D2\n"
    if "L0" in text:
        declarations += "logical_observable L0\n"
    return DetectorErrorModel.from_stim_text(text + declarations)


def states(mechanisms: int) -> torch.Tensor:
    return torch.tensor(list(itertools.product((0, 1), repeat=mechanisms)))


def project(
    bits: tuple[int, ...], dem: DetectorErrorModel
) -> tuple[set[int], set[int]]:
    detectors: set[int] = set()
    logical: set[int] = set()
    for fired, error in zip(bits, dem.errors, strict=True):
        if fired:
            detectors.symmetric_difference_update(error.detectors)
            logical.symmetric_difference_update(error.observables)
    return detectors, logical


@pytest.mark.parametrize(
    "text",
    [
        "error(0.1) D0 L0\nerror(0.2) D1\n",
        CYCLE,
        "error(0.1) D0 D1 L0\nerror(0.1) D1 D2 L0\nerror(0.1) D0 D2\n",
        "error(0.1) D0 D1 L0\nerror(0.1) D1 D2\nerror(0.1) D0 D2\n",
        "error(0.1) D0 D1 D2 L0\nerror(0.2) D0 D2\nerror(0.3) D1\n",
    ],
)
def test_all_latent_pairs_against_independent_signature_xor(text: str) -> None:
    dem = fixture_dem(text)
    latent = list(itertools.product((0, 1), repeat=len(dem.errors)))
    pairs = list(itertools.product(latent, repeat=2))
    target = torch.tensor([a for a, _ in pairs])
    candidate = torch.tensor([b for _, b in pairs])
    result = diagnose_equivalence(
        dem.detector_error_matrix(), dem.observables_flips_matrix(), target, candidate
    )
    projected = [(project(a, dem), project(b, dem)) for a, b in pairs]
    assert result.shots == len(pairs)
    assert result.latent_equal == sum(a == b for a, b in pairs)
    assert result.detector_equal == sum(a[0] == b[0] for a, b in projected)
    assert result.detector_and_logical_equal == sum(a == b for a, b in projected)
    assert result.detector_equal_logical_different == sum(
        a[0] == b[0] and a[1] != b[1] for a, b in projected
    )
    assert result.latent_mismatch_bits == sum(
        sum(x != y for x, y in zip(a, b, strict=True)) for a, b in pairs
    )


def test_cycle_kernel_with_a_logical_flip_is_not_equivalence() -> None:
    target = torch.zeros(1, 3)
    candidate = torch.ones(1, 3)
    for suffix, expected_joint in (("", 1), (" L0", 0)):
        dem = fixture_dem(CYCLE.replace("D0 D1", "D0 D1" + suffix))
        result = diagnose_equivalence(
            dem.detector_error_matrix(),
            dem.observables_flips_matrix(),
            target,
            candidate,
        )
        assert result.latent_equal == 0
        assert result.detector_equal == 1
        assert result.detector_and_logical_equal == expected_joint
        assert result.detector_equal_logical_different == 1 - expected_joint


def test_exact_feature_groups_preserve_conflicting_labels_across_chunks() -> None:
    dem = fixture_dem(CYCLE)
    model = Predecoder(dem)
    latent = states(3)
    # Independent signature oracle constructs the syndrome, not a diagnostic matrix.
    detectors = torch.tensor(
        [
            [int(i in project(tuple(row.tolist()), dem)[0]) for i in range(3)]
            for row in latent
        ]
    )
    features = model._features(detectors)
    logits = model(detectors)
    table = FeatureTable()
    for start in (0, 3, 6):
        table.update(
            features[start : start + 3],
            logits[start : start + 3],
            latent[start : start + 3],
        )
    summary = table.summarize()
    # This is uniform exhaustive enumeration, not the p=0.1 Bernoulli posterior.
    assert summary.observations == 24
    assert [g.observations for g in summary.groups] == [6, 12, 6]
    assert [g.positive_labels for g in summary.groups] == [3, 6, 3]
    assert all(g.positive_fraction == 0.5 for g in summary.groups)
    assert summary.singleton_groups == 0
    assert summary.mixed_label_groups == 3
    assert summary.mixed_label_observations == 24
    assert summary.empirical_conditional_bce == pytest.approx(math.log(2))
    for group in summary.groups:
        selected = (features == torch.tensor(group.features)).all(dim=-1)
        assert group.mean_logit == pytest.approx(
            float(logits[selected].detach().double().mean())
        )
        assert group.mean_probability == pytest.approx(
            float(logits[selected].detach().double().sigmoid().mean())
        )
        assert group.minimum_logit <= group.mean_logit <= group.maximum_logit
    whole = FeatureTable()
    whole.update(features, logits, latent)
    assert (
        whole.summarize().empirical_conditional_bce == summary.empirical_conditional_bce
    )
    assert all(p.grad is None for p in model.parameters())


def test_adjacent_float_features_are_not_rounded_into_one_group() -> None:
    adjacent = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    features = torch.tensor([[[0.0, 1.0, 1.0], [0.0, 1.0, adjacent]]])
    table = FeatureTable()
    table.update(features, torch.zeros(1, 2), torch.tensor([[0, 1]]))
    summary = table.summarize()
    assert len(summary.groups) == summary.singleton_groups == 2
    assert summary.mixed_label_groups == summary.mixed_label_observations == 0
    assert summary.empirical_conditional_bce == 0


def test_feature_extraction_preserves_logits_correction_and_checkpoint(
    tmp_path: Path,
) -> None:
    model = Predecoder(fixture_dem(CYCLE))
    bits = states(3)
    # Reference expression is the pre-existing forward formula.
    active = bits.float() @ model.h
    old_features = torch.stack(
        (
            active,
            model.h.sum(0).expand_as(active),
            torch.logit(model.prior).expand_as(active),
        ),
        dim=-1,
    )
    assert torch.equal(model(bits), model.network(old_features).squeeze(-1))
    before = model.correct(bits)
    path = tmp_path / "checkpoint.pt"
    Trainer(model).save(path)
    raw = torch.load(path, weights_only=True)
    assert set(raw) == {"version", "dem", "model", "optimizer", "generator", "step"}
    restored = Predecoder(fixture_dem(CYCLE))
    Trainer(restored).restore(path)
    assert torch.equal(model(bits), restored(bits))
    assert all(
        torch.equal(a, b) for a, b in zip(before, restored.correct(bits), strict=True)
    )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0, 0.5])
def test_invalid_latents_fail_before_projection(invalid: float) -> None:
    with pytest.raises(ValueError, match="binary"):
        diagnose_equivalence(
            torch.ones(1, 1),
            torch.empty(0, 1),
            torch.tensor([[invalid]]),
            torch.zeros(1, 1),
        )


@pytest.mark.parametrize("target", [torch.empty(0, 1), torch.zeros(1, 2)])
def test_empty_or_incompatible_latents_are_refused(target: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="shape"):
        diagnose_equivalence(torch.ones(1, 1), torch.empty(0, 1), target, target)


def test_device_and_complex_projection_inputs_are_refused() -> None:
    with pytest.raises(ValueError, match="CPU"):
        diagnose_equivalence(
            torch.ones(1, 1, device="meta"),
            torch.empty(0, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
        )
    with pytest.raises(ValueError, match="binary"):
        diagnose_equivalence(
            torch.ones(1, 1, dtype=torch.complex64),
            torch.empty(0, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
        )


def test_bad_feature_update_does_not_change_accumulated_evidence() -> None:
    table = FeatureTable()
    with pytest.raises(ValueError, match="at least one"):
        table.summarize()
    table.update(torch.zeros(1, 1, 3), torch.zeros(1, 1), torch.zeros(1, 1))
    before = table.summarize()
    for features in (
        torch.zeros(1, 1, 4),
        torch.full((1, 1, 3), math.nan),
        torch.ones(1, 1, 3, dtype=torch.int8),
    ):
        with pytest.raises(ValueError, match="finite exact"):
            table.update(features, torch.zeros(1, 1), torch.zeros(1, 1))
    with pytest.raises(ValueError, match="CPU"):
        table.update(
            torch.zeros(1, 1, 3, device="meta"), torch.zeros(1, 1), torch.zeros(1, 1)
        )
    assert table.summarize() == before
