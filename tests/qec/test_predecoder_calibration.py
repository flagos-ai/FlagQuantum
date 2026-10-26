"""CPU loss accounting, calibration and validation-only decisions."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest
import torch
from torch import nn

from flagquantum.qec import DetectorErrorModel
from flagquantum.qec.decoders._ai.assessment import assess
from flagquantum.qec.decoders._ai.calibration import fit_temperature, select_threshold
from flagquantum.qec.decoders._ai.diagnostics import diagnose
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders.matching import MatchingDecodeResult

pytestmark = pytest.mark.integration


class NoCorrection:
    def decode(self, detection_events: Iterable[int]) -> MatchingDecodeResult:
        return MatchingDecodeResult((), (), 0.0)


def scored_model() -> Predecoder:
    model = Predecoder(
        DetectorErrorModel.from_stim_text(
            "error(0.2) D0 L0\ndetector D0\nlogical_observable L0\n"
        )
    )
    model.network = nn.Sequential(nn.Linear(3, 1))
    with torch.no_grad():
        model.network[0].weight.zero_()
        model.network[0].weight[0, 0] = 2
        model.network[0].bias.fill_(-0.8)
    return model


def test_loss_accounting_does_not_hide_rare_positive_labels() -> None:
    logits = torch.zeros(2, 3, requires_grad=True)
    labels = torch.tensor([[1, 0, 0], [0, 0, 0]])
    result = diagnose(logits, labels)
    assert result.total_labels == 6
    assert result.positive_labels == 1
    assert result.negative_labels == 5
    assert result.positive_bce_sum == pytest.approx(math.log(2))
    assert result.negative_bce_sum == pytest.approx(5 * math.log(2))
    assert result.mean_bce == pytest.approx(math.log(2))
    assert result.mean_probability == 0.5
    assert result.brier_score == 0.25
    assert logits.grad is None


def test_temperature_fit_and_identity_tie_are_deterministic() -> None:
    logits = torch.tensor([[-0.5, 0.5]])
    labels = torch.tensor([[0, 1]])
    result = fit_temperature(logits, labels)
    assert result.temperature == 0.25
    assert result.calibrated_mean_bce < result.raw_mean_bce
    assert result.calibrated_brier < result.raw_brier
    assert fit_temperature(torch.zeros_like(logits), labels).temperature == 1.0
    assert torch.equal(logits, torch.tensor([[-0.5, 0.5]]))


def test_validation_selects_active_corrections_and_respects_temperature() -> None:
    model = scored_model()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    raw = select_threshold(model, NoCorrection(), shots=128, seed=43)
    scaled = select_threshold(
        model, NoCorrection(), shots=128, seed=43, temperature=2.0
    )
    # The raw positive logit is 1.2; sigmoid(1.2) > 0.7, but sigmoid(0.6) < 0.7.
    assert raw.threshold == 0.7
    assert scaled.threshold == 0.5
    assert raw.trials[0].counts.hybrid_failures == 0
    assert raw.trials[-1].counts.hybrid_failures > 0
    assert len({trial.counts.baseline_failures for trial in raw.trials}) == 1
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_temperature_reaches_hybrid_assessment_without_changing_default() -> None:
    model = scored_model()
    kwargs = {"shots": 128, "seed": 43, "threshold": 0.7}
    default = assess(model, NoCorrection(), **kwargs)
    assert default == assess(model, NoCorrection(), temperature=1.0, **kwargs)
    scaled = assess(model, NoCorrection(), temperature=2.0, **kwargs)
    assert default.hybrid_failures == 0
    assert scaled.hybrid_failures == scaled.baseline_failures > 0


def test_tied_validation_failures_choose_the_larger_threshold() -> None:
    model = scored_model()
    with torch.no_grad():
        model.network[0].weight.zero_()
        model.network[0].bias.zero_()
    selected = select_threshold(model, NoCorrection(), shots=128, seed=43)
    assert selected.threshold == 1.0
    assert selected.trials[-1].counts.hybrid_failures == (
        selected.trials[-1].counts.baseline_failures
    )


@pytest.mark.parametrize("grid", [(), (0.0,), (float("nan"),), (True,), (1.0, 1.0)])
def test_invalid_temperature_grids_are_refused(grid: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="temperature"):
        fit_temperature(torch.zeros(1, 1), torch.zeros(1, 1), temperatures=grid)


@pytest.mark.parametrize(
    "grid", [(), (0.0,), (1.1,), (float("inf"),), (True,), (0.5, 0.5)]
)
def test_invalid_threshold_grids_are_refused(grid: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="threshold"):
        select_threshold(
            scored_model(), NoCorrection(), shots=8, seed=43, thresholds=grid
        )


@pytest.mark.parametrize("temperature", [0.0, float("inf"), float("nan"), True])
def test_invalid_inference_temperature_is_refused(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        scored_model().decode(torch.ones(1, 1), NoCorrection(), temperature=temperature)


@pytest.mark.parametrize(
    "logits,labels",
    [
        (torch.empty(0, 1), torch.empty(0, 1)),
        (torch.zeros(1, 2), torch.zeros(2, 1)),
        (torch.tensor([[float("nan")]]), torch.zeros(1, 1)),
        (torch.zeros(1, 1), torch.tensor([[2]])),
        (torch.ones(1, 1, dtype=torch.complex64), torch.zeros(1, 1)),
    ],
)
def test_invalid_calibration_arrays_are_refused(
    logits: torch.Tensor, labels: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        fit_temperature(logits, labels)


def test_inference_temperature_cannot_silently_overflow_logits() -> None:
    with pytest.raises(ValueError, match="nonfinite calibrated"):
        scored_model().decode(torch.ones(1, 1), NoCorrection(), temperature=1e-100)


def test_smoke_study_freezes_all_choices_before_testing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pymatching")
    from examples.qec import calibrate_predecoder as study

    events: list[str] = []
    train, validate, evaluate = (
        study.train_model,
        study.validate_model,
        study.evaluate_model,
    )

    def train_recorded(
        config: study.StudyConfig, output: Path, seed: int, kind: str
    ) -> study.TrainingRun:
        events.append("train")
        return train(config, output, seed, kind)

    def validate_recorded(
        config: study.StudyConfig, training: study.TrainingRun
    ) -> study.ValidationRun:
        events.append("validate")
        assert len(list(output.glob("*.pt"))) == 2
        return validate(config, training)

    def evaluate_recorded(
        config: study.StudyConfig, validated: study.ValidationRun
    ) -> study.ModelResult:
        events.append("test")
        return evaluate(config, validated)

    output = tmp_path / "study"
    monkeypatch.setattr(study, "train_model", train_recorded)
    monkeypatch.setattr(study, "validate_model", validate_recorded)
    monkeypatch.setattr(study, "evaluate_model", evaluate_recorded)
    monkeypatch.setattr(
        sys,
        "argv",
        ["calibrate_predecoder", "--smoke", "--output-directory", str(output)],
    )
    study.main()
    assert events == ["train", "train", "validate", "validate", "test", "test"]
    report = json.loads((output / "results.json").read_text())
    assert report["mode"] == "smoke"
    assert report["distinct_test_draws"] == 32
    assert len(report["rows"]) == 2
    for row in report["rows"]:
        assert len(row["controls"]) == 2
        assert (
            row["checkpoint_sha256"]
            == hashlib.sha256((output / row["checkpoint"]).read_bytes()).hexdigest()
        )
        for control in row["controls"]:
            assert control["test"]["shots"] == 32
            assert len(control["validation"]["trials"]) == 5
    with pytest.raises(FileExistsError):
        study.main()
