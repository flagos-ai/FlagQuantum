"""Reduced calibration fixtures test composition, never scientific benefit."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.qec import diagnose_predecoder as archive
from examples.qec import ising_layout
from examples.qec import matched_checkpoints as checkpoints
from examples.qec import matched_initialization as initialization
from examples.qec import matched_selection as selection
from examples.qec import matched_training as training
from flagquantum.qec.decoders._ai import calibration
from flagquantum.qec.decoders._ai.predecoder import Predecoder, TrainingBatch
from flagquantum.qec.decoders._ai.spacetime import SpacetimePredecoder
from flagquantum.qec.decoders._ai.training import Trainer
from flagquantum.qec.dem.model import DemError, DetectorErrorModel

pytestmark = pytest.mark.integration


def scalar_bce(scores: list[float], labels: list[float], temperature: float) -> float:
    return sum(
        max(x / temperature, 0)
        - x / temperature * y
        + math.log1p(math.exp(-abs(x / temperature)))
        for x, y in zip(scores, labels, strict=True)
    ) / len(labels)


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, str]:
    dem = DetectorErrorModel(
        4, 1, (DemError(0.1, (0, 2, 3), (0,)), DemError(0.04, (1,), ()))
    )
    template = SpacetimePredecoder.from_dem(
        dem, grid_shape=(2, 2, 2), sites=(0, 2, 5, 7), kinds=(1, 2, 1, 2)
    )
    monkeypatch.setattr(
        ising_layout, "build_model", lambda _: (template, {"layout_sha256": "a" * 64})
    )
    initial, trained = tmp_path / "initial", tmp_path / "trained"
    initialization.initialize(tmp_path, initial)
    monkeypatch.setattr(training, "HORIZONS", (0, 2, 4))
    monkeypatch.setattr(checkpoints, "HORIZONS", (0, 2, 4))
    monkeypatch.setattr(training, "TRAIN_BATCH", 4)
    monkeypatch.setattr(training, "SCORE_SHOTS", 4)
    monkeypatch.setattr(training, "SCORE_CHUNK", 2)
    monkeypatch.setattr(selection, "CALIBRATION_SHOTS", 4)
    monkeypatch.setattr(selection, "CALIBRATION_CHUNK", 2)
    training.run_training(
        tmp_path,
        initial,
        trained,
        initialization_sha256=archive.digest(initial / "initialization.json"),
    )
    return initial, trained, archive.digest(trained / "manifest.json")


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("unexpected update or sample")


def test_shared_capture_all_six_fits_match_independent_scalar_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inputs: tuple[Path, Path, str]
) -> None:
    initial, trained, digest = inputs
    calls: list[tuple[int, int]] = []
    sample = Predecoder.sample

    def observed(
        self: Predecoder, shots: int, generator: torch.Generator
    ) -> TrainingBatch:
        calls.append((generator.initial_seed(), shots))
        return sample(self, shots, generator)

    monkeypatch.setattr(Predecoder, "sample", observed)
    monkeypatch.setattr(Trainer, "train", forbidden)
    output = tmp_path / "calibration"
    result = selection.calibrate(
        tmp_path, initial, trained, output, training_sha256=digest
    )
    assert Counter(calls) == Counter({(s + 20100000, 2): 2 for s in (11, 22, 33)})
    assert result["independent_calibration_draws"] == 12
    assert result["optimizer_updates"] == 0
    assert result["scientific_benefit_claim_allowed"] is False
    assert [(r["seed"], r["architecture"], r["step"]) for r in result["records"]] == [
        (s, a, 4) for s in (11, 22, 33) for a in ("cnn", "mlp")
    ]
    for row in result["records"]:
        scores, labels = [], []
        for name, expected in row["shared_score_files"].items():
            assert archive.digest(output / name) == expected
            batch = torch.load(output / name, weights_only=True)
            scores.extend(batch[row["architecture"] + "_logits"].flatten().tolist())
            labels.extend(batch["labels"].flatten().tolist())

        expected_t = min(
            (0.25, 0.5, 1.0, 2.0, 4.0),
            key=lambda t: (scalar_bce(scores, labels, t), abs(math.log(t)), t),
        )
        assert row["fit"]["temperature"] == expected_t
        assert row["fit"]["calibrated_mean_bce"] == pytest.approx(
            scalar_bce(scores, labels, expected_t), abs=1e-14
        )
        assert row["fit"]["raw_mean_bce"] == pytest.approx(
            scalar_bce(scores, labels, 1.0), abs=1e-14
        )
    for cnn, mlp in zip(result["records"][::2], result["records"][1::2], strict=True):
        assert cnn["shared_score_files"] == mlp["shared_score_files"]
        assert cnn["generator_before_sha256"] == mlp["generator_before_sha256"]
        assert cnn["generator_after_sha256"] == mlp["generator_after_sha256"]
    with pytest.raises(FileExistsError):
        selection.calibrate(tmp_path, initial, trained, output, training_sha256=digest)
    assert len(calls) == 6


@pytest.mark.parametrize("damage", ["external", "last_checkpoint"])
def test_complete_training_gate_precedes_calibration_draws(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inputs: tuple[Path, Path, str],
    damage: str,
) -> None:
    initial, trained, digest = inputs
    if damage == "external":
        digest = "b" * 64
    else:
        with (trained / "mlp-33-step-4/checkpoint.pt").open("ab") as stream:
            stream.write(b"damage")
    monkeypatch.setattr(Predecoder, "sample", forbidden)
    output = tmp_path / "calibration"
    with pytest.raises(ValueError):
        selection.calibrate(tmp_path, initial, trained, output, training_sha256=digest)
    assert not output.exists()


@pytest.mark.parametrize(
    "damage",
    [
        "model",
        "optimizer",
        "generator",
        "nonfinite",
        "shape",
        "source",
        "extra",
        "score_file",
        "fit",
        "trained_file",
        "initial_file",
    ],
)
def test_failure_preserves_partial_without_success_or_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inputs: tuple[Path, Path, str],
    damage: str,
) -> None:
    initial, trained, digest = inputs
    models, files, verified = training.restore_final(
        tmp_path, initial, trained, training_sha256=digest
    )
    monkeypatch.setattr(
        training, "restore_final", lambda *a, **k: (models, files, verified)
    )
    output = tmp_path / "calibration"
    forward = Predecoder.forward

    def changed(self: Predecoder, detectors: torch.Tensor) -> torch.Tensor:
        logits = forward(self, detectors)
        if damage == "model":
            next(self.parameters()).add_(1)
        if damage == "optimizer":
            models[1][2].optimizer.param_groups[0]["lr"] = 0.2
        if damage == "generator":
            models[1][2].generator.manual_seed(99)
        if damage == "nonfinite":
            return logits * float("nan")
        if damage == "shape":
            return logits[:, :1]
        if damage == "extra":
            (output / "extra.pt").write_bytes(b"extra")
        return logits

    monkeypatch.setattr(Predecoder, "forward", changed)
    if damage in ("source", "score_file", "fit", "trained_file", "initial_file"):
        fit = calibration.fit_temperature

        def changed_fit(logits: torch.Tensor, labels: torch.Tensor) -> Any:
            if damage == "source":
                monkeypatch.setattr(
                    selection, "_sources", lambda: {"changed": "c" * 64}
                )
            if damage == "score_file":
                (output / "seed-11-chunk-0000.pt").write_bytes(b"changed")
            if damage == "trained_file":
                (trained / "seed-11-training.jsonl").write_bytes(b"changed")
            if damage == "initial_file":
                (initial / "initialization.json").write_bytes(b"changed")
            if damage == "fit":
                raise ValueError("temperature numerical failure")
            return fit(logits, labels)

        monkeypatch.setattr(calibration, "fit_temperature", changed_fit)
    monkeypatch.setattr(Trainer, "train", forbidden)
    with pytest.raises(ValueError):
        selection.calibrate(tmp_path, initial, trained, output, training_sha256=digest)
    assert (output / "plan.json").exists()
    assert not (output / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        selection.calibrate(tmp_path, initial, trained, output, training_sha256=digest)


def test_fitter_tie_keeps_temperature_one_without_mutating_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inputs: tuple[Path, Path, str]
) -> None:
    initial, trained, digest = inputs
    models, files, verified = training.restore_final(
        tmp_path, initial, trained, training_sha256=digest
    )
    monkeypatch.setattr(
        training, "restore_final", lambda *a, **k: (models, files, verified)
    )
    monkeypatch.setattr(
        Predecoder, "forward", lambda self, x: torch.zeros((len(x), self.prior.numel()))
    )
    monkeypatch.setattr(
        SpacetimePredecoder,
        "forward",
        lambda self, x: torch.zeros((len(x), self.prior.numel())),
    )
    before = [training._state(t) for _, _, t in models]
    result = selection.calibrate(
        tmp_path, initial, trained, tmp_path / "calibration", training_sha256=digest
    )
    assert all(r["fit"]["temperature"] == 1 for r in result["records"])
    assert before == [training._state(t) for _, _, t in models]


def test_registered_calibration_constants() -> None:
    assert (
        selection.CALIBRATION_SHOTS,
        selection.CALIBRATION_CHUNK,
        selection.CALIBRATION_SEED_OFFSET,
    ) == (4096, 32, 20100000)
