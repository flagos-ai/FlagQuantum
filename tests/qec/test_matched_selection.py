"""Reduced calibration fixtures test composition, never scientific benefit."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.qec import continued_assessment as old_shards
from examples.qec import diagnose_predecoder as archive
from examples.qec import ising_layout
from examples.qec import matched_checkpoints as checkpoints
from examples.qec import matched_initialization as initialization
from examples.qec import matched_selection as selection
from examples.qec import matched_training as training
from examples.qec import matched_validation as validation
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
        ising_layout,
        "build_model",
        lambda _: (
            template,
            {
                "layout_sha256": "a" * 64,
                "circuit_sha256": hashlib.sha256(
                    b"matched synthetic circuit"
                ).hexdigest(),
            },
        ),
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


class PhysicalFixture:
    num_detectors = 4
    num_observables = 1

    def __init__(self, bad_second: bool = False) -> None:
        self.compiled: list[int] = []
        self.calls: list[int] = []
        self.bad_second = bad_second
        self.bits = torch.tensor(
            [[int(i >> j & 1) for j in range(4)] for i in range(8)], dtype=torch.int8
        )
        self.truth = (self.bits[:, 0] ^ self.bits[:, 2]).reshape(-1, 1)

    def __str__(self) -> str:
        return "matched synthetic circuit"

    def compile_detector_sampler(self, *, seed: int) -> PhysicalFixture:
        self.compiled.append(seed)
        return self

    def sample(
        self, shots: int, *, separate_observables: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert separate_observables
        start = sum(self.calls)
        self.calls.append(shots)
        if self.bad_second and len(self.calls) == 2:
            raise ValueError("synthetic capture failure")
        return (
            self.bits[start : start + shots].clone(),
            self.truth[start : start + shots].clone(),
        )


def global_fixture(bits: torch.Tensor) -> torch.Tensor:
    return (bits[:, 0].to(torch.int8) ^ bits[:, 1].to(torch.int8)).reshape(-1, 1)


@pytest.fixture
def calibrated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inputs: tuple[Path, Path, str]
) -> tuple[Path, Path, str, Path, str]:
    initial, trained, digest = inputs
    output = tmp_path / "calibration"
    selection.calibrate(tmp_path, initial, trained, output, training_sha256=digest)
    monkeypatch.setattr(validation, "VALIDATION_SHOTS", 8)
    monkeypatch.setattr(validation, "CHUNK", 4)
    return initial, trained, digest, output, archive.digest(output / "manifest.json")


def test_validation_single_sampler_all60_trials_match_scalar_xor_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibrated: tuple[Path, Path, str, Path, str],
) -> None:
    initial, trained, digest, cal, cal_hash = calibrated
    models, _, _ = training.restore_final(
        tmp_path, initial, trained, training_sha256=digest
    )
    circuit = PhysicalFixture()
    monkeypatch.setattr(Predecoder, "sample", forbidden)
    monkeypatch.setattr(Trainer, "train", forbidden)
    output = tmp_path / "choices"
    result = validation.validate_and_freeze(
        tmp_path,
        initial,
        trained,
        cal,
        output,
        training_sha256=digest,
        calibration_sha256=cal_hash,
        circuit=circuit,
        global_decode=global_fixture,
        device=torch.device("cpu"),
    )
    assert circuit.compiled == [21000011] and circuit.calls == [4, 4]
    assert len(result["paths"]) == 12
    baseline = global_fixture(circuit.bits).flatten().tolist()
    truth = circuit.truth.flatten().tolist()
    for index, (_, architecture, trainer) in enumerate(models):
        logits = trainer.model(circuit.bits).detach().tolist()
        h, logical = trainer.model.h.tolist(), trainer.model.logical.tolist()
        for row in result["paths"][2 * index : 2 * index + 2]:
            assert row["architecture"] == architecture and row["step"] == 4
            oracle = []
            for trial in row["selection"]["trials"]:
                predicted = []
                for bits, scores in zip(circuit.bits.tolist(), logits, strict=True):
                    active = [
                        int(
                            1 / (1 + math.exp(-x / row["temperature"]))
                            >= trial["threshold"]
                        )
                        for x in scores
                    ]
                    residual = [
                        int(b)
                        ^ (
                            sum(int(v) * a for v, a in zip(column, active, strict=True))
                            % 2
                        )
                        for b, column in zip(bits, h, strict=True)
                    ]
                    frame = (
                        sum(int(v) * a for v, a in zip(logical[0], active, strict=True))
                        % 2
                    )
                    predicted.append(residual[0] ^ residual[1] ^ frame)
                first = [b != t for b, t in zip(baseline, truth, strict=True)]
                second = [b != t for b, t in zip(predicted, truth, strict=True)]
                expected = dict(
                    shots=8,
                    baseline_failures=sum(first),
                    hybrid_failures=sum(second),
                    wins=sum(a and not b for a, b in zip(first, second, strict=True)),
                    losses=sum(b and not a for a, b in zip(first, second, strict=True)),
                )
                assert trial["counts"] == expected
                oracle.append((sum(second), -trial["threshold"]))
            assert row["selection"]["threshold"] == -min(oracle)[1]
    assert (output / "choices.sha256").read_text().strip() == archive.digest(
        output / "choices.json"
    )
    with pytest.raises(FileExistsError):
        validation.validate_and_freeze(
            tmp_path,
            initial,
            trained,
            cal,
            output,
            training_sha256=digest,
            calibration_sha256=cal_hash,
            circuit=circuit,
            global_decode=global_fixture,
            device=torch.device("cpu"),
        )
    assert circuit.calls == [4, 4]


@pytest.mark.parametrize(
    "damage", ["external", "order", "temperature", "cal_file", "model_state", "circuit"]
)
def test_validation_refuses_inputs_before_compiling_sampler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibrated: tuple[Path, Path, str, Path, str],
    damage: str,
) -> None:
    initial, trained, digest, cal, cal_hash = calibrated
    circuit = PhysicalFixture()
    if damage == "external":
        cal_hash = "a" * 64
    elif damage == "cal_file":
        (cal / "seed-11-chunk-0000.pt").write_bytes(b"damaged")
    elif damage == "circuit":
        circuit.num_observables = 2
    else:
        manifest = cal / "manifest.json"
        raw = json.loads(manifest.read_text())
        if damage == "order":
            raw["records"].reverse()
        if damage == "temperature":
            raw["records"][0]["fit"]["temperature"] = float("nan")
        if damage == "model_state":
            raw["models"][0]["training_state_sha256"] = "a" * 64
        manifest.write_text(json.dumps(raw))
        cal_hash = archive.digest(manifest)
    monkeypatch.setattr(Predecoder, "sample", forbidden)
    monkeypatch.setattr(Trainer, "train", forbidden)
    output = tmp_path / "choices"
    with pytest.raises(ValueError):
        validation.validate_and_freeze(
            tmp_path,
            initial,
            trained,
            cal,
            output,
            training_sha256=digest,
            calibration_sha256=cal_hash,
            circuit=circuit,
            global_decode=global_fixture,
            device=torch.device("cpu"),
        )
    assert circuit.compiled == [] and not output.exists()


@pytest.mark.parametrize(
    "damage", ["capture", "decode", "shard", "calibration", "source", "state"]
)
def test_validation_partial_failures_never_freeze_choices_or_resample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibrated: tuple[Path, Path, str, Path, str],
    damage: str,
) -> None:
    initial, trained, digest, cal, cal_hash = calibrated
    output = tmp_path / "choices"
    circuit = PhysicalFixture(bad_second=damage == "capture")
    models, files, checked = training.restore_final(
        tmp_path, initial, trained, training_sha256=digest
    )
    monkeypatch.setattr(
        training, "restore_final", lambda *a, **k: (models, files, checked)
    )
    monkeypatch.setattr(Predecoder, "sample", forbidden)

    def changed(bits: torch.Tensor) -> torch.Tensor:
        if damage == "decode":
            return torch.full((len(bits), 1), 2)
        if damage == "shard":
            (output / "validation/chunk-0001.pt").write_bytes(b"changed")
        if damage == "calibration":
            (cal / "manifest.json").write_bytes(b"changed")
        if damage == "source":
            monkeypatch.setattr(validation, "_sources", lambda: {"changed": "b" * 64})
        if damage == "state":
            models[0][2].generator.manual_seed(99)
        return global_fixture(bits)

    with pytest.raises(ValueError):
        validation.validate_and_freeze(
            tmp_path,
            initial,
            trained,
            cal,
            output,
            training_sha256=digest,
            calibration_sha256=cal_hash,
            circuit=circuit,
            global_decode=changed,
            device=torch.device("cpu"),
        )
    assert (output / "plan.json").exists() and not (output / "choices.json").exists()
    before = list(circuit.calls)
    with pytest.raises(FileExistsError):
        validation.validate_and_freeze(
            tmp_path,
            initial,
            trained,
            cal,
            output,
            training_sha256=digest,
            calibration_sha256=cal_hash,
            circuit=circuit,
            global_decode=changed,
            device=torch.device("cpu"),
        )
    assert circuit.calls == before


def test_registered_validation_constants_and_unchanged_old_defaults() -> None:
    assert (
        validation.VALIDATION_SEED,
        validation.VALIDATION_SHOTS,
        validation.CHUNK,
    ) == (21000011, 131072, 4096)
    assert (
        old_shards.VALIDATION_SEED,
        old_shards.VALIDATION_SHOTS,
        old_shards.CHUNK,
    ) == (15000011, 131072, 4096)
