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

from examples.qec import compare_ising as comparison
from examples.qec import continued_assessment as old_shards
from examples.qec import continued_parity as old_parity
from examples.qec import diagnose_predecoder as archive
from examples.qec import ising_layout
from examples.qec import matched_checkpoints as checkpoints
from examples.qec import matched_initialization as initialization
from examples.qec import matched_parity as probes
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
                "dem_sha256": hashlib.sha256(template.dem_text.encode()).hexdigest(),
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


@pytest.fixture
def decisions(
    tmp_path: Path, calibrated: tuple[Path, Path, str, Path, str]
) -> tuple[Path, Path, Path, Path, str]:
    initial, trained, digest, cal, cal_hash = calibrated
    output = tmp_path / "choices"
    validation.validate_and_freeze(
        tmp_path,
        initial,
        trained,
        cal,
        output,
        training_sha256=digest,
        calibration_sha256=cal_hash,
        circuit=PhysicalFixture(),
        global_decode=global_fixture,
        device=torch.device("cpu"),
    )
    return initial, trained, cal, output, archive.digest(output / "choices.json")


class ProbeFixture(PhysicalFixture):
    def __init__(self) -> None:
        super().__init__()
        self.bits = self.bits.repeat(8, 1)
        self.truth = self.truth.repeat(8, 1)


def test_frozen_choices_single_probe_peer_replay_and_parity_oracles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decisions: tuple[Path, Path, Path, Path, str],
) -> None:
    initial, trained, cal, choices, sha = decisions
    monkeypatch.setattr(Predecoder, "sample", forbidden)
    monkeypatch.setattr(Trainer, "train", forbidden)
    models, raw = validation.restore_choices(
        tmp_path, initial, trained, cal, choices, expected_sha256=sha
    )
    assert [(s, a, t.step) for s, a, t in models] == [
        (s, a, 4) for s in (11, 22, 33) for a in ("cnn", "mlp")
    ]
    circuit = ProbeFixture()
    output = tmp_path / "probes"
    record = probes.capture_probes(
        tmp_path,
        initial,
        trained,
        cal,
        choices,
        output,
        expected_choices_sha256=sha,
        circuit=circuit,
    )
    assert circuit.compiled == [23000011] and circuit.calls == [64]
    assert record.purpose == "matched_physical_probe"
    bits = probes.load_probes(
        output,
        expected_manifest_sha256=archive.digest(output / "manifest.json"),
        choices_sha256=sha,
        circuit_sha256=raw["inputs"]["circuit_sha256"],
    )
    assert torch.equal(bits, circuit.bits) and circuit.calls == [64]

    def identity(bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return bits, torch.zeros(len(bits), 1, dtype=torch.int8)

    pairs = dict.fromkeys(probes.PATHS, (identity, identity))
    evidence = probes.check_paths(pairs, bits, global_fixture)
    assert len(evidence) == 15
    assert all(e.synthetic_rows == 6 and e.physical_rows == 64 for e in evidence)
    assert len({e.physical_predictions_sha256 for e in evidence}) == 1
    with pytest.raises(FileExistsError):
        probes.capture_probes(
            tmp_path,
            initial,
            trained,
            cal,
            choices,
            output,
            expected_choices_sha256=sha,
            circuit=circuit,
        )
    assert circuit.calls == [64]


@pytest.mark.parametrize(
    "damage",
    [
        "external",
        "record_order",
        "missing_mode",
        "temperature",
        "threshold_tie",
        "baseline",
        "counts",
        "source",
        "shard",
        "extra",
    ],
)
def test_frozen_decision_refusals_precede_probe_sampler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decisions: tuple[Path, Path, Path, Path, str],
    damage: str,
) -> None:
    initial, trained, cal, choices, sha = decisions
    path = choices / "choices.json"
    raw = json.loads(path.read_text())
    if damage == "external":
        sha = "a" * 64
    elif damage == "shard":
        (choices / "validation/chunk-0001.pt").write_bytes(b"changed")
    elif damage == "extra":
        (choices / "extra.json").write_text("{}")
    else:
        if damage == "record_order":
            raw["paths"].reverse()
        if damage == "missing_mode":
            raw["paths"].pop()
        if damage == "temperature":
            raw["paths"][0]["temperature"] = 2.0
        if damage == "threshold_tie":
            raw["paths"][0]["selection"]["threshold"] = 0.123
        if damage == "baseline":
            raw["paths"][0]["selection"]["threshold"] = 1.0
            for trial in raw["paths"][0]["selection"]["trials"]:
                trial["counts"] = dict(
                    shots=8, baseline_failures=1, hybrid_failures=1, wins=0, losses=0
                )
        if damage == "counts":
            raw["paths"][0]["selection"]["trials"][0]["counts"]["shots"] = True
        if damage == "source":
            raw["source_sha256"] = {"changed": "a" * 64}
        path.write_text(json.dumps(raw))
        sha = archive.digest(path)
        (choices / "choices.sha256").write_text(sha + "\n")
    monkeypatch.setattr(Predecoder, "sample", forbidden)
    monkeypatch.setattr(Trainer, "train", forbidden)
    circuit = ProbeFixture()
    output = tmp_path / "probes"
    with pytest.raises(ValueError):
        probes.capture_probes(
            tmp_path,
            initial,
            trained,
            cal,
            choices,
            output,
            expected_choices_sha256=sha,
            circuit=circuit,
        )
    assert not output.exists() and circuit.compiled == []


@pytest.mark.parametrize("damage", ["nonbinary", "choice_change", "protocol_change"])
def test_partial_probe_never_writes_success_or_repeats_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decisions: tuple[Path, Path, Path, Path, str],
    damage: str,
) -> None:
    initial, trained, cal, choices, sha = decisions

    class Damaged(ProbeFixture):
        def sample(
            self, shots: int, *, separate_observables: bool
        ) -> tuple[torch.Tensor, torch.Tensor]:
            bits, truth = super().sample(
                shots, separate_observables=separate_observables
            )
            if damage == "nonbinary":
                bits[0, 0] = 2
            elif damage == "choice_change":
                (choices / "choices.json").write_text("{}")
            else:
                digest = archive.digest
                monkeypatch.setattr(
                    archive,
                    "digest",
                    lambda path: (
                        "a" * 64 if path == checkpoints.PROTOCOLS[1] else digest(path)
                    ),
                )
            return bits, truth

    circuit = Damaged()
    output = tmp_path / "probes"
    monkeypatch.setattr(Predecoder, "sample", forbidden)
    with pytest.raises(ValueError):
        probes.capture_probes(
            tmp_path,
            initial,
            trained,
            cal,
            choices,
            output,
            expected_choices_sha256=sha,
            circuit=circuit,
        )
    assert (output / "plan.json").exists() and not (output / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        probes.capture_probes(
            tmp_path,
            initial,
            trained,
            cal,
            choices,
            output,
            expected_choices_sha256=sha,
            circuit=circuit,
        )
    assert circuit.calls == [64]


@pytest.mark.parametrize("damage", ["seed", "purpose", "payload"])
def test_peer_replay_rejects_changed_probe_binding(
    tmp_path: Path, decisions: tuple[Path, Path, Path, Path, str], damage: str
) -> None:
    initial, trained, cal, choices, sha = decisions
    output = tmp_path / "probes"
    probes.capture_probes(
        tmp_path,
        initial,
        trained,
        cal,
        choices,
        output,
        expected_choices_sha256=sha,
        circuit=ProbeFixture(),
    )
    manifest = output / "manifest.json"
    raw = json.loads(manifest.read_text())
    if damage == "seed":
        raw["seed"] = 17000011
    if damage == "purpose":
        raw["purpose"] = "continued_physical_probe"
    if damage == "payload":
        (output / "physical.pt").write_bytes(b"changed")
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        probes.load_probes(
            output,
            expected_manifest_sha256=archive.digest(manifest),
            choices_sha256=sha,
            circuit_sha256=raw["circuit_sha256"],
        )


def test_registered_explicit_path_factory_and_wrong_mode_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decisions: tuple[Path, Path, Path, Path, str],
) -> None:
    # This factory fixture supplies synthetic step2000 identities only. It does
    # not claim real training at that horizon or authenticate a device/pipeline.
    initial, trained, cal, choices, sha = decisions
    models, raw = validation.restore_choices(
        tmp_path, initial, trained, cal, choices, expected_sha256=sha
    )
    for _, _, trainer in models:
        trainer.step = 2000
    for row in raw["paths"]:
        row["step"] = 2000

    class Pipeline(torch.nn.Module):
        def forward(self, bits: torch.Tensor) -> torch.Tensor:
            return torch.cat((torch.zeros(len(bits), 1, dtype=torch.int8), bits), dim=1)

    pairs = probes.path_pairs(models, raw, Pipeline(), device=torch.device("cpu"))
    assert tuple(pairs) == probes.PATHS
    for name, (first, second) in pairs.items():
        if name.startswith(("cnn-", "mlp-")):
            bits = torch.tensor([[0, 0, 0, 0], [1, 0, 1, 0]], dtype=torch.int8)
            assert all(
                torch.equal(a, b)
                for a, b in zip(first(bits), second(bits), strict=True)
            )
    raw["paths"][0]["mode"] = "unknown"
    with pytest.raises(ValueError):
        probes.path_pairs(models, raw, Pipeline(), device=torch.device("cpu"))
    assert probes.PROBE_SEED == 23000011 and len(set(probes.PATHS)) == 15


@pytest.fixture
def receipt_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tf32_disabled: None,
    decisions: tuple[Path, Path, Path, Path, str],
) -> dict[str, Any]:
    # Every hardware/file boundary below is an explicit software fake. No CUDA
    # execution, official weight load or actual host receipt occurs in this test.
    initial, trained, cal, choices, sha = decisions
    probe = tmp_path / "probe"
    probes.capture_probes(
        tmp_path,
        initial,
        trained,
        cal,
        choices,
        probe,
        expected_choices_sha256=sha,
        circuit=ProbeFixture(),
    )
    pipeline = torch.nn.Linear(1, 1)
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "manifest.json").write_text(
        json.dumps({"torch": str(torch.__version__)})
    )
    monkeypatch.setattr(archive, "ARCHIVE", metadata)
    weight = tmp_path / "synthetic.safetensors"
    weight.write_bytes(b"not official weights: software fixture only")
    digest = archive.digest
    monkeypatch.setattr(
        archive,
        "digest",
        lambda p: comparison.FAST_SHA256 if p == weight else digest(p),
    )
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    calls: list[object] = []

    def hardware(image: str) -> dict[str, object]:
        calls.append(image)
        return {
            "device": "cuda:0",
            "name": "A800 software fixture",
            "image_id": image,
            "tf32": False,
            "autocast": False,
            "deterministic_algorithms": True,
            "nvidia_smi_uuid_name_driver": f"GPU-abcdef01-0000-0000-0000-{len(calls):012d},A800,fixture",
        }

    monkeypatch.setattr(comparison, "cuda_metadata", hardware)
    monkeypatch.setattr(
        comparison,
        "reference",
        lambda *a, **k: comparison.Reference(
            ProbeFixture(), pipeline, {}, {"sha256": comparison.FAST_SHA256}
        ),
    )
    models, raw = validation.restore_choices(
        tmp_path, initial, trained, cal, choices, expected_sha256=sha
    )
    dem = DetectorErrorModel.from_stim_text(models[0][2].model.dem_text)
    monkeypatch.setattr(comparison, "training_dem", lambda _: dem)
    monkeypatch.setattr(comparison, "global_decoder", lambda _: global_fixture)

    def identity(bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return bits, torch.zeros(len(bits), 1, dtype=torch.int8)

    def pairs(*args: Any, device: torch.device) -> dict[str, tuple[Any, Any]]:
        assert device == torch.device("cuda:0")
        return dict.fromkeys(probes.PATHS, (identity, identity))

    monkeypatch.setattr(probes, "path_pairs", pairs)
    monkeypatch.setattr(ProbeFixture, "compile_detector_sampler", forbidden)
    monkeypatch.setattr(Predecoder, "sample", forbidden)
    monkeypatch.setattr(Trainer, "train", forbidden)

    def record(host: str, output: Path) -> dict[str, Any]:
        return probes.record_device_receipt(
            tmp_path,
            initial,
            trained,
            cal,
            choices,
            probe,
            output,
            host=host,
            choices_sha256=sha,
            probes_sha256=archive.digest(probe / "manifest.json"),
            ising_weights=weight,
        )

    def verify(receipts: dict[str, tuple[Path, str]]) -> None:
        probes.verify_two_hosts(
            tmp_path,
            initial,
            trained,
            cal,
            choices,
            probe,
            receipts,
            choices_sha256=sha,
            probes_sha256=archive.digest(probe / "manifest.json"),
        )

    return dict(
        record=record,
        verify=verify,
        pipeline=pipeline,
        calls=calls,
        initial=initial,
        trained=trained,
        cal=cal,
        choices=choices,
        choice_sha=sha,
        probes=probe,
        weights=weight,
        metadata=metadata,
    )


def test_hardware_writer_and_two_host_guards_with_explicit_software_boundaries(
    tmp_path: Path, receipt_fixture: dict[str, Any]
) -> None:
    fixture = receipt_fixture
    receipts = {}
    for host in ("a800-172", "a800-171"):
        output = tmp_path / host
        raw = fixture["record"](host, output)
        assert raw["purpose"] == "matched_cuda_bit_parity"
        assert [p["name"] for p in raw["paths"]] == list(probes.PATHS)
        assert all(
            p["physical_rows"] == 64 and p["synthetic_rows"] == 6 for p in raw["paths"]
        )
        receipts[host] = (
            output / "receipt.json",
            archive.digest(output / "receipt.json"),
        )
    fixture["verify"](receipts)
    assert len(fixture["calls"]) == 2
    with pytest.raises(FileExistsError):
        fixture["record"]("a800-172", tmp_path / "a800-172")
    assert len(fixture["calls"]) == 2


@pytest.mark.parametrize("damage", ["choices", "probe", "weights", "torch", "host"])
def test_receipt_inputs_refused_before_device_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_fixture: dict[str, Any],
    damage: str,
) -> None:
    f = receipt_fixture
    host = "a800-172"
    if damage == "choices":
        (f["choices"] / "choices.json").write_text("{}")
    if damage == "probe":
        (f["probes"] / "physical.pt").write_bytes(b"changed")
    if damage == "weights":
        digest = archive.digest
        monkeypatch.setattr(
            archive, "digest", lambda p: "b" * 64 if p == f["weights"] else digest(p)
        )
    if damage == "torch":
        (f["metadata"] / "manifest.json").write_text(json.dumps({"torch": "wrong"}))
    if damage == "host":
        host = "unregistered"
    with pytest.raises(ValueError):
        f["record"](host, tmp_path / "receipt")
    assert f["calls"] == [] and not (tmp_path / "receipt").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "cpu",
        "uuid",
        "tf32",
        "pipeline",
        "input",
        "numerical",
        "autocast",
        "fp64",
        "precision",
        "native_state",
        "extra",
        "source",
    ],
)
def test_receipt_failure_has_no_success_and_refuses_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_fixture: dict[str, Any],
    damage: str,
) -> None:
    f = receipt_fixture
    output = tmp_path / "receipt"
    if damage == "fp64":
        f["pipeline"].double()
    if damage == "native_state":
        factory = probes.path_pairs

        def changed_pairs(
            *args: Any, device: torch.device
        ) -> dict[str, tuple[Any, Any]]:
            args[0][0][2].generator.manual_seed(99)
            return factory(*args, device=device)

        monkeypatch.setattr(probes, "path_pairs", changed_pairs)
    if damage in ("cpu", "uuid", "tf32"):
        metadata = comparison.cuda_metadata

        def changed(image: str) -> dict[str, Any]:
            result = metadata(image)
            if damage == "cpu":
                result["device"] = "cpu"
            if damage == "uuid":
                result["nvidia_smi_uuid_name_driver"] = "invalid"
            if damage == "tf32":
                result["tf32"] = True
            return result

        monkeypatch.setattr(comparison, "cuda_metadata", changed)
    else:
        check = probes.check_paths

        def failed(*args: Any, **kwargs: Any) -> list[old_parity.PathEvidence]:
            evidence = check(*args, **kwargs)
            if damage == "precision":
                # Restored by the `tf32_disabled` fixture; assigning the flag
                # directly is what torch supports, and monkeypatch cannot undo it
                # under pytest 9.2.
                torch.backends.cuda.matmul.allow_tf32 = True
            if damage == "extra":
                (output / "extra.json").write_text("{}")
            if damage == "source":
                monkeypatch.setattr(probes, "_sources", lambda: {"changed": "a" * 64})
            if damage == "pipeline":
                next(f["pipeline"].parameters()).add_(1)
            if damage == "input":
                (f["cal"] / "manifest.json").write_text("{}")
            if damage == "numerical":
                raise ValueError("injected parity failure")
            if damage == "autocast":
                monkeypatch.setattr(torch, "is_autocast_enabled", lambda *a: True)
            return evidence

        monkeypatch.setattr(probes, "check_paths", failed)
    with pytest.raises(ValueError):
        f["record"]("a800-172", output)
    assert not (output / "receipt.json").exists()
    if output.exists():
        before = len(f["calls"])
        # Restore only the read-side autocast flag; a partial output still cannot retry.
        monkeypatch.setattr(torch, "is_autocast_enabled", lambda *a: False)
        with pytest.raises(FileExistsError):
            f["record"]("a800-172", output)
        assert len(f["calls"]) == before


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "hash",
        "paths",
        "counts",
        "torch",
        "pipeline",
        "protocols",
        "source_files",
        "weights",
        "training",
        "calibration",
        "repeat_uuid",
        "cpu",
    ],
)
def test_two_host_receipt_parser_refuses_rehashed_inconsistent_fixtures(
    tmp_path: Path, receipt_fixture: dict[str, Any], damage: str
) -> None:
    f = receipt_fixture
    receipts = {}
    for host in ("a800-172", "a800-171"):
        output = tmp_path / host
        f["record"](host, output)
        path = output / "receipt.json"
        receipts[host] = (path, archive.digest(path))
    if damage == "missing":
        del receipts["a800-171"]
    elif damage == "hash":
        receipts["a800-171"] = (receipts["a800-171"][0], "a" * 64)
    else:
        path = receipts["a800-171"][0]
        raw = json.loads(path.read_text())
        if damage == "paths":
            raw["paths"].reverse()
        if damage == "counts":
            raw["paths"][0]["physical_rows"] = 63
        if damage == "pipeline":
            raw["pipeline_state_sha256"] = "a" * 64
        if damage == "torch":
            raw["torch_version"] = "wrong"
        if damage == "protocols":
            raw["protocols"] = {}
        if damage == "source_files":
            raw["source_files"] = {}
        if damage == "weights":
            raw["fast_weights_sha256"] = "a" * 64
        if damage == "training":
            raw["training_sha256"] = "a" * 64
        if damage == "calibration":
            raw["calibration_sha256"] = "a" * 64
        if damage == "repeat_uuid":
            first = json.loads(receipts["a800-172"][0].read_text())
            raw["hardware"]["nvidia_smi_uuid_name_driver"] = (
                " " + first["hardware"]["nvidia_smi_uuid_name_driver"].upper()
            )
        if damage == "cpu":
            raw["hardware"]["device"] = "cpu"
        path.write_text(json.dumps(raw))
        receipts["a800-171"] = (path, archive.digest(path))
    with pytest.raises(ValueError):
        f["verify"](receipts)
