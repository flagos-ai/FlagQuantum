"""Reduced synthetic budgets test composition, never scientific benefit."""

from __future__ import annotations

import io
import json
import math
from collections import Counter
from pathlib import Path

import pytest
import torch

from examples.qec import diagnose_predecoder as archive
from examples.qec import ising_layout
from examples.qec import matched_checkpoints as checkpoints
from examples.qec import matched_initialization as initialization
from examples.qec import matched_training as training
from flagquantum.qec.decoders._ai.predecoder import Predecoder, TrainingBatch
from flagquantum.qec.decoders._ai.spacetime import SpacetimePredecoder
from flagquantum.qec.decoders._ai.training import Trainer
from flagquantum.qec.dem.model import DemError, DetectorErrorModel

pytestmark = pytest.mark.integration


def prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str, SpacetimePredecoder]:
    dem = DetectorErrorModel(
        4, 1, (DemError(0.1, (0, 2, 3), (0,)), DemError(0.04, (1,), ()))
    )
    template = SpacetimePredecoder.from_dem(
        dem, grid_shape=(2, 2, 2), sites=(0, 2, 5, 7), kinds=(1, 2, 1, 2)
    )

    def verified(source: Path) -> tuple[SpacetimePredecoder, dict[str, object]]:
        return template, {
            "layout_sha256": "a" * 64,
            "purpose": "synthetic_fixture_only",
        }

    monkeypatch.setattr(ising_layout, "build_model", verified)
    directory = tmp_path / "initial"
    initialization.initialize(tmp_path, directory)
    monkeypatch.setattr(training, "HORIZONS", (0, 2, 4))
    monkeypatch.setattr(checkpoints, "HORIZONS", (0, 2, 4))
    monkeypatch.setattr(training, "TRAIN_BATCH", 4)
    monkeypatch.setattr(training, "SCORE_SHOTS", 4)
    monkeypatch.setattr(training, "SCORE_CHUNK", 2)
    return directory, archive.digest(directory / "initialization.json"), template


def test_complete_fixture_counts_exact_draws_pair_hashes_and_scalar_score_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial, digest, _ = prepare(tmp_path, monkeypatch)
    calls: list[tuple[int, int]] = []
    sample = Predecoder.sample

    def observed(
        self: Predecoder, shots: int, generator: torch.Generator
    ) -> TrainingBatch:
        calls.append((generator.initial_seed(), shots))
        return sample(self, shots, generator)

    monkeypatch.setattr(Predecoder, "sample", observed)
    output = tmp_path / "training"
    training.run_training(tmp_path, initial, output, initialization_sha256=digest)
    raw = json.loads((output / "manifest.json").read_text())
    assert (
        raw["independent_training_draws"],
        raw["training_exposures"],
        raw["optimizer_updates"],
        raw["independent_heldout_draws"],
    ) == (48, 96, 24, 12)
    assert raw["scientific_benefit_claim_allowed"] is False
    assert len(raw["records"]) == 18 and len(raw["heldout_files"]) == 6
    assert Counter(calls) == Counter(
        {
            **{(seed + 18100000, 4): 8 for seed in (11, 22, 33)},
            **{(seed + 19100000, 2): 2 for seed in (11, 22, 33)},
        }
    )
    for seed in (11, 22, 33):
        rows = [
            json.loads(line)
            for line in (output / f"seed-{seed}-training.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [(r["architecture"], r["step"]) for r in rows] == [
            (a, s) for s in range(1, 5) for a in ("cnn", "mlp")
        ]
        for cnn, mlp in zip(rows[::2], rows[1::2], strict=True):
            assert cnn["batch_sha256"] == mlp["batch_sha256"]
        for architecture in ("cnn", "mlp"):
            own = [r for r in rows if r["architecture"] == architecture]
            assert all(
                a["batch_sha256"]["generator_after"]
                == b["batch_sha256"]["generator_before"]
                for a, b in zip(own[:-1], own[1:], strict=True)
            )
        records = [r for r in raw["records"] if r["seed"] == seed]
        assert (
            len({json.dumps(r["heldout_input_files"], sort_keys=True) for r in records})
            == 1
        )
        for record in records:
            name = f"{record['architecture']}-{seed}-step-{record['step']}"
            loss_sum, labels = 0.0, 0
            for file in record["heldout_input_files"]:
                batch = torch.load(output / f"heldout-{seed}" / file, weights_only=True)
                scores = torch.load(
                    output / (name + "-scores") / file, weights_only=True
                )
                loss_sum += sum(
                    max(float(x), 0)
                    - float(x) * float(y)
                    + math.log1p(math.exp(-abs(float(x))))
                    for x, y in zip(
                        scores["logits"].flatten().tolist(),
                        batch["labels"].flatten().tolist(),
                        strict=True,
                    )
                )
                labels += batch["labels"].numel()
            assert record["score"]["mean_bce"] == pytest.approx(
                loss_sum / labels, abs=1e-14
            )
    before = len(calls)
    with pytest.raises(FileExistsError):
        training.run_training(tmp_path, initial, output, initialization_sha256=digest)
    assert len(calls) == before


@pytest.mark.parametrize(
    "damage", ["sixth", "missing", "environment", "rng", "external_hash"]
)
def test_all_initial_inputs_checked_before_any_sample_or_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    initial, digest, _ = prepare(tmp_path, monkeypatch)
    manifest = initial / "initialization.json"
    raw = json.loads(manifest.read_text())
    if damage == "sixth":
        with (initial / "mlp-33" / "checkpoint.pt").open("ab") as stream:
            stream.write(b"bad")
    elif damage == "missing":
        raw["initial_checkpoints"].pop()
    elif damage == "environment":
        raw["torch"] = "wrong"
    elif damage == "rng":
        raw["initialization_rng_file_sha256"] = "b" * 64
    if damage in ("missing", "environment", "rng"):
        manifest.write_text(json.dumps(raw))
        digest = archive.digest(manifest)
    if damage == "external_hash":
        digest = "b" * 64

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("all six must pass before samples or updates")

    monkeypatch.setattr(Predecoder, "sample", forbidden)
    monkeypatch.setattr(Trainer, "train", forbidden)
    output = tmp_path / "training"
    with pytest.raises(ValueError):
        training.run_training(tmp_path, initial, output, initialization_sha256=digest)
    assert not output.exists()


def test_paired_mismatch_stops_before_second_optimizer_and_preserves_partial_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial, digest, _ = prepare(tmp_path, monkeypatch)
    sample = Predecoder.sample

    def mismatched(
        self: Predecoder, shots: int, generator: torch.Generator
    ) -> TrainingBatch:
        batch = sample(self, shots, generator)
        if type(self) is Predecoder and 18100000 < generator.initial_seed() < 18200000:
            return TrainingBatch(batch.sample, 1 - batch.mechanisms)
        return batch

    monkeypatch.setattr(Predecoder, "sample", mismatched)
    output = tmp_path / "training"
    with pytest.raises(ValueError, match="before update"):
        training.run_training(tmp_path, initial, output, initialization_sha256=digest)
    assert not (output / "manifest.json").exists()
    rows = (output / "seed-11-training.jsonl").read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["architecture"] == "cnn"
    with pytest.raises(FileExistsError):
        training.run_training(tmp_path, initial, output, initialization_sha256=digest)


def test_temporary_sampling_observer_is_removed_on_nonfinite_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial, digest, _ = prepare(tmp_path, monkeypatch)
    models, _, _ = training._load_initial(tmp_path, initial, digest)
    cnn, mlp = models[0][2], models[1][2]

    def bad(bits: torch.Tensor) -> torch.Tensor:
        return torch.full((len(bits), cnn.model.prior.numel()), float("nan"))

    monkeypatch.setattr(cnn.model, "forward", bad)
    with pytest.raises(ValueError, match="nonfinite"):
        training.train_pair(cnn, mlp, 2, io.StringIO())
    assert cnn.step == mlp.step == 0
    assert "sample" not in cnn.model.__dict__ and "sample" not in mlp.model.__dict__


@pytest.mark.parametrize("damage", ["extra", "hash", "mutation"])
def test_heldout_refusal_and_training_state_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    initial, digest, _ = prepare(tmp_path, monkeypatch)
    models, _, _ = training._load_initial(tmp_path, initial, digest)
    trainer = models[0][2]
    directory = tmp_path / "heldout"
    files = training._heldout(trainer.model, 11, directory)
    if damage == "extra":
        (directory / "extra").write_text("bad")
    elif damage == "hash":
        with (directory / "chunk-0000.pt").open("ab") as stream:
            stream.write(b"bad")
    else:
        original = trainer.model.forward

        def mutate(bits: torch.Tensor) -> torch.Tensor:
            logits = original(bits)
            next(trainer.model.parameters()).add_(1)
            return logits

        monkeypatch.setattr(trainer.model, "forward", mutate)
    before = trainer.generator.get_state().clone()
    with pytest.raises(ValueError):
        training._score(trainer, directory, files, tmp_path / "scores")
    assert (
        trainer.model.training
        and torch.equal(before, trainer.generator.get_state())
        and trainer.step == 0
    )


def test_source_change_after_work_never_writes_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial, digest, _ = prepare(tmp_path, monkeypatch)
    calls: list[int] = []

    def changed() -> dict[str, str]:
        calls.append(1)
        return {"source": "first" if len(calls) == 1 else "changed"}

    monkeypatch.setattr(training, "_sources", changed)
    output = tmp_path / "training"
    with pytest.raises(ValueError, match="frozen source"):
        training.run_training(tmp_path, initial, output, initialization_sha256=digest)
    assert (output / "mlp-33-step-4" / "binding.json").is_file()
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("target", [0, True, 3, 4])
def test_skipped_or_unregistered_horizon_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: int
) -> None:
    initial, digest, _ = prepare(tmp_path, monkeypatch)
    models, _, _ = training._load_initial(tmp_path, initial, digest)
    with pytest.raises(ValueError):
        training.train_pair(models[0][2], models[1][2], target, io.StringIO())


def test_production_constants_remain_frozen() -> None:
    assert training.HORIZONS == (0, 200, 2000)
    assert (training.TRAIN_BATCH, training.SCORE_SHOTS, training.SCORE_CHUNK) == (
        128,
        4096,
        32,
    )


@pytest.mark.parametrize("field", ["latent", "detectors", "truth"])
def test_wrong_actual_batch_shape_refused_before_optimizer_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    from flagquantum.qec.dem.model import DemSample

    initial, digest, _ = prepare(tmp_path, monkeypatch)
    models, _, _ = training._load_initial(tmp_path, initial, digest)
    cnn, mlp = models[0][2], models[1][2]
    original = Predecoder.sample

    def bad(self: Predecoder, shots: int, generator: torch.Generator) -> TrainingBatch:
        batch = original(self, shots, generator)
        labels = batch.mechanisms[:-1] if field == "latent" else batch.mechanisms
        detectors = (
            batch.sample.detectors[:, :-1]
            if field == "detectors"
            else batch.sample.detectors
        )
        truth = (
            batch.sample.observables[:, :0]
            if field == "truth"
            else batch.sample.observables
        )
        return TrainingBatch(DemSample(detectors, truth), labels)

    monkeypatch.setattr(Predecoder, "sample", bad)
    with pytest.raises(ValueError, match="registered shapes"):
        training.train_pair(cnn, mlp, 2, io.StringIO())
    assert cnn.step == mlp.step == 0 and "sample" not in cnn.model.__dict__
