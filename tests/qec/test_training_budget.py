"""Tiny-DEM continuation and independent held-out scoring, not study outcomes."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from examples.qec import diagnose_predecoder as archive
from examples.qec import study_training_budget as study
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.training import Trainer
from flagquantum.qec.dem.model import DetectorErrorModel

pytestmark = pytest.mark.integration


def tiny_trainer() -> Trainer:
    dem = DetectorErrorModel.from_stim_text(
        "error(0.1) D0 D1 L0\nerror(0.2) D1 D2\nerror(0.1) D0 D2\n"
        "detector D0\ndetector D1\ndetector D2\n"
    )
    torch.manual_seed(71)
    return Trainer(Predecoder(dem), seed=91)


def assert_same(a: object, b: object) -> None:
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for k in a:
            assert_same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            assert_same(x, y)
    else:
        assert a == b


def test_resume_preserves_full_trajectory_and_original_predictions(
    tmp_path: Path,
) -> None:
    trainer = tiny_trainer()
    trainer.train(3, batch_size=8)
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    resumed = tiny_trainer()
    resumed.restore(path)
    probe = trainer.model.sample(8, torch.Generator().manual_seed(9)).sample.detectors
    assert torch.equal(trainer.model(probe), resumed.model(probe))
    assert_same(trainer.model.correct(probe), resumed.model.correct(probe))
    expected = trainer.train(5, batch_size=8)
    assert resumed.train(2, batch_size=8) == expected[:2]
    resumed.save(path)
    final = tiny_trainer()
    final.restore(path)
    assert final.train(3, batch_size=8) == expected[2:]
    assert_same(trainer.model.state_dict(), final.model.state_dict())
    assert_same(trainer.optimizer.state_dict(), final.optimizer.state_dict())
    assert_same(trainer.generator.get_state(), final.generator.get_state())
    assert trainer.step == final.step == 8


def test_scoring_is_paired_chunk_invariant_and_does_not_touch_training() -> None:
    trainer = tiny_trainer()
    trainer.train(2, batch_size=8)
    model = deepcopy(trainer.model.state_dict())
    optimizer = deepcopy(trainer.optimizer.state_dict())
    generator = trainer.generator.get_state().clone()
    before_mode = trainer.model.training
    whole = study.score(trainer, 11, shots=32)
    chunked = study.score(trainer, 11, shots=32, chunk=1, reference=whole)
    assert whole.latent_bytes_sha256 == chunked.latent_bytes_sha256
    assert whole.per_shot_bce == pytest.approx(chunked.per_shot_bce, abs=1e-7)
    assert whole.raw_loss.mean_bce == pytest.approx(chunked.raw_loss.mean_bce)
    assert max(abs(x) for x in chunked.paired_bce_difference) < 1e-7
    for a, b in zip(whole.groups, chunked.groups, strict=True):
        actual: dict[str, object] = dict(a)
        expected: dict[str, object] = dict(b)
        for key, value in actual.items():
            if isinstance(value, float):
                assert value == pytest.approx(expected[key])
            else:
                assert value == expected[key]
    assert whole.controls == chunked.controls
    assert whole.zero_probability_bce is None
    assert whole.zero_probability_bce_is_infinite
    assert_same(trainer.model.state_dict(), model)
    assert_same(trainer.optimizer.state_dict(), optimizer)
    assert_same(trainer.generator.get_state(), generator)
    assert trainer.model.training == before_mode
    trainer.train(1, batch_size=8)
    changed = study.score(trainer, 11, shots=32, reference=whole)
    assert changed.per_shot_bce != whole.per_shot_bce
    assert changed.paired_difference_mean == pytest.approx(
        sum(
            x - y for x, y in zip(changed.per_shot_bce, whole.per_shot_bce, strict=True)
        )
        / 32
    )
    with pytest.raises(ValueError, match="paired scoring stream"):
        study.score(trainer, 22, shots=32, reference=whole)


@pytest.fixture
def continuation_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Synthetic counter metadata for refusal tests; no 200-step claim."""
    models = []
    for seed in archive.SEEDS:
        trainer = tiny_trainer()
        trainer.train(1, batch_size=8)
        trainer.step = 200
        for value in trainer.optimizer.state.values():
            value["step"].fill_(200)
        trainer.save(tmp_path / f"native-{seed}.pt")
        models.append(
            archive.ArchivedModel(
                seed,
                trainer.model,
                "fixture",
                "fixture",
                {"temperature": 1.0, "threshold": 1.0},
            )
        )
    monkeypatch.setattr(archive, "load_inputs", lambda _: (models, {}))
    return tmp_path


@pytest.mark.parametrize(
    "failure", ["lr", "counter", "moment", "missing_state", "model"]
)
def test_late_seed_bad_optimizer_aborts_before_any_update(
    continuation_inputs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    path = continuation_inputs / "native-33.pt"
    raw = torch.load(path, weights_only=True)
    if failure == "lr":
        raw["optimizer"]["param_groups"][0]["lr"] = 0.01
    elif failure == "counter":
        next(iter(raw["optimizer"]["state"].values()))["step"].fill_(199)
    elif failure == "moment":
        next(iter(raw["optimizer"]["state"].values()))["exp_avg"].fill_(float("nan"))
    elif failure == "missing_state":
        raw["optimizer"]["state"].clear()
    else:
        raw["model"]["prior"].fill_(float("nan"))
    torch.save(raw, path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("bad input reached optimizer or sampling")

    monkeypatch.setattr(Trainer, "train", forbidden)
    monkeypatch.setattr(Predecoder, "sample", forbidden)
    output = tmp_path / "fresh-output"
    with pytest.raises(ValueError):
        study.run(continuation_inputs, output)
    assert not output.exists()


def test_all_three_continuation_states_restore_before_collection(
    continuation_inputs: Path,
) -> None:
    trainers, _ = study.load_trainers(continuation_inputs)
    assert [s for s, _ in trainers] == list(archive.SEEDS)
    assert all(t.step == 200 for _, t in trainers)
    assert all(
        float(v["step"]) == 200 for _, t in trainers for v in t.optimizer.state.values()
    )


def test_existing_output_refuses_before_input_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object) -> None:
        raise AssertionError("existing output reached input access")

    monkeypatch.setattr(study, "load_trainers", forbidden)
    with pytest.raises(FileExistsError):
        study.run(tmp_path / "missing", tmp_path)


@pytest.mark.parametrize("shots,chunk", [(0, 32), (32, 0), (32, 64), (31, 32)])
def test_invalid_scoring_chunks_are_rejected(shots: int, chunk: int) -> None:
    with pytest.raises(ValueError, match="chunk"):
        study.score(tiny_trainer(), 11, shots=shots, chunk=chunk)


def test_complete_tiny_fixture_retains_every_horizon_and_update(
    continuation_inputs: Path,
    tmp_path: Path,
) -> None:
    """Execute the full fixed schedule on a three-mechanism synthetic DEM only."""
    output = tmp_path / "tiny-study"
    study.run(continuation_inputs, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["horizons"] == [200, 600, 2000]
    assert manifest["new_primary_test_shots"] == 0
    for seed in archive.SEEDS:
        updates = [
            json.loads(line)
            for line in (output / f"seed-{seed}-training.jsonl")
            .read_text()
            .splitlines()
        ]
        assert [r["step"] for r in updates] == list(range(201, 2001))
        reports = [
            json.loads((output / f"seed-{seed}-step-{step}.json").read_text())
            for step in study.HORIZONS
        ]
        assert len({r["latent_bytes_sha256"] for r in reports}) == 1
        assert all(r["shots"] == 4096 for r in reports)
        assert reports[0]["bce_ratio_to_step200"] == 1
        for step in study.HORIZONS:
            raw = torch.load(
                output / f"native-{seed}-step-{step}.pt", weights_only=True
            )
            assert raw["step"] == step
            assert all(
                float(v["step"]) == step for v in raw["optimizer"]["state"].values()
            )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["consistent_descriptive_training_response"] == all(
        summary["per_seed_descriptive_response"].values()
    )
    assert not summary["logical_quality_claim_allowed"]
