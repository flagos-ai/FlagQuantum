"""Synthetic checkpoint refusals; no formal study data or device receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.qec import diagnose_predecoder as archive
from examples.qec import matched_checkpoints as binding
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.spacetime import SpacetimePredecoder
from flagquantum.qec.decoders._ai.training import Trainer
from flagquantum.qec.dem.model import DemError, DetectorErrorModel

pytestmark = pytest.mark.integration
LAYOUT = "a" * 64


def model(cnn: bool = True) -> Predecoder:
    torch.manual_seed(11)
    dem = DetectorErrorModel(4, 1, (DemError(0.1, (0, 2, 3), (0,)),))
    if cnn:
        return SpacetimePredecoder.from_dem(
            dem, grid_shape=(2, 2, 2), sites=(0, 2, 5, 7), kinds=(1, 2, 1, 2)
        )
    return Predecoder(dem, allow_hyperedges=True)


def save(tmp_path: Path, cnn: bool = True) -> tuple[Trainer, Path, str]:
    trainer = Trainer(model(cnn), seed=18100011)
    directory = tmp_path / "bound"
    digest = binding.save_bound(trainer, directory, seed=11, layout_sha256=LAYOUT)
    return trainer, directory, digest


def rewrite(directory: Path, field: str, value: Any) -> str:
    path = directory / "binding.json"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    return archive.digest(path)


@pytest.mark.parametrize("cnn", [False, True])
def test_original_schema_and_exact_training_resume(tmp_path: Path, cnn: bool) -> None:
    trainer, directory, digest = save(tmp_path, cnn)
    payload = torch.load(directory / "checkpoint.pt", weights_only=True)
    assert set(payload) == {"version", "dem", "model", "optimizer", "generator", "step"}
    assert payload["version"] == 1
    initial = archive.state_digest(trainer.model.state_dict())
    restored = binding.restore_bound(
        model(cnn), directory, seed=11, layout_sha256=LAYOUT, binding_sha256=digest
    )
    assert archive.state_digest(restored.model.state_dict()) == initial
    assert torch.equal(restored.generator.get_state(), trainer.generator.get_state())
    assert trainer.train(2, batch_size=4) == restored.train(2, batch_size=4)
    assert archive.state_digest(trainer.model.state_dict()) == archive.state_digest(
        restored.model.state_dict()
    )


def test_changed_layout_refused_before_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer, directory, digest = save(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("must refuse before checkpoint restore")

    monkeypatch.setattr(Trainer, "restore", forbidden)
    with pytest.raises(ValueError, match="identity"):
        binding.restore_bound(
            trainer.model,
            directory,
            seed=11,
            layout_sha256="b" * 64,
            binding_sha256=digest,
        )


@pytest.mark.parametrize(
    "change", ["binding", "checkpoint", "seed", "architecture", "protocol", "geometry"]
)
def test_changed_frozen_inputs_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    trainer, directory, digest = save(tmp_path)
    fresh = model()
    seed = 11
    if change in ("binding", "checkpoint"):
        path = directory / ("binding.json" if change == "binding" else "checkpoint.pt")
        with path.open("ab") as stream:
            stream.write(b" ")
    elif change == "seed":
        seed = 22
    elif change == "architecture":
        fresh = model(False)
    elif change == "protocol":
        protocol = tmp_path / "protocol.md"
        protocol.write_text("changed")
        monkeypatch.setattr(binding, "PROTOCOLS", (protocol,))
    elif change == "geometry":
        assert isinstance(fresh, SpacetimePredecoder)
        fresh.sites = fresh.sites.flip(0)
    before = archive.state_digest(fresh.state_dict())
    with pytest.raises(ValueError):
        binding.restore_bound(
            fresh, directory, seed=seed, layout_sha256=LAYOUT, binding_sha256=digest
        )
    assert archive.state_digest(fresh.state_dict()) == before
    assert trainer.step == 0


@pytest.mark.parametrize(
    "change", ["buffers", "generator", "step", "nonfinite", "optimizer"]
)
def test_rehashed_bad_checkpoint_is_not_trusted(tmp_path: Path, change: str) -> None:
    _, directory, _ = save(tmp_path)
    path = directory / "checkpoint.pt"
    payload = torch.load(path, weights_only=True)
    if change == "buffers":
        payload["model"]["h"].zero_()
    elif change == "generator":
        payload["generator"] = torch.Generator().manual_seed(99).get_state()
    elif change == "step":
        payload["step"] = 200
    elif change == "nonfinite":
        payload["model"]["network.0.weight"].fill_(float("nan"))
    elif change == "optimizer":
        payload["optimizer"]["param_groups"][0]["lr"] = 0.2
    torch.save(payload, path)
    digest = rewrite(directory, "checkpoint_sha256", archive.digest(path))
    fresh = model()
    before = archive.state_digest(fresh.state_dict())
    with pytest.raises(ValueError):
        binding.restore_bound(
            fresh, directory, seed=11, layout_sha256=LAYOUT, binding_sha256=digest
        )
    assert archive.state_digest(fresh.state_dict()) == before


def test_duplicate_and_partial_output_are_not_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer, directory, _ = save(tmp_path)
    with pytest.raises(FileExistsError):
        binding.save_bound(trainer, directory, seed=11, layout_sha256=LAYOUT)
    partial = tmp_path / "partial"

    def failed(path: Path) -> None:
        path.write_bytes(b"partial")
        raise RuntimeError("injected disk failure")

    monkeypatch.setattr(trainer, "save", failed)
    with pytest.raises(RuntimeError):
        binding.save_bound(trainer, partial, seed=11, layout_sha256=LAYOUT)
    assert (partial / "checkpoint.pt").read_bytes() == b"partial"
    with pytest.raises(FileExistsError):
        binding.save_bound(trainer, partial, seed=11, layout_sha256=LAYOUT)


@pytest.mark.parametrize("seed,layout", [(True, LAYOUT), (44, LAYOUT), (11, "bad")])
def test_invalid_identity_leaves_no_output(
    tmp_path: Path, seed: int, layout: str
) -> None:
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError):
        binding.save_bound(Trainer(model()), directory, seed=seed, layout_sha256=layout)
    assert not directory.exists()


def test_trained_moments_and_stream_resume_and_refuse_damage(tmp_path: Path) -> None:
    trainer = Trainer(model(False), seed=18100011)
    trainer.train(200, batch_size=4)
    directory = tmp_path / "trained"
    digest = binding.save_bound(trainer, directory, seed=11, layout_sha256=LAYOUT)
    restored = binding.restore_bound(
        model(False), directory, seed=11, layout_sha256=LAYOUT, binding_sha256=digest
    )
    assert trainer.train(1, batch_size=4) == restored.train(1, batch_size=4)
    assert torch.equal(trainer.generator.get_state(), restored.generator.get_state())
    path = directory / "checkpoint.pt"
    payload = torch.load(path, weights_only=True)
    for state in payload["optimizer"]["state"].values():
        state["exp_avg_sq"].fill_(-1)
    torch.save(payload, path)
    digest = rewrite(directory, "checkpoint_sha256", archive.digest(path))
    with pytest.raises(ValueError, match="second moment"):
        binding.restore_bound(
            model(False),
            directory,
            seed=11,
            layout_sha256=LAYOUT,
            binding_sha256=digest,
        )


@pytest.mark.parametrize("damage", ["seed", "horizon", "nonfinite"])
def test_invalid_trainer_refused_before_creating_directory(
    tmp_path: Path, damage: str
) -> None:
    trainer = Trainer(model(), seed=18100011)
    if damage == "seed":
        trainer.generator.manual_seed(99)
    elif damage == "horizon":
        trainer.step = 1
    else:
        with torch.no_grad():
            next(trainer.model.parameters()).fill_(float("nan"))
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError):
        binding.save_bound(trainer, directory, seed=11, layout_sha256=LAYOUT)
    assert not directory.exists()
