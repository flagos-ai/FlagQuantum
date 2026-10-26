"""Synthetic initialization oracles; not pinned-source or physical receipts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from examples.qec import diagnose_predecoder as archive
from examples.qec import ising_layout
from examples.qec import matched_checkpoints as checkpoints
from examples.qec import matched_initialization as initialization
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.spacetime import SpacetimePredecoder
from flagquantum.qec.decoders._ai.training import Trainer
from flagquantum.qec.dem.model import DemError, DetectorErrorModel

pytestmark = pytest.mark.integration


def fixture(monkeypatch: pytest.MonkeyPatch) -> SpacetimePredecoder:
    dem = DetectorErrorModel(4, 1, (DemError(0.01, (0, 2, 3), (0,)),))
    template = SpacetimePredecoder.from_dem(
        dem, grid_shape=(2, 2, 2), sites=(0, 2, 5, 7), kinds=(1, 2, 1, 2)
    )

    def verified(source: Path) -> tuple[SpacetimePredecoder, dict[str, object]]:
        torch.rand(1)  # Simulate the real template constructor consuming Torch RNG.
        return template, {
            "layout_sha256": "a" * 64,
            "purpose": "synthetic_fixture_only",
        }

    monkeypatch.setattr(ising_layout, "build_model", verified)
    return template


def test_all_six_seeded_states_and_rng_match_independent_constructors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = fixture(monkeypatch)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(
            "no forward, sampler or optimizer update in initialization"
        )

    monkeypatch.setattr(Predecoder, "sample", forbidden)
    monkeypatch.setattr(Predecoder, "forward", forbidden)
    monkeypatch.setattr(SpacetimePredecoder, "forward", forbidden)
    monkeypatch.setattr(Trainer, "train", forbidden)
    original = torch.get_rng_state().clone()
    output = tmp_path / "initial"
    report = initialization.initialize(tmp_path, output)
    assert torch.equal(torch.get_rng_state(), original)
    assert report["new_samples"] == report["new_optimizer_steps"] == 0
    assert report["formal_launch_authorized"] is False
    assert [
        (row["seed"], row["architecture"]) for row in report["initial_checkpoints"]
    ] == [(s, a) for s in (11, 22, 33) for a in ("cnn", "mlp")]
    rng = torch.load(output / "initialization-rng.pt", weights_only=True)
    dem = DetectorErrorModel.from_stim_text(template.dem_text)
    with torch.random.fork_rng(devices=[]):
        for row in report["initial_checkpoints"]:
            seed, architecture = row["seed"], row["architecture"]
            torch.manual_seed(seed)
            assert torch.equal(
                torch.get_rng_state(), rng[f"{architecture}-{seed}-before"]
            )
            if architecture == "cnn":
                model: Predecoder = SpacetimePredecoder.from_dem(
                    dem, grid_shape=(2, 2, 2), sites=(0, 2, 5, 7), kinds=(1, 2, 1, 2)
                )
            else:
                model = Predecoder(dem, allow_hyperedges=True)
            assert torch.equal(
                torch.get_rng_state(), rng[f"{architecture}-{seed}-after"]
            )
            assert row["model_state_sha256"] == archive.state_digest(model.state_dict())
            restored = checkpoints.restore_bound(
                model,
                output / f"{architecture}-{seed}",
                seed=seed,
                layout_sha256="a" * 64,
                binding_sha256=row["binding_sha256"],
            )
            assert restored.step == 0 and not restored.optimizer.state
            assert torch.equal(
                restored.generator.get_state(),
                torch.Generator().manual_seed(seed + 18100000).get_state(),
            )
    assert (output / "initialization.json").is_file()
    assert report["initialization_rng_file_sha256"] == archive.digest(
        output / "initialization-rng.pt"
    )


@pytest.mark.parametrize(
    "seeds", [(11, 22), (11, 22, 22), (22, 11, 33), (11, 22, 33, 44)]
)
def test_missing_duplicate_or_reordered_seed_inventory_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seeds: tuple[int, ...]
) -> None:
    fixture(monkeypatch)
    monkeypatch.setattr(archive, "SEEDS", seeds)
    output = tmp_path / "bad"
    with pytest.raises(ValueError, match="all six"):
        initialization.initialize(tmp_path, output)
    assert not output.exists()


def test_duplicate_and_failed_partial_freeze_cannot_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture(monkeypatch)
    output = tmp_path / "initial"
    original = checkpoints.save_bound
    calls: list[int] = []

    def failing(
        trainer: Trainer, directory: Path, *, seed: int, layout_sha256: str
    ) -> str:
        calls.append(seed)
        if len(calls) == 6:
            raise RuntimeError("injected sixth checkpoint write failure")
        return original(trainer, directory, seed=seed, layout_sha256=layout_sha256)

    monkeypatch.setattr(checkpoints, "save_bound", failing)
    with pytest.raises(RuntimeError):
        initialization.initialize(tmp_path, output)
    assert not (output / "initialization.json").exists()
    assert len(list(output.glob("*/binding.json"))) == 5
    with pytest.raises(FileExistsError):
        initialization.initialize(tmp_path, output)
    assert len(calls) == 6


def test_refused_restore_never_writes_complete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture(monkeypatch)

    def refused(*args: Any, **kwargs: Any) -> Trainer:
        raise ValueError("injected identity refusal")

    monkeypatch.setattr(checkpoints, "restore_bound", refused)
    output = tmp_path / "initial"
    with pytest.raises(ValueError, match="identity"):
        initialization.initialize(tmp_path, output)
    assert not (output / "initialization.json").exists()


def test_source_refusal_creates_no_output_and_preserves_caller_rng(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed(source: Path) -> tuple[SpacetimePredecoder, dict[str, object]]:
        torch.rand(1)
        raise ValueError("injected pinned-source mismatch")

    monkeypatch.setattr(ising_layout, "build_model", failed)
    before = torch.get_rng_state().clone()
    output = tmp_path / "initial"
    with pytest.raises(ValueError, match="pinned-source"):
        initialization.initialize(tmp_path, output)
    assert torch.equal(torch.get_rng_state(), before)
    assert not output.exists()


def test_changed_protocol_during_freeze_does_not_publish_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture(monkeypatch)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen")
    monkeypatch.setattr(checkpoints, "PROTOCOLS", (protocol,))
    original = checkpoints.restore_bound

    def changed(*args: Any, **kwargs: Any) -> Trainer:
        result = original(*args, **kwargs)
        protocol.write_text("changed")
        return result

    monkeypatch.setattr(checkpoints, "restore_bound", changed)
    output = tmp_path / "initial"
    with pytest.raises(ValueError, match="documents changed"):
        initialization.initialize(tmp_path, output)
    assert not (output / "initialization.json").exists()
