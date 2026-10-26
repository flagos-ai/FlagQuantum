"""Archived-input preflight and bounded CPU label-study workflow."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import struct
from pathlib import Path

import pytest
import torch

from examples.qec import diagnose_predecoder as study
from flagquantum.qec.decoders._ai.diagnostics import diagnose
from flagquantum.qec.decoders._ai.predecoder import Predecoder, TrainingBatch
from flagquantum.qec.decoders._ai.training import Trainer
from flagquantum.qec.dem.model import DetectorErrorModel

pytestmark = pytest.mark.integration


@pytest.fixture
def archived(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Synthetic tiny checkpoints, not scientific 200-step training evidence."""
    directory = tmp_path / "inputs"
    directory.mkdir()
    dem = DetectorErrorModel.from_stim_text(
        "error(0.1) D0 D1\nerror(0.1) D1 D2\nerror(0.1) D0 D2\n"
        "detector D0\ndetector D1\ndetector D2\n"
    )
    (directory / "training.dem").write_text(dem.to_stim_text())
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "python": platform.python_version(),
                "dependencies": {},
            }
        )
    )
    (directory / "choices.json").write_text(
        json.dumps(
            {
                str(s): {
                    "flagquantum_calibrated": {"temperature": 1.0, "threshold": 1.0}
                }
                for s in study.SEEDS
            }
        )
    )
    fingerprints: dict[str, str] = {}
    for seed in study.SEEDS:
        torch.manual_seed(seed)
        model = Predecoder(dem)
        trainer = Trainer(model)
        trainer.step = 200  # Deliberate fixture metadata; no optimizer steps.
        trainer.save(directory / f"native-{seed}.pt")
        fingerprints[str(seed)] = study.state_digest(model.state_dict())
    (directory / "audit.json").write_text(
        json.dumps({"native_model_state_sha256": fingerprints})
    )
    (directory / "results.json").write_text(
        json.dumps(
            {
                "manifest_sha256": study.digest(directory / "manifest.json"),
                "choices_sha256": study.digest(directory / "choices.json"),
                "training_dem_sha256": study.digest(directory / "training.dem"),
                "checkpoint_sha256": {
                    str(s): study.digest(directory / f"native-{s}.pt")
                    for s in study.SEEDS
                },
            }
        )
    )
    anchor = tmp_path / "anchor"
    shutil.copytree(directory, anchor)
    monkeypatch.setattr(study, "ARCHIVE", anchor)
    return directory, anchor


def test_native_bytes_match_the_prior_audit_without_numpy() -> None:
    value = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float32).T
    expected = hashlib.sha256()
    expected.update(b"weight\0torch.float32\0(2, 2)\0")
    expected.update(struct.pack("=4f", 1.0, 3.0, -2.0, 4.0))
    assert study.state_digest({"weight": value}) == expected.hexdigest()


@pytest.mark.parametrize("failure", ["missing", "metadata", "checkpoint", "dem"])
def test_corrupt_archived_inputs_abort_before_sampling(
    archived: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    directory, _ = archived
    if failure == "missing":
        (directory / "native-33.pt").unlink()
    elif failure == "metadata":
        (directory / "choices.json").write_text("{}")
    elif failure == "checkpoint":
        (directory / "native-33.pt").write_bytes(b"corrupted")
    else:
        (directory / "training.dem").write_text("changed")

    def no_sampling(*args: object) -> None:
        raise AssertionError("unverified archive reached new samples")

    monkeypatch.setattr(Predecoder, "sample", no_sampling)
    output = tmp_path / "must-not-exist"
    with pytest.raises((ValueError, FileNotFoundError)):
        study.run(directory, output, smoke=True)
    assert not output.exists()


@pytest.mark.parametrize("failure", ["step", "checkpoint_dem", "tensor_state"])
def test_matching_file_hash_does_not_replace_semantic_checkpoint_checks(
    archived: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    directory, anchor = archived
    path = directory / "native-33.pt"
    raw = torch.load(path, weights_only=True)
    if failure == "step":
        raw["step"] = 199
    elif failure == "checkpoint_dem":
        raw["dem"] = "different DEM"
    else:
        raw["model"]["prior"][0] = 0.2
    torch.save(raw, path)
    results = json.loads((directory / "results.json").read_text())
    results["checkpoint_sha256"]["33"] = study.digest(path)
    for target in (directory, anchor):
        (target / "results.json").write_text(json.dumps(results))

    def no_sampling(*args: object) -> None:
        raise AssertionError("invalid late seed reached primary diagnostic sampling")

    monkeypatch.setattr(Predecoder, "sample", no_sampling)
    with pytest.raises(
        ValueError,
        match={
            "step": "step-200",
            "checkpoint_dem": "DEM mismatch",
            "tensor_state": "tensor-state",
        }[failure],
    ):
        study.run(directory, tmp_path / "no-output", smoke=True)
    assert not (tmp_path / "no-output").exists()


def test_chunked_report_matches_the_same_complete_latent_stream(
    archived: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, _ = archived
    models, _ = study.load_inputs(directory)
    model = models[0].model
    generator = torch.Generator().manual_seed(11 + 9100000)
    batch = model.sample(4096, generator)
    logits = model(batch.sample.detectors)
    expected_loss = diagnose(logits, batch.mechanisms)
    expected_hash = hashlib.sha256(
        study.tensor_bytes(batch.mechanisms.to(torch.int8))
    ).hexdigest()
    original = Predecoder.sample
    chunks: list[int] = []

    def bounded(
        self: Predecoder, shots: int, generator: torch.Generator
    ) -> TrainingBatch:
        chunks.append(shots)
        assert shots <= 32
        return original(self, shots, generator)

    monkeypatch.setattr(Predecoder, "sample", bounded)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    report = study.diagnose_model(models[0], 4096)
    assert chunks == [32] * 128
    assert report.latent_bytes_sha256 == expected_hash
    assert report.raw_loss.total_labels == expected_loss.total_labels
    assert report.raw_loss.positive_labels == expected_loss.positive_labels
    assert report.raw_loss.positive_bce_sum == pytest.approx(
        expected_loss.positive_bce_sum
    )
    assert report.raw_loss.negative_bce_sum == pytest.approx(
        expected_loss.negative_bce_sum
    )
    assert report.raw_loss.brier_score == pytest.approx(expected_loss.brier_score)
    assert report.raw_loss.mean_probability == pytest.approx(
        expected_loss.mean_probability
    )
    assert report.feature_summary.observations == expected_loss.total_labels
    assert all(v.shots == 4096 for v in report.controls.values())
    assert report.controls["zero"] == report.controls["frozen_calibrated"]
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_verified_smoke_report_is_fresh_and_has_derived_group_data(
    archived: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    directory, _ = archived
    output = tmp_path / "report"
    study.run(directory, output, smoke=True)
    manifest = json.loads((output / "manifest.json").read_text())
    report = json.loads((output / "seed-11.json").read_text())
    assert manifest["purpose"] == "smoke"
    assert manifest["new_training_steps"] == manifest["new_primary_test_shots"] == 0
    assert report["sampler_seed"] == 9100011
    assert report["diagnostic_shots"] == 32
    assert set(report["controls"]) == {
        "zero",
        "raw_0.5",
        "raw_0.9",
        "frozen_calibrated",
    }
    for group in report["feature_summary"]["groups"]:
        assert (
            group["positive_labels"] + group["negative_labels"] == group["observations"]
        )
        assert (
            group["positive_fraction"]
            == group["positive_labels"] / group["observations"]
        )
    assert not (output / "seed-22.json").exists()
    with pytest.raises(FileExistsError):
        study.run(directory, output, smoke=True)
