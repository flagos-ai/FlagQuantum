"""Synthetic archive and circuit fixtures; no physical research collection."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

from examples.qec import continued_assessment as study
from examples.qec import diagnose_predecoder as archive
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.training import Trainer
from flagquantum.qec.dem.model import DetectorErrorModel

pytestmark = pytest.mark.integration
CIRCUIT_SHA = hashlib.sha256(b"synthetic circuit fixture only").hexdigest()


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """Synthetic horizon metadata, never evidence of 200/2000 training steps."""
    original = tmp_path / "original"
    original.mkdir()
    continued = tmp_path / "continued"
    continued.mkdir()
    (continued / "manifest.json").write_text('{"purpose":"synthetic_fixture"}')
    dem = DetectorErrorModel.from_stim_text("error(0.1) D0 L0\ndetector D0\n")
    models = []
    rows = []
    for seed in archive.SEEDS:
        torch.manual_seed(seed)
        model = Predecoder(dem)
        fingerprint = archive.state_digest(model.state_dict())
        models.append(
            archive.ArchivedModel(
                seed,
                model,
                "fixture",
                fingerprint,
                {"temperature": 1.0, "threshold": 1.0},
            )
        )
        for step in (200, 2000):
            trainer = Trainer(model)
            trainer.step = step
            path = continued / f"native-{seed}-step-{step}.pt"
            trainer.save(path)
            rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "checkpoint_sha256": archive.digest(path),
                    "model_state_sha256": fingerprint,
                }
            )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "raw_file_sha256": {
                    "full/manifest.json": archive.digest(continued / "manifest.json")
                },
                "checkpoint_audit": {"rows": rows},
            }
        )
    )
    monkeypatch.setattr(study, "AUDIT", audit)

    def verified(directory: Path) -> tuple[list[archive.ArchivedModel], dict[str, str]]:
        assert directory == original
        return models, {"manifest.json": "original-fixture"}

    monkeypatch.setattr(archive, "load_inputs", verified)
    return original, continued, audit


def test_all_six_checkpoints_verify_without_sampling_or_updates(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("input verification reached new data or training")

    monkeypatch.setattr(Predecoder, "sample", forbidden)
    monkeypatch.setattr(Trainer, "train", forbidden)
    original, continued, _ = inputs
    models, hashes = study.load_six(original, continued)
    assert [(m.seed, m.step) for m in models] == [
        (s, h) for s in archive.SEEDS for h in (200, 2000)
    ]
    assert all(not m.model.training for m in models)
    assert len(hashes) == 8


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "file_hash",
        "step",
        "version",
        "dem",
        "fingerprint",
        "nonfinite",
        "manifest",
    ],
)
def test_late_checkpoint_and_metadata_refuse_before_sampling(
    inputs: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    original, continued, audit = inputs
    path = continued / "native-33-step-2000.pt"
    data = json.loads(audit.read_text())
    row = data["checkpoint_audit"]["rows"][-1]
    if failure == "missing":
        path.unlink()
    elif failure == "file_hash":
        path.write_bytes(b"broken")
    elif failure == "manifest":
        (continued / "manifest.json").write_text("{}")
    elif failure == "fingerprint":
        row["model_state_sha256"] = "0" * 64
    else:
        raw = torch.load(path, weights_only=True)
        if failure == "step":
            raw["step"] = 600
        elif failure == "version":
            raw["version"] = 2
        elif failure == "dem":
            raw["dem"] = "wrong DEM"
        else:
            raw["model"]["prior"].fill_(float("nan"))
            row["model_state_sha256"] = archive.state_digest(raw["model"])
        torch.save(raw, path)
        row["checkpoint_sha256"] = archive.digest(path)
    audit.write_text(json.dumps(data))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("corrupt input reached sampling or training")

    monkeypatch.setattr(Predecoder, "sample", forbidden)
    monkeypatch.setattr(Trainer, "train", forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        study.load_six(original, continued)


class SyntheticCircuit:
    num_detectors = 6
    num_observables = 1

    def __init__(self, *, bad_second: bool = False) -> None:
        self.compiled: list[int] = []
        self.calls: list[int] = []
        self.bad_second = bad_second

    def __str__(self) -> str:
        return "synthetic circuit fixture only"

    def compile_detector_sampler(self, *, seed: int) -> SyntheticCircuit:
        self.compiled.append(seed)
        return self

    def sample(
        self, shots: int, *, separate_observables: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert separate_observables
        index = len(self.calls)
        self.calls.append(shots)
        bits = torch.zeros(shots, 6)
        bits[0] = torch.tensor([(index >> i) & 1 for i in range(6)])
        if self.bad_second and index == 1:
            bits[0, 0] = float("nan")
        return bits, torch.zeros(shots, 1)


def test_one_sampler_fixed_order_shared_immutable_shards(tmp_path: Path) -> None:
    circuit = SyntheticCircuit()
    output = tmp_path / "validation"
    record = study.capture_validation(circuit, output)
    assert circuit.compiled == [15000011]
    assert circuit.calls == [4096] * 32
    assert record.shots == 131072 and len(record.shards) == 32
    first = [
        bits[0].clone()
        for bits, _ in study.replay_validation(output, circuit_sha256=CIRCUIT_SHA)
    ]
    second = [
        bits[0].clone()
        for bits, _ in study.replay_validation(output, circuit_sha256=CIRCUIT_SHA)
    ]
    assert len({tuple(x.tolist()) for x in first}) == 32
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert circuit.calls == [4096] * 32
    with pytest.raises(FileExistsError):
        study.capture_validation(circuit, output)
    assert circuit.compiled == [15000011]


def test_late_corrupt_shard_is_detected_before_first_replay(tmp_path: Path) -> None:
    output = tmp_path / "validation"
    study.capture_validation(SyntheticCircuit(), output)
    (output / "chunk-0031.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        next(study.replay_validation(output, circuit_sha256=CIRCUIT_SHA))


def test_partial_capture_is_retained_without_completion_or_retry(
    tmp_path: Path,
) -> None:
    output = tmp_path / "validation"
    circuit = SyntheticCircuit(bad_second=True)
    with pytest.raises(ValueError, match="binary arrays"):
        study.capture_validation(circuit, output)
    assert (output / "plan.json").exists() and (output / "chunk-0000.pt").exists()
    assert not (output / "manifest.json").exists()
    assert not (output / "chunk-0001.pt").exists()
    with pytest.raises(FileNotFoundError):
        next(study.replay_validation(output, circuit_sha256=CIRCUIT_SHA))
    with pytest.raises(FileExistsError):
        study.capture_validation(circuit, output)
    assert circuit.compiled == [15000011]


def test_cli_is_input_verification_only(
    inputs: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original, continued, _ = inputs
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "continued_assessment.py",
            "--original-directory",
            str(original),
            "--continued-directory",
            str(continued),
        ],
    )
    study.main()
    assert json.loads(capsys.readouterr().out) == {
        "purpose": "verify_inputs_only",
        "verified_checkpoints": 6,
        "new_samples": 0,
        "new_optimizer_steps": 0,
    }


@pytest.mark.parametrize("failure", ["missing", "extra", "reordered", "nonbinary"])
def test_shard_replay_refuses_incomplete_or_tampered_capture(
    tmp_path: Path, failure: str
) -> None:
    output = tmp_path / "validation"
    study.capture_validation(SyntheticCircuit(), output)
    manifest = json.loads((output / "manifest.json").read_text())
    if failure == "missing":
        (output / "chunk-0031.pt").unlink()
    elif failure == "extra":
        (output / "chunk-0032.pt").write_bytes(b"unexpected")
    elif failure == "reordered":
        manifest["shards"][0], manifest["shards"][1] = (
            manifest["shards"][1],
            manifest["shards"][0],
        )
    else:
        path = output / "chunk-0000.pt"
        raw = torch.load(path, weights_only=True)
        raw["detectors"][0, 0] = 2
        torch.save(raw, path)
        manifest["shards"][0]["file_sha256"] = archive.digest(path)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        next(study.replay_validation(output, circuit_sha256=CIRCUIT_SHA))


def test_other_circuit_capture_cannot_be_silently_reused(tmp_path: Path) -> None:
    output = tmp_path / "validation"
    study.capture_validation(SyntheticCircuit(), output)
    with pytest.raises(ValueError, match="frozen stage"):
        next(study.replay_validation(output, circuit_sha256="0" * 64))
