"""Exact synthetic primary-shard fixtures; no physical study or device evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from examples.qec import compare_ising as comparison
from examples.qec import continued_assessment as inputs
from examples.qec import continued_parity as parity
from examples.qec import continued_primary as primary
from examples.qec import continued_selection as selection
from examples.qec import diagnose_predecoder as archive
from flagquantum.qec.decoders._ai.comparison import Correction

pytestmark = pytest.mark.integration
CIRCUIT_SHA = hashlib.sha256(b"exact synthetic primary circuit").hexdigest()


class Circuit:
    num_detectors = 3
    num_observables = 2

    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.calls: list[int] = []

    def __str__(self) -> str:
        return "exact synthetic primary circuit"

    def compile_detector_sampler(self, *, seed: int) -> Circuit:
        self.seeds.append(seed)
        return self

    def sample(
        self, shots: int, *, separate_observables: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert separate_observables
        chunk = len(self.calls)
        self.calls.append(shots)
        index = torch.arange(shots)
        bits = torch.stack(
            (
                index % 2,
                (index % 4 == 0).to(torch.int64),
                torch.full_like(index, chunk),
            ),
            dim=1,
        ).to(torch.int8)
        return bits, torch.zeros(shots, 2, dtype=torch.int8)


def global_zero(bits: torch.Tensor) -> torch.Tensor:
    return torch.zeros(len(bits), 2, dtype=torch.int8)


def corrections() -> dict[str, Correction]:
    def make(name: str) -> Correction:
        def correct(bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            frame = torch.zeros(len(bits), 2, dtype=torch.int8)
            if name in ("global", "fast"):
                frame[:, 0] = bits[:, 0]
            elif name == "mask-only" or "-step-200-" in name:
                frame[:, 1] = bits[:, 1]
            return bits, frame

        return correct

    return {name: make(name) for name in parity.PATHS}


@pytest.fixture
def budget(monkeypatch: pytest.MonkeyPatch) -> None:
    # This exact software fixture does not change the registered production budget.
    monkeypatch.setattr(primary, "SHOTS", 64)
    monkeypatch.setattr(primary, "CHUNK", 32)


def binding() -> primary.Binding:
    return primary.Binding(
        archive.digest(inputs.PROTOCOL),
        CIRCUIT_SHA,
        "a" * 64,
        "b" * 64,
        {"a800-172": "c" * 64, "a800-171": "d" * 64},
        primary._sources(),
        {"purpose": "synthetic software fixture only"},
    )


@pytest.fixture
def captured(
    tmp_path: Path, budget: None
) -> tuple[Path, primary.PrimaryManifest, Circuit]:
    output = tmp_path / "primary"
    circuit = Circuit()
    record = primary._capture(circuit, corrections(), global_zero, output, binding())
    return output, record, circuit


def test_single_sampler_shared_predictions_and_registered_counts(
    captured: tuple[Path, primary.PrimaryManifest, Circuit],
    tmp_path: Path,
) -> None:
    directory, manifest, circuit = captured
    assert circuit.seeds == [16000011] and circuit.calls == [32, 32]
    assert manifest.shots == 64 and len(manifest.shards) == 2
    assert manifest.shards[0].detectors_sha256 != manifest.shards[1].detectors_sha256
    sha = archive.digest(directory / "manifest.json")
    chunks = list(primary.replay_primary(directory, expected_manifest_sha256=sha))
    assert len(chunks) == 2
    for truth, predicted in chunks:
        assert tuple(predicted) == parity.PATHS
        assert truth.shape == (32, 2)
    output = tmp_path / "summary.json"
    primary.summarize_primary(directory, output, manifest_sha256=sha)
    report = json.loads(output.read_text())
    assert len(primary.COMPARISONS) == 26 and len(report["comparisons"]) == 26
    assert report["independent_physical_shots"] == 64
    assert report["comparison_alpha"] == 0.05 / 26
    assert report["native_quality_gate"] == "supported_10_percent_reduction"
    assert report["competitive_with_fast"] is True
    assert report["speed_claim_allowed"] is False
    for row in report["comparisons"]:
        expected = (
            32
            if row["candidate"] == "fast"
            else (
                16
                if (row["candidate"] == "mask-only" or "-step-200-" in row["candidate"])
                else 0
            )
        )
        assert row["counts"]["shots"] == 64
        assert row["counts"]["baseline_failures"] == 32
        assert row["counts"]["hybrid_failures"] == expected
        # Mask/step200 errors are disjoint from odd-index baseline errors.
        assert row["counts"]["wins"] == (0 if row["candidate"] == "fast" else 32)
        assert row["counts"]["losses"] == (
            0 if row["candidate"] == "fast" else expected
        )
        assert row["exact"]["comparison_alpha"] == 0.05 / 26
        assert "legacy_hoeffding" in row
    assert len(report["horizon_counts_descriptive_only"]) == 6
    assert all(
        row["counts"]["baseline_failures"] == 16
        and row["counts"]["hybrid_failures"] == 0
        for row in report["horizon_counts_descriptive_only"]
    )
    assert circuit.calls == [32, 32]
    with pytest.raises(FileExistsError):
        primary.summarize_primary(directory, output, manifest_sha256=sha)
    with pytest.raises(FileExistsError):
        primary._capture(circuit, corrections(), global_zero, directory, binding())
    assert circuit.calls == [32, 32]


@pytest.mark.parametrize(
    "failure", ["missing", "extra", "reorder", "corrupt", "prediction", "manifest"]
)
def test_replay_refuses_incomplete_or_changed_primary(
    captured: tuple[Path, primary.PrimaryManifest, Circuit],
    failure: str,
) -> None:
    directory, _, _ = captured
    path = directory / "manifest.json"
    raw = json.loads(path.read_text())
    if failure == "missing":
        (directory / "chunk-0001.pt").unlink()
    elif failure == "extra":
        (directory / "chunk-0002.pt").write_bytes(b"extra")
    elif failure == "reorder":
        raw["shards"].reverse()
    elif failure == "corrupt":
        (directory / "chunk-0001.pt").write_bytes(b"changed")
    elif failure == "prediction":
        shard = directory / "chunk-0000.pt"
        payload = torch.load(shard, weights_only=True)
        payload["predictions"]["fast"][0, 0] = 2
        torch.save(payload, shard)
        raw["shards"][0]["file_sha256"] = archive.digest(shard)
    else:
        raw["seed"] = 15000011
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        next(
            primary.replay_primary(
                directory, expected_manifest_sha256=archive.digest(path)
            )
        )


def test_partial_prediction_failure_never_completes_or_retries(
    tmp_path: Path,
    budget: None,
) -> None:
    paths = corrections()

    def fails_second(bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if bool(bits[:, 2].any()):
            raise ValueError("synthetic late decoder failure")
        return bits, torch.zeros(len(bits), 2, dtype=torch.int8)

    paths[parity.PATHS[-1]] = fails_second
    circuit = Circuit()
    output = tmp_path / "partial"
    with pytest.raises(ValueError, match="late decoder failure"):
        primary._capture(circuit, paths, global_zero, output, binding())
    assert (output / "plan.json").exists() and (output / "chunk-0000.pt").exists()
    assert not (output / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        primary._capture(circuit, paths, global_zero, output, binding())
    assert circuit.calls == [32, 32]


@pytest.mark.parametrize("guard", ["choices", "receipts"])
def test_frozen_guards_precede_cuda_reference_and_sampler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    guard: str,
) -> None:
    def reject(*args: object, **kwargs: object) -> None:
        raise ValueError("synthetic frozen guard rejection")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid inputs reached primary execution")

    monkeypatch.setattr(
        selection,
        "verify_choices",
        reject if guard == "choices" else lambda *a, **k: None,
    )
    monkeypatch.setattr(parity, "verify_two_hosts", reject)
    monkeypatch.setattr(comparison, "cuda_metadata", forbidden)
    monkeypatch.setattr(comparison, "reference", forbidden)
    monkeypatch.setattr(primary, "_capture", forbidden)
    with pytest.raises(ValueError, match="frozen guard rejection"):
        primary.run_primary(
            [],
            tmp_path,
            tmp_path,
            {},
            tmp_path / "output",
            choices_sha256="a" * 64,
            probes_sha256="b" * 64,
            ising_source=tmp_path,
            ising_weights=tmp_path,
        )
    assert not (tmp_path / "output").exists()


def test_wrong_official_weights_refuse_before_primary_sampler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(selection, "verify_choices", lambda *a, **k: None)
    monkeypatch.setattr(parity, "verify_two_hosts", lambda *a, **k: None)
    monkeypatch.setattr(
        comparison,
        "cuda_metadata",
        lambda *a: {
            "nvidia_smi_uuid_name_driver": "synthetic-test-device,A800,fixture"
        },
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "hardware": {
                    "nvidia_smi_uuid_name_driver": "synthetic-test-device,A800,fixture"
                },
                "torch_version": str(torch.__version__),
            }
        )
    )
    weights = tmp_path / "wrong.safetensors"
    weights.write_bytes(b"explicitly invalid unit fixture")
    monkeypatch.setattr(
        comparison, "source_digest", lambda root: comparison.ISING_SOURCE_SHA256
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("bad weights reached sampling")

    monkeypatch.setattr(primary, "_capture", forbidden)
    with pytest.raises(ValueError, match="official Fast checkpoint"):
        primary.run_primary(
            [],
            tmp_path,
            tmp_path,
            {"a800-172": (receipt, "c" * 64)},
            tmp_path / "unused",
            choices_sha256="a" * 64,
            probes_sha256="b" * 64,
            ising_source=tmp_path,
            ising_weights=weights,
        )
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("case", ["one_bad_seed", "zero_reference"])
def test_quality_needs_all_fixed_seeds_and_nonzero_reference_bound(
    captured: tuple[Path, primary.PrimaryManifest, Circuit],
    tmp_path: Path,
    case: str,
) -> None:
    directory, _, _ = captured
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    for row in manifest["shards"]:
        shard = directory / row["filename"]
        payload = torch.load(shard, weights_only=True)
        predictions = payload["predictions"]
        key = "native-22-step-2000-calibrated" if case == "one_bad_seed" else "global"
        predictions[key] = (
            predictions["global"].clone()
            if case == "one_bad_seed"
            else torch.zeros_like(predictions["global"])
        )
        torch.save(payload, shard)
        row["file_sha256"] = archive.digest(shard)
        row["prediction_sha256"][key] = inputs._bit_hash(predictions[key])
    path.write_text(json.dumps(manifest))
    output = tmp_path / "conjunction.json"
    primary.summarize_primary(directory, output, manifest_sha256=archive.digest(path))
    report = json.loads(output.read_text())
    assert report["native_quality_gate"] == "inconclusive"
    assert report["independent_physical_shots"] == 64
    if case == "zero_reference":
        assert all(
            r["exact"]["ratio_upper"] is None
            for r in report["comparisons"]
            if r["reference"] == "global"
        )


def test_exact_statistical_failure_cannot_write_success_or_fallback(
    captured: tuple[Path, primary.PrimaryManifest, Circuit],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.qec.decoders._ai import exact_statistics

    directory, _, _ = captured

    def refused(*args: object, **kwargs: object) -> None:
        raise ArithmeticError("synthetic uncertified exact interval")

    monkeypatch.setattr(exact_statistics, "summarize_exact", refused)
    with pytest.raises(ArithmeticError, match="uncertified"):
        primary.summarize_primary(
            directory,
            tmp_path / "not-written.json",
            manifest_sha256=archive.digest(directory / "manifest.json"),
        )
    assert not (tmp_path / "not-written.json").exists()


def test_missing_prediction_path_refuses_before_sampler(
    tmp_path: Path, budget: None
) -> None:
    circuit = Circuit()
    paths = corrections()
    paths.pop(parity.PATHS[-1])
    with pytest.raises(ValueError, match="fifteen"):
        primary._capture(circuit, paths, global_zero, tmp_path / "unused", binding())
    assert not circuit.seeds and not (tmp_path / "unused").exists()


def test_runtime_change_since_receipt_refuses_before_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(selection, "verify_choices", lambda *a, **k: None)
    monkeypatch.setattr(parity, "verify_two_hosts", lambda *a, **k: None)
    receipt = tmp_path / "runtime.json"
    receipt.write_text(json.dumps({"torch_version": "explicitly different fixture"}))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("changed runtime reached CUDA")

    monkeypatch.setattr(comparison, "cuda_metadata", forbidden)
    with pytest.raises(ValueError, match="Torch version"):
        primary.run_primary(
            [],
            tmp_path,
            tmp_path,
            {"a800-172": (receipt, "c" * 64)},
            tmp_path / "unused",
            choices_sha256="a" * 64,
            probes_sha256="b" * 64,
            ising_source=tmp_path,
            ising_weights=tmp_path,
        )
    assert not (tmp_path / "unused").exists()
