"""Synthetic parity/receipt fixtures, never evidence of A800 execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

from examples.qec import compare_ising as comparison
from examples.qec import continued_assessment as inputs
from examples.qec import continued_parity as parity
from examples.qec import continued_selection as selection
from examples.qec import diagnose_predecoder as archive
from flagquantum.qec.decoders._ai.comparison import Correction

pytestmark = pytest.mark.integration
CHOICES_SHA = "a" * 64
CIRCUIT_SHA = hashlib.sha256(b"synthetic probe circuit").hexdigest()


class Circuit:
    num_detectors = 3
    num_observables = 1

    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.calls: list[int] = []

    def __str__(self) -> str:
        return "synthetic probe circuit"

    def compile_detector_sampler(self, *, seed: int) -> Circuit:
        self.seeds.append(seed)
        return self

    def sample(
        self, shots: int, *, separate_observables: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert separate_observables
        self.calls.append(shots)
        return torch.tensor([[1, 1, 0]], dtype=torch.int8).repeat(
            shots, 1
        ), torch.zeros(shots, 1)


@pytest.fixture
def probes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Only exercise the probe stage here; real choices integration is tested separately.
    monkeypatch.setattr(selection, "verify_choices", lambda *a, **k: None)
    circuit = Circuit()
    output = tmp_path / "probes"
    record = parity.capture_probes(
        circuit,
        tmp_path / "choices",
        [],
        output,
        expected_choices_sha256=CHOICES_SHA,
        circuit_sha256=CIRCUIT_SHA,
    )
    assert record.seed == 17000011 and record.physical_shots == 64
    assert circuit.seeds == [17000011] and circuit.calls == [64]
    return output


def identity(bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return bits.clone(), torch.zeros(len(bits), 1, dtype=torch.int8)


def matching_physical_only(bits: torch.Tensor) -> torch.Tensor:
    # One-hot/zero/one synthetic syndromes must never reach the matcher.
    assert len(bits) <= 32
    assert bool((bits == torch.tensor([1, 1, 0])).all())
    return torch.zeros(len(bits), 1, dtype=torch.int8)


def test_one_frozen_probe_replays_without_regeneration(probes: Path) -> None:
    sha = archive.digest(probes / "manifest.json")
    first = parity.load_probes(
        probes,
        expected_manifest_sha256=sha,
        choices_sha256=CHOICES_SHA,
        circuit_sha256=CIRCUIT_SHA,
    )
    second = parity.load_probes(
        probes,
        expected_manifest_sha256=sha,
        choices_sha256=CHOICES_SHA,
        circuit_sha256=CIRCUIT_SHA,
    )
    assert torch.equal(first, second)
    pairs = dict.fromkeys(parity.PATHS, (identity, identity))
    evidence = parity.check_paths(pairs, first, matching_physical_only)
    assert len(evidence) == 15
    assert all(r.synthetic_rows == 5 and r.physical_rows == 64 for r in evidence)
    assert len({r.residual_frame_sha256 for r in evidence}) == 1
    assert len({r.physical_predictions_sha256 for r in evidence}) == 1


@pytest.mark.parametrize("failure", ["choices", "circuit", "file", "manifest"])
def test_probe_wrong_binding_or_corruption_refuses(probes: Path, failure: str) -> None:
    sha = archive.digest(probes / "manifest.json")
    choices, circuit = CHOICES_SHA, CIRCUIT_SHA
    if failure == "choices":
        choices = "b" * 64
    elif failure == "circuit":
        circuit = "b" * 64
    elif failure == "file":
        (probes / "physical.pt").write_bytes(b"changed")
    else:
        (probes / "manifest.json").write_text("{}")
    with pytest.raises(ValueError):
        parity.load_probes(
            probes,
            expected_manifest_sha256=sha,
            choices_sha256=choices,
            circuit_sha256=circuit,
        )


def test_capture_rejects_bad_choices_before_sampler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(*args: object, **kwargs: object) -> None:
        raise ValueError("synthetic frozen-choice rejection")

    monkeypatch.setattr(selection, "verify_choices", reject)
    circuit = Circuit()
    with pytest.raises(ValueError, match="frozen-choice rejection"):
        parity.capture_probes(
            circuit,
            tmp_path / "choices",
            [],
            tmp_path / "unused",
            expected_choices_sha256=CHOICES_SHA,
            circuit_sha256=CIRCUIT_SHA,
        )
    assert not circuit.seeds and not (tmp_path / "unused").exists()


@pytest.mark.parametrize(
    "failure", ["residual", "frame", "final_prediction", "missing_path"]
)
def test_bit_or_final_prediction_mismatch_refuses(
    probes: Path,
    failure: str,
) -> None:
    bits = parity.load_probes(
        probes,
        expected_manifest_sha256=archive.digest(probes / "manifest.json"),
        choices_sha256=CHOICES_SHA,
        circuit_sha256=CIRCUIT_SHA,
    )

    def bad(bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual, frame = identity(bits)
        if failure == "residual":
            residual[:, 0] ^= 1
        else:
            frame ^= 1
        return residual, frame

    pairs: dict[str, tuple[Correction, Correction]] = dict.fromkeys(
        parity.PATHS, (identity, identity)
    )
    if failure in ("residual", "frame"):
        pairs[parity.PATHS[-1]] = (identity, bad)
    elif failure == "missing_path":
        pairs.pop(parity.PATHS[-1])
    calls = 0

    def inconsistent_global(bits: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.full((len(bits), 1), calls % 2, dtype=torch.int8)

    with pytest.raises(ValueError):
        parity.check_paths(
            pairs,
            bits,
            (
                inconsistent_global
                if failure == "final_prediction"
                else matching_physical_only
            ),
        )


def test_one_hot_batches_are_complete_and_bounded() -> None:
    chunks = list(parity.synthetic_probes(75))
    assert max(len(x) for x in chunks) <= 32
    assert torch.equal(torch.cat(chunks[2:]), torch.eye(75, dtype=torch.int8))
    assert not bool(chunks[0].any()) and bool(chunks[1].all())


@pytest.fixture
def receipts(probes: Path, tmp_path: Path) -> dict[str, tuple[Path, str]]:
    # Fabricated unit-test documents only. No CUDA writer or physical device ran.
    bits = parity.load_probes(
        probes,
        expected_manifest_sha256=archive.digest(probes / "manifest.json"),
        choices_sha256=CHOICES_SHA,
        circuit_sha256=CIRCUIT_SHA,
    )
    evidence = parity.check_paths(
        dict.fromkeys(parity.PATHS, (identity, identity)), bits, matching_physical_only
    )
    result = {}
    for index, host in enumerate(("a800-172", "a800-171")):
        raw = parity.ParityReceipt(
            "continued_cuda_bit_parity",
            host,
            archive.digest(inputs.PROTOCOL),
            CHOICES_SHA,
            CIRCUIT_SHA,
            archive.digest(probes / "manifest.json"),
            archive.digest(Path(parity.__file__)),
            parity.IMAGE,
            json.loads((archive.ARCHIVE / "manifest.json").read_text())["torch"],
            "b" * 64,
            {
                "device": "cuda:0",
                "name": "A800 synthetic fixture",
                "image_id": parity.IMAGE,
                "tf32": False,
                "autocast": False,
                "deterministic_algorithms": True,
                "nvidia_smi_uuid_name_driver": f"GPU-00000000-0000-0000-0000-{index:012d},A800,fixture",
            },
            evidence,
        )
        path = tmp_path / f"{host}.json"
        path.write_text(json.dumps(asdict(raw)))
        result[host] = (path, archive.digest(path))
    return result


def test_two_host_parser_accepts_identical_bound_software_fixtures(
    receipts: dict[str, tuple[Path, str]],
    probes: Path,
) -> None:
    parity.verify_two_hosts(
        receipts,
        choices_sha256=CHOICES_SHA,
        probes_sha256=archive.digest(probes / "manifest.json"),
        circuit_sha256=CIRCUIT_SHA,
        probe_directory=probes,
    )


@pytest.mark.parametrize(
    "failure",
    [
        "missing_host",
        "hash",
        "choices",
        "probes",
        "source",
        "repeated_gpu",
        "path_bits",
        "incomplete_probes",
        "cpu",
        "tf32",
    ],
)
def test_two_host_parser_refuses_incomplete_changed_or_mismatched_evidence(
    receipts: dict[str, tuple[Path, str]],
    probes: Path,
    failure: str,
) -> None:
    path, _ = receipts["a800-171"]
    raw: dict[str, Any] = json.loads(path.read_text())
    if failure == "missing_host":
        receipts.pop("a800-171")
    elif failure in ("choices", "probes", "source"):
        raw[
            {
                "choices": "choices_sha256",
                "probes": "probes_sha256",
                "source": "source_sha256",
            }[failure]
        ] = (
            "0" * 64
        )
    elif failure == "repeated_gpu":
        first = json.loads(receipts["a800-172"][0].read_text())
        raw["hardware"]["nvidia_smi_uuid_name_driver"] = first["hardware"][
            "nvidia_smi_uuid_name_driver"
        ]
    elif failure == "hash":
        raw["host"] = "tampered"
    elif failure == "path_bits":
        raw["paths"][-1]["physical_predictions_sha256"] = "0" * 64
    elif failure == "incomplete_probes":
        raw["paths"][-1]["synthetic_rows"] = 3
    elif failure == "cpu":
        raw["hardware"]["device"] = "cpu"
    elif failure == "tf32":
        raw["hardware"]["tf32"] = True
    path.write_text(json.dumps(raw))
    if failure not in ("missing_host", "hash"):
        receipts["a800-171"] = path, archive.digest(path)
    with pytest.raises(ValueError):
        parity.verify_two_hosts(
            receipts,
            choices_sha256=CHOICES_SHA,
            probes_sha256=archive.digest(probes / "manifest.json"),
            circuit_sha256=CIRCUIT_SHA,
            probe_directory=probes,
        )


def test_canonical_path_composition_uses_each_frozen_native_setting(
    tmp_path: Path,
) -> None:
    from examples.qec.continued_assessment import VerifiedCheckpoint
    from flagquantum.qec.decoders._ai.predecoder import Predecoder
    from flagquantum.qec.dem.model import DetectorErrorModel

    dem = DetectorErrorModel.from_stim_text(
        "error(0.1) D0 D1 D719 L0\n" + "".join(f"detector D{i}\n" for i in range(720))
    )
    models = []
    paths = []
    for seed in (11, 22, 33):
        for step in (200, 2000):
            model = Predecoder(dem, allow_hyperedges=True).eval()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                last = model.network[2]
                assert isinstance(last, torch.nn.Linear)
                last.bias.fill_(10)
            models.append(
                VerifiedCheckpoint(
                    seed,
                    step,
                    model,
                    "synthetic",
                    archive.state_digest(model.state_dict()),
                )
            )
            for mode in ("raw", "calibrated"):
                paths.append(
                    {
                        "seed": seed,
                        "step": step,
                        "mode": mode,
                        "temperature": 1.0,
                        "selection": {"threshold": 1.0 if mode == "raw" else 0.5},
                    }
                )
    (tmp_path / "choices.json").write_text(json.dumps({"paths": paths}))

    class Pipeline(torch.nn.Module):
        def forward(self, bits: torch.Tensor) -> torch.Tensor:
            return torch.cat((torch.zeros(len(bits), 1, dtype=torch.int8), bits), dim=1)

    pairs = parity.path_pairs(models, tmp_path, Pipeline(), device=torch.device("cpu"))
    assert tuple(pairs) == parity.PATHS
    bits = torch.zeros(32, 720, dtype=torch.int8)
    for name, (first, second) in pairs.items():
        expected = first(bits)
        assert all(
            torch.equal(a, b) for a, b in zip(expected, second(bits), strict=True)
        )
        if name.startswith("native"):
            assert bool(
                (expected[1] == (1 if name.endswith("calibrated") else 0)).all()
            )
    evidence = parity.check_paths(
        pairs,
        torch.zeros(64, 720, dtype=torch.int8),
        lambda b: torch.zeros(len(b), 1, dtype=torch.int8),
    )
    assert all(e.synthetic_rows == 722 for e in evidence)
    assert all(
        archive.state_digest(m.model.state_dict()) == m.model_state_sha256
        for m in models
    )


def test_hardware_writer_rejects_choices_before_device_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choices = tmp_path / "choices"
    choices.mkdir()
    (choices / "choices.json").write_text("{}")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("bad choices reached CUDA or model loading")

    monkeypatch.setattr(comparison, "cuda_metadata", forbidden)
    monkeypatch.setattr(comparison, "reference", forbidden)
    with pytest.raises(ValueError, match="choices hash mismatch"):
        parity.record_device_receipt(
            [],
            choices,
            tmp_path / "probes",
            tmp_path / "receipt",
            host="a800-172",
            choices_sha256=CHOICES_SHA,
            probes_sha256="c" * 64,
            ising_source=tmp_path / "source",
            ising_weights=tmp_path / "weights",
        )
    assert not (tmp_path / "receipt").exists()


def test_partial_probe_capture_retains_failure_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(selection, "verify_choices", lambda *a, **k: None)

    class BadCircuit(Circuit):
        def sample(
            self, shots: int, *, separate_observables: bool
        ) -> tuple[torch.Tensor, torch.Tensor]:
            bits, truth = super().sample(
                shots, separate_observables=separate_observables
            )
            bits[0, 0] = 2
            return bits, truth

    circuit = BadCircuit()
    output = tmp_path / "partial-probes"
    for error in (ValueError, FileExistsError):
        with pytest.raises(error):
            parity.capture_probes(
                circuit,
                tmp_path / "choices",
                [],
                output,
                expected_choices_sha256=CHOICES_SHA,
                circuit_sha256=CIRCUIT_SHA,
            )
    assert (output / "plan.json").exists()
    assert not (output / "manifest.json").exists()
    assert not (output / "physical.pt").exists()
    assert circuit.calls == [64]


def test_nonbinary_probe_rejected_even_with_consistent_file_hashes(
    probes: Path,
) -> None:
    path = probes / "physical.pt"
    payload = torch.load(path, weights_only=True)
    payload["detectors"][0, 0] = 2
    torch.save(payload, path)
    manifest = probes / "manifest.json"
    raw = json.loads(manifest.read_text())
    raw["file_sha256"] = archive.digest(path)
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="binary arrays"):
        parity.load_probes(
            probes,
            expected_manifest_sha256=archive.digest(manifest),
            choices_sha256=CHOICES_SHA,
            circuit_sha256=CIRCUIT_SHA,
        )


def test_hardware_writer_refuses_active_cpu_autocast(tmp_path: Path) -> None:
    with (
        torch.autocast("cpu", dtype=torch.bfloat16),
        pytest.raises(ValueError, match="autocast"),
    ):
        parity.record_device_receipt(
            [],
            tmp_path,
            tmp_path,
            tmp_path / "unused",
            host="a800-172",
            choices_sha256=CHOICES_SHA,
            probes_sha256="c" * 64,
            ising_source=tmp_path,
            ising_weights=tmp_path,
        )
    assert not (tmp_path / "unused").exists()
