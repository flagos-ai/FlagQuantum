"""Frozen probe and two-host parity prerequisites; no primary sampler or CLI."""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from examples.qec import compare_ising as comparison
from examples.qec import continued_assessment as inputs
from examples.qec import continued_selection as selection
from examples.qec import diagnose_predecoder as archive
from flagquantum.qec.decoders._ai.comparison import (
    Correction,
    GlobalDecode,
    decode,
    on_device,
    require_parity,
)
from flagquantum.qec.decoders._ai.predecoder import Predecoder

IMAGE = "sha256:9465e1fca0a8c22af1b2ee657c8a6ee7e9f9f8e74d873b0add84b274da58b9e2"
PROBE_SEED = 17000011
PATHS = ("global", "fast", "mask-only") + tuple(
    f"native-{s}-step-{h}-{mode}"
    for s in archive.SEEDS
    for h in (200, 2000)
    for mode in ("raw", "calibrated")
)


@dataclass(frozen=True)
class ProbeManifest:
    purpose: str
    protocol_sha256: str
    choices_sha256: str
    circuit_sha256: str
    seed: int
    physical_shots: int
    detectors: int
    observables: int
    file_sha256: str
    detectors_sha256: str
    truth_sha256: str


def capture_probes(
    circuit: Any,
    choices: Path,
    models: list[inputs.VerifiedCheckpoint],
    output: Path,
    *,
    expected_choices_sha256: str,
    circuit_sha256: str,
) -> ProbeManifest:
    """Capture the single physical probe after choices pass their frozen binding.

    Complete study input/source/weight preflight remains the caller's prerequisite.
    A second host replays this capture and must not construct another sampler.
    """
    selection.verify_choices(
        choices,
        expected_sha256=expected_choices_sha256,
        models=models,
        circuit_sha256=circuit_sha256,
    )
    return _capture_probes(
        circuit,
        output,
        choices_sha256=expected_choices_sha256,
        circuit_sha256=circuit_sha256,
        protocol_sha256=archive.digest(inputs.PROTOCOL),
        seed=PROBE_SEED,
        purpose="continued_physical_probe",
    )


def _capture_probes(
    circuit: Any,
    output: Path,
    *,
    choices_sha256: str,
    circuit_sha256: str,
    protocol_sha256: str,
    seed: int,
    purpose: str,
    verify: Callable[[], None] | None = None,
) -> ProbeManifest:
    if hashlib.sha256(str(circuit).encode()).hexdigest() != circuit_sha256:
        raise ValueError("probe circuit differs from the frozen input")
    widths = circuit.num_detectors, circuit.num_observables
    if any(type(n) is not int or n < 1 for n in widths):
        raise ValueError("probe requires positive detector/observable dimensions")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "plan.json").open("x") as stream:
        json.dump(
            {
                "choices_sha256": choices_sha256,
                "circuit_sha256": circuit_sha256,
                "seed": seed,
            },
            stream,
        )
    sampler = circuit.compile_detector_sampler(seed=seed)
    bits, truth = sampler.sample(64, separate_observables=True)
    bits = inputs._bits(torch.as_tensor(bits), shots=64, width=widths[0])
    truth = inputs._bits(torch.as_tensor(truth), shots=64, width=widths[1])
    path = output / "physical.pt"
    with path.open("xb") as stream:
        torch.save({"detectors": bits, "truth": truth}, stream)
    result = ProbeManifest(
        purpose,
        protocol_sha256,
        choices_sha256,
        circuit_sha256,
        seed,
        64,
        *widths,
        archive.digest(path),
        inputs._bit_hash(bits),
        inputs._bit_hash(truth),
    )
    if verify is not None:
        verify()
    with (output / "manifest.json").open("x") as stream:
        stream.write(json.dumps(asdict(result), indent=2) + "\n")
    return result


def load_probes(
    directory: Path,
    *,
    expected_manifest_sha256: str,
    choices_sha256: str,
    circuit_sha256: str,
) -> torch.Tensor:
    return _load_probes(
        directory,
        expected_manifest_sha256=expected_manifest_sha256,
        choices_sha256=choices_sha256,
        circuit_sha256=circuit_sha256,
        protocol_sha256=archive.digest(inputs.PROTOCOL),
        seed=PROBE_SEED,
        purpose="continued_physical_probe",
    )


def _load_probes(
    directory: Path,
    *,
    expected_manifest_sha256: str,
    choices_sha256: str,
    circuit_sha256: str,
    protocol_sha256: str,
    seed: int,
    purpose: str,
) -> torch.Tensor:
    if archive.digest(directory / "manifest.json") != expected_manifest_sha256:
        raise ValueError("probe manifest hash mismatch")
    raw = json.loads((directory / "manifest.json").read_text())
    if (
        raw["purpose"] != purpose
        or raw["protocol_sha256"] != protocol_sha256
        or raw["choices_sha256"] != choices_sha256
        or raw["circuit_sha256"] != circuit_sha256
        or raw["seed"] != seed
        or raw["physical_shots"] != 64
        or archive.digest(directory / "physical.pt") != raw["file_sha256"]
    ):
        raise ValueError("physical probe binding mismatch")
    widths = raw["detectors"], raw["observables"]
    if any(type(n) is not int or n < 1 for n in widths):
        raise ValueError("invalid probe dimensions")
    payload = torch.load(
        directory / "physical.pt", weights_only=True, map_location="cpu"
    )
    if not isinstance(payload, dict) or set(payload) != {"detectors", "truth"}:
        raise ValueError("invalid physical probe payload")
    if not all(isinstance(v, torch.Tensor) for v in payload.values()):
        raise ValueError("physical probe requires tensors")
    bits = inputs._bits(payload["detectors"], shots=64, width=widths[0])
    truth = inputs._bits(payload["truth"], shots=64, width=widths[1])
    if (
        inputs._bit_hash(bits) != raw["detectors_sha256"]
        or inputs._bit_hash(truth) != raw["truth_sha256"]
    ):
        raise ValueError("physical probe bit hash mismatch")
    return bits


def synthetic_probes(width: int) -> Iterator[torch.Tensor]:
    """All zero/one/one-hot syndromes, without materializing a dense identity."""
    if type(width) is not int or width < 1:
        raise ValueError("probe width must be a positive integer")
    yield torch.zeros(1, width, dtype=torch.int8)
    yield torch.ones(1, width, dtype=torch.int8)
    for start in range(0, width, 32):
        count = min(32, width - start)
        bits = torch.zeros(count, width, dtype=torch.int8)
        bits[torch.arange(count), torch.arange(start, start + count)] = 1
        yield bits


@dataclass(frozen=True)
class PathEvidence:
    name: str
    synthetic_rows: int
    physical_rows: int
    residual_frame_sha256: str
    physical_predictions_sha256: str


@torch.inference_mode()
def check_paths(
    pairs: Mapping[str, tuple[Correction, Correction]],
    physical: torch.Tensor,
    global_decode: GlobalDecode,
) -> list[PathEvidence]:
    """Bit oracles shared by software tests and the hardware receipt writer.

    Only physical rows require matching. Arbitrary one-hot syndromes may be
    unmatchable and are tested for full residual/frame equality alone.
    """
    return _check_paths(pairs, physical, global_decode, expected_paths=PATHS)


@torch.inference_mode()
def _check_paths(
    pairs: Mapping[str, tuple[Correction, Correction]],
    physical: torch.Tensor,
    global_decode: GlobalDecode,
    *,
    expected_paths: tuple[str, ...],
) -> list[PathEvidence]:
    if tuple(pairs) != expected_paths or physical.ndim != 2 or physical.shape[0] != 64:
        raise ValueError("requires all fifteen ordered paths and 64 physical probes")
    inputs._bits(physical, shots=64, width=physical.shape[1])
    result = []
    for name, (first, second) in pairs.items():
        hashed, predictions = hashlib.sha256(), hashlib.sha256()
        rows = 0
        for bits in synthetic_probes(physical.shape[1]):
            require_parity(bits, first, second)
            for value in first(bits):
                hashed.update(archive.tensor_bytes(value.to(torch.int8)))
            rows += len(bits)
        for start in range(0, 64, 32):
            bits = physical[start : start + 32]
            require_parity(bits, first, second, global_decode)
            predictions.update(archive.tensor_bytes(decode(bits, first, global_decode)))
            for value in first(bits):
                hashed.update(archive.tensor_bytes(value.to(torch.int8)))
        result.append(
            PathEvidence(name, rows, 64, hashed.hexdigest(), predictions.hexdigest())
        )
    return result


def path_pairs(
    models: list[inputs.VerifiedCheckpoint],
    choices: Path,
    pipeline: nn.Module,
    *,
    device: torch.device,
) -> dict[str, tuple[Correction, Correction]]:
    """Compose canonical native/Fast corrections; caller verifies frozen choices."""
    raw = json.loads((choices / "choices.json").read_text())
    paths = raw["paths"]
    expected = [
        (s, h, m)
        for s in archive.SEEDS
        for h in (200, 2000)
        for m in ("raw", "calibrated")
    ]
    if [(p["seed"], p["step"], p["mode"]) for p in paths] != expected:
        raise ValueError("requires all twelve frozen native paths")
    native = [
        (
            f"native-{model.seed}-step-{model.step}-{mode}",
            model.model,
            path["selection"]["threshold"],
            path["temperature"],
        )
        for index, model in enumerate(models)
        for mode, path in zip(
            ("raw", "calibrated"), paths[2 * index : 2 * index + 2], strict=True
        )
    ]
    return _path_pairs(native, pipeline, device=device)


def _path_pairs(
    native: list[tuple[str, Predecoder, float, float]],
    pipeline: nn.Module,
    *,
    device: torch.device,
) -> dict[str, tuple[Correction, Correction]]:
    config = comparison.Config(False, distance=9, rounds=9, a800=True)
    upstream = copy.deepcopy(pipeline).to(device).eval()
    global_only = comparison.correction("pymatching", None, pipeline)

    def mask(bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return comparison.boundary_only(bits, config), torch.zeros(
            (len(bits), 1), dtype=torch.int8, device=bits.device
        )

    pairs = {
        "global": (global_only, global_only),
        "fast": (
            comparison.correction("ising", None, pipeline),
            on_device(comparison.correction("ising", None, upstream), device),
        ),
        "mask-only": (mask, on_device(mask, device)),
    }
    accelerated: dict[int, Predecoder] = {}
    for name, model, threshold, temperature in native:
        if name in pairs:
            raise ValueError("duplicate correction path identity")
        if id(model) not in accelerated:
            accelerated[id(model)] = copy.deepcopy(model).to(device).eval()
        pairs[name] = (
            functools.partial(
                model.correct, threshold=threshold, temperature=temperature
            ),
            on_device(
                functools.partial(
                    accelerated[id(model)].correct,
                    threshold=threshold,
                    temperature=temperature,
                ),
                device,
            ),
        )
    return pairs


@dataclass(frozen=True)
class ParityReceipt:
    purpose: str
    host: str
    protocol_sha256: str
    choices_sha256: str
    circuit_sha256: str
    probes_sha256: str
    source_sha256: str
    image: str
    torch_version: str
    pipeline_state_sha256: str
    hardware: dict[str, object]
    paths: list[PathEvidence]


def _device_reference(
    source: Path,
    weights: Path,
    *,
    circuit_sha256: str,
    dem_sha256: str,
) -> tuple[dict[str, Any], comparison.Reference]:
    hardware = comparison.cuda_metadata(IMAGE)
    reference = comparison.reference(
        source,
        weights,
        comparison.Config(False, distance=9, rounds=9, a800=True),
    )
    if (
        hashlib.sha256(str(reference.circuit).encode()).hexdigest() != circuit_sha256
        or hashlib.sha256(
            comparison.training_dem(reference.circuit).to_stim_text().encode()
        ).hexdigest()
        != dem_sha256
    ):
        raise ValueError("pinned reference differs from the archived circuit/DEM")
    return hardware, reference


@torch.inference_mode()
def record_device_receipt(
    models: list[inputs.VerifiedCheckpoint],
    choices: Path,
    probes: Path,
    output: Path,
    *,
    host: str,
    choices_sha256: str,
    probes_sha256: str,
    ising_source: Path,
    ising_weights: Path,
) -> ParityReceipt:
    """Hardware-only writer: pinned reference, explicit CUDA, no CPU fallback.

    Producing-environment/dependency preflight and the formal execution lock are
    still required at the future study entry point. This function samples no data.
    """
    if host not in ("a800-172", "a800-171"):
        raise ValueError("requires a registered A800 host")
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise ValueError("parity receipts require autocast to be disabled")
    anchor = json.loads((archive.ARCHIVE / "results.json").read_text())
    circuit_sha = anchor["circuit_sha256"]
    selection.verify_choices(
        choices,
        expected_sha256=choices_sha256,
        models=models,
        circuit_sha256=circuit_sha,
    )
    if any(
        hashlib.sha256(m.model.dem_text.encode()).hexdigest()
        != anchor["training_dem_sha256"]
        for m in models
    ):
        raise ValueError("native model DEM differs from archived reference")
    physical = load_probes(
        probes,
        expected_manifest_sha256=probes_sha256,
        choices_sha256=choices_sha256,
        circuit_sha256=circuit_sha,
    )
    if output.exists():
        raise FileExistsError(output)
    hardware, reference = _device_reference(
        ising_source,
        ising_weights,
        circuit_sha256=circuit_sha,
        dem_sha256=anchor["training_dem_sha256"],
    )
    pipeline_hash = archive.state_digest(reference.pipeline.state_dict())
    output.mkdir(parents=True, exist_ok=False)
    with (output / "plan.json").open("x") as stream:
        json.dump(
            {
                "host": host,
                "choices_sha256": choices_sha256,
                "probes_sha256": probes_sha256,
            },
            stream,
        )
    evidence = check_paths(
        path_pairs(models, choices, reference.pipeline, device=torch.device("cuda:0")),
        physical,
        comparison.global_decoder(reference.circuit),
    )
    selection.verify_choices(
        choices,
        expected_sha256=choices_sha256,
        models=models,
        circuit_sha256=circuit_sha,
    )
    load_probes(
        probes,
        expected_manifest_sha256=probes_sha256,
        choices_sha256=choices_sha256,
        circuit_sha256=circuit_sha,
    )
    if archive.state_digest(reference.pipeline.state_dict()) != pipeline_hash:
        raise ValueError("parity execution mutated the reference pipeline")
    result = ParityReceipt(
        "continued_cuda_bit_parity",
        host,
        archive.digest(inputs.PROTOCOL),
        choices_sha256,
        circuit_sha,
        probes_sha256,
        archive.digest(Path(__file__)),
        IMAGE,
        str(torch.__version__),
        pipeline_hash,
        hardware,
        evidence,
    )
    with (output / "receipt.json").open("x") as stream:
        stream.write(json.dumps(asdict(result), indent=2, allow_nan=False) + "\n")
    return result


def verify_two_hosts(
    receipts: dict[str, tuple[Path, str]],
    *,
    choices_sha256: str,
    probes_sha256: str,
    circuit_sha256: str,
    probe_directory: Path,
) -> None:
    """Validate externally frozen receipt hashes before any future primary sampler."""
    if set(receipts) != {"a800-172", "a800-171"}:
        raise ValueError("requires both registered hosts")
    bits = load_probes(
        probe_directory,
        expected_manifest_sha256=probes_sha256,
        choices_sha256=choices_sha256,
        circuit_sha256=circuit_sha256,
    )
    expected_torch = json.loads((archive.ARCHIVE / "manifest.json").read_text())[
        "torch"
    ]
    _verify_two_hosts(
        receipts,
        choices_sha256=choices_sha256,
        probes_sha256=probes_sha256,
        circuit_sha256=circuit_sha256,
        width=bits.shape[1],
        expected_torch=expected_torch,
        purpose="continued_cuda_bit_parity",
        protocol_sha256=archive.digest(inputs.PROTOCOL),
        source_sha256=archive.digest(Path(__file__)),
        paths=PATHS,
    )


def _hardware_uuid(hardware: dict[str, Any]) -> str:
    if (
        hardware["device"] != "cuda:0"
        or "A800" not in hardware["name"]
        or hardware["image_id"] != IMAGE
        or hardware["tf32"] is not False
        or hardware["autocast"] is not False
        or hardware["deterministic_algorithms"] is not True
    ):
        raise ValueError("invalid isolated A800 hardware scope")
    uuid = hardware["nvidia_smi_uuid_name_driver"].split(",")[0].strip()
    if (
        re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid)
        is None
    ):
        raise ValueError("invalid GPU UUID")
    return str(uuid).lower()


def _verify_two_hosts(
    receipts: dict[str, tuple[Path, str]],
    *,
    choices_sha256: str,
    probes_sha256: str,
    circuit_sha256: str,
    width: int,
    expected_torch: str,
    purpose: str,
    protocol_sha256: str,
    source_sha256: str,
    paths: tuple[str, ...],
) -> None:
    if set(receipts) != {"a800-172", "a800-171"}:
        raise ValueError("requires both registered hosts")
    rows = []
    for host in ("a800-172", "a800-171"):
        path, expected = receipts[host]
        if archive.digest(path) != expected:
            raise ValueError("device receipt hash mismatch")
        raw = json.loads(path.read_text())
        hardware = raw["hardware"]
        if (
            raw["purpose"] != purpose
            or raw["host"] != host
            or raw["protocol_sha256"] != protocol_sha256
            or raw["choices_sha256"] != choices_sha256
            or raw["probes_sha256"] != probes_sha256
            or raw["circuit_sha256"] != circuit_sha256
            or raw["image"] != IMAGE
            or raw["source_sha256"] != source_sha256
            or [r["name"] for r in raw["paths"]] != list(paths)
            or any(
                r["physical_rows"] != 64 or r["synthetic_rows"] != width + 2
                for r in raw["paths"]
            )
        ):
            raise ValueError("device receipt binding or scope mismatch")
        if any(
            re.fullmatch(r"[0-9a-f]{64}", row[key]) is None
            for row in raw["paths"]
            for key in ("residual_frame_sha256", "physical_predictions_sha256")
        ):
            raise ValueError("invalid path evidence fingerprint")
        _hardware_uuid(hardware)
        if (
            re.fullmatch(r"[0-9a-f]{64}", raw["pipeline_state_sha256"]) is None
            or raw["torch_version"] != expected_torch
        ):
            raise ValueError("invalid device identity or producing environment")
        rows.append(raw)
    if (
        rows[0]["paths"] != rows[1]["paths"]
        or rows[0]["pipeline_state_sha256"] != rows[1]["pipeline_state_sha256"]
        or rows[0]["torch_version"] != rows[1]["torch_version"]
        or rows[0]["hardware"]["nvidia_smi_uuid_name_driver"]
        .split(",")[0]
        .strip()
        .lower()
        == rows[1]["hardware"]["nvidia_smi_uuid_name_driver"]
        .split(",")[0]
        .strip()
        .lower()
    ):
        raise ValueError("two-host device evidence does not agree or repeats a device")
