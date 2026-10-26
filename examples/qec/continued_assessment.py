"""Verify continued-study inputs; physical execution is not exposed by this CLI."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, TypedDict, cast

import torch

from examples.qec import diagnose_predecoder as archive
from flagquantum.qec.decoders._ai.comparison import binary_array
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.training import Trainer

AUDIT = (
    archive.ROOT
    / "docs/development/evidence/qec_ai_training_response_20261025/audit.json"
)
PROTOCOL = archive.ROOT / "docs/development/QEC_AI_CONTINUED_PHYSICAL_ASSESSMENT.md"
VALIDATION_SEED = 15000011
VALIDATION_SHOTS = 131072
CHUNK = 4096


class CheckpointRecord(TypedDict):
    seed: int
    step: int
    checkpoint_sha256: str
    model_state_sha256: str


@dataclass(frozen=True)
class VerifiedCheckpoint:
    seed: int
    step: int
    model: Predecoder
    checkpoint_sha256: str
    model_state_sha256: str


def load_six(
    original: Path, continued: Path
) -> tuple[list[VerifiedCheckpoint], dict[str, str]]:
    """Verify all six archived identities without sampling or optimizer updates."""
    originals, inputs = archive.load_inputs(original)
    audit = json.loads(AUDIT.read_text())
    manifest = continued / "manifest.json"
    expected_manifest = audit["raw_file_sha256"]["full/manifest.json"]
    if archive.digest(manifest) != expected_manifest:
        raise ValueError("continued manifest hash mismatch")
    inputs = {"original/" + k: v for k, v in inputs.items()}
    inputs["continued/manifest.json"] = expected_manifest
    records = cast(list[CheckpointRecord], audit["checkpoint_audit"]["rows"])
    expected = {(r["seed"], r["step"]): r for r in records}
    loaded: list[VerifiedCheckpoint] = []
    for archived in originals:
        for step in (200, 2000):
            record = expected[(archived.seed, step)]
            path = continued / f"native-{archived.seed}-step-{step}.pt"
            file_hash = archive.digest(path)
            if file_hash != record["checkpoint_sha256"]:
                raise ValueError("continued checkpoint hash mismatch")
            raw: object = torch.load(path, map_location="cpu", weights_only=True)
            if (
                not isinstance(raw, dict)
                or type(raw.get("version")) is not int
                or raw["version"] != 1
                or type(raw.get("step")) is not int
                or raw["step"] != step
            ):
                raise ValueError("continued checkpoint version or horizon mismatch")
            if raw.get("dem") != archived.model.dem_text:
                raise ValueError("continued checkpoint DEM mismatch")
            model = copy.deepcopy(archived.model)
            Trainer(model).restore(path)
            fingerprint = archive.state_digest(model.state_dict())
            if fingerprint != record["model_state_sha256"]:
                raise ValueError("continued model tensor fingerprint mismatch")
            if not all(
                bool(torch.isfinite(v).all()) for v in model.state_dict().values()
            ):
                raise ValueError("continued model contains nonfinite values")
            if step == 200 and fingerprint != archived.model_state_sha256:
                raise ValueError("step-200 model differs from the original archive")
            model.eval()
            loaded.append(
                VerifiedCheckpoint(archived.seed, step, model, file_hash, fingerprint)
            )
            inputs["continued/" + path.name] = file_hash
    if [(m.seed, m.step) for m in loaded] != [
        (s, h) for s in archive.SEEDS for h in (200, 2000)
    ]:
        raise ValueError("requires all three seeds and both fixed horizons")
    return loaded, inputs


@dataclass(frozen=True)
class Shard:
    index: int
    shots: int
    filename: str
    file_sha256: str
    detectors_sha256: str
    observables_sha256: str


@dataclass(frozen=True)
class ValidationManifest:
    purpose: str
    protocol_sha256: str
    circuit_sha256: str
    sampler_seed: int
    shots: int
    chunk_size: int
    detectors: int
    observables: int
    shards: list[Shard]


def _bits(value: torch.Tensor, *, shots: int, width: int) -> torch.Tensor:
    if value.is_complex():
        raise ValueError("physical bits must be real")
    binary_array(value, shots=shots, width=width)
    return value.to(torch.int8).contiguous()


def _bit_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(archive.tensor_bytes(value)).hexdigest()


def capture_validation(circuit: Any, output: Path) -> ValidationManifest:
    """Internal validation-stage capture; caller must preflight all inputs first.

    The sole sampler is constructed once. Failed partial captures cannot be
    resumed or overwritten. No primary capture is available in this slice.
    """
    return _capture_validation(
        circuit,
        output,
        protocol_sha256=archive.digest(PROTOCOL),
        sampler_seed=VALIDATION_SEED,
        shots=VALIDATION_SHOTS,
        chunk_size=CHUNK,
    )


def _capture_validation(
    circuit: Any,
    output: Path,
    *,
    protocol_sha256: str,
    sampler_seed: int,
    shots: int,
    chunk_size: int,
) -> ValidationManifest:
    if (
        any(type(n) is not int or n < 1 for n in (shots, chunk_size))
        or shots % chunk_size
    ):
        raise ValueError(
            "validation shots must be positive and divisible by chunk size"
        )
    if output.exists():
        raise FileExistsError(output)
    widths = (circuit.num_detectors, circuit.num_observables)
    if any(type(n) is not int or n < 1 for n in widths):
        raise ValueError("requires positive detector/observable dimensions")
    plan = ValidationManifest(
        "validation_input_capture",
        protocol_sha256,
        hashlib.sha256(str(circuit).encode()).hexdigest(),
        sampler_seed,
        shots,
        chunk_size,
        widths[0],
        widths[1],
        [],
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "plan.json").open("x") as stream:
        stream.write(json.dumps(asdict(plan), indent=2) + "\n")
    sampler = circuit.compile_detector_sampler(seed=sampler_seed)
    records: list[Shard] = []
    for index in range(shots // chunk_size):
        detectors, observables = sampler.sample(chunk_size, separate_observables=True)
        detectors = _bits(torch.as_tensor(detectors), shots=chunk_size, width=widths[0])
        observables = _bits(
            torch.as_tensor(observables), shots=chunk_size, width=widths[1]
        )
        path = output / f"chunk-{index:04d}.pt"
        with path.open("xb") as stream:
            torch.save({"detectors": detectors, "observables": observables}, stream)
        records.append(
            Shard(
                index,
                chunk_size,
                path.name,
                archive.digest(path),
                _bit_hash(detectors),
                _bit_hash(observables),
            )
        )
    result = replace(plan, shards=records)
    with (output / "manifest.json").open("x") as stream:
        stream.write(json.dumps(asdict(result), indent=2) + "\n")
    return result


def replay_validation(
    directory: Path, *, circuit_sha256: str
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Verify a completed capture and replay its immutable, ordered input shards."""
    yield from _replay_validation(
        directory,
        circuit_sha256=circuit_sha256,
        protocol_sha256=archive.digest(PROTOCOL),
        sampler_seed=VALIDATION_SEED,
        shots=VALIDATION_SHOTS,
        chunk_size=CHUNK,
    )


def _replay_validation(
    directory: Path,
    *,
    circuit_sha256: str,
    protocol_sha256: str,
    sampler_seed: int,
    shots: int,
    chunk_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["purpose"] != "validation_input_capture"
        or manifest["circuit_sha256"] != circuit_sha256
        or manifest["sampler_seed"] != sampler_seed
        or manifest["shots"] != shots
        or manifest["chunk_size"] != chunk_size
        or manifest["protocol_sha256"] != protocol_sha256
    ):
        raise ValueError("validation manifest does not match the frozen stage")
    widths = (manifest["detectors"], manifest["observables"])
    if any(type(n) is not int or n < 1 for n in widths):
        raise ValueError("invalid validation dimensions")
    records = manifest["shards"]
    names = [f"chunk-{i:04d}.pt" for i in range(shots // chunk_size)]
    if (
        len(records) != len(names)
        or sorted(p.name for p in directory.glob("chunk-*.pt")) != names
    ):
        raise ValueError("validation shard inventory is incomplete or unexpected")
    # Verify the whole inventory before exposing the first input to selection.
    for index, (name, record) in enumerate(zip(names, records, strict=True)):
        if (
            record["index"] != index
            or record["shots"] != chunk_size
            or record["filename"] != name
            or archive.digest(directory / name) != record["file_sha256"]
        ):
            raise ValueError("validation shard order or hash mismatch")
    for name, record in zip(names, records, strict=True):
        path = directory / name
        if archive.digest(path) != record["file_sha256"]:
            raise ValueError("validation shard changed during replay")
        raw: object = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(raw, dict)
            or set(raw) != {"detectors", "observables"}
            or not all(isinstance(v, torch.Tensor) for v in raw.values())
        ):
            raise ValueError("validation shard payload is malformed")
        detectors = _bits(raw["detectors"], shots=chunk_size, width=widths[0])
        observables = _bits(raw["observables"], shots=chunk_size, width=widths[1])
        if (
            _bit_hash(detectors) != record["detectors_sha256"]
            or _bit_hash(observables) != record["observables_sha256"]
        ):
            raise ValueError("validation binary content fingerprint mismatch")
        yield detectors, observables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-directory", type=Path, required=True)
    parser.add_argument("--continued-directory", type=Path, required=True)
    args = parser.parse_args()
    models, _ = load_six(args.original_directory, args.continued_directory)
    print(
        json.dumps(
            {
                "purpose": "verify_inputs_only",
                "verified_checkpoints": len(models),
                "new_samples": 0,
                "new_optimizer_steps": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
