"""Matched-model frozen probes and guarded device receipts; no primary or CLI.

Trusted execution and final source acceptance remain prerequisites.
CPU or synthetic evidence returned here is software verification only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from examples.qec import compare_ising as comparison
from examples.qec import continued_parity as parity
from examples.qec import diagnose_predecoder as archive
from examples.qec import matched_checkpoints as checkpoints
from examples.qec import matched_training as training
from examples.qec import matched_validation as validation
from flagquantum.qec.decoders._ai.comparison import Correction, GlobalDecode
from flagquantum.qec.decoders._ai.training import Trainer

PROBE_SEED = 23000011
PATHS = ("global", "fast", "mask-only") + tuple(
    f"{arch}-{seed}-step-2000-{mode}"
    for seed in (11, 22, 33)
    for arch in ("cnn", "mlp")
    for mode in ("raw", "calibrated")
)


def capture_probes(
    source: Path,
    initial: Path,
    trained: Path,
    calibration: Path,
    choices: Path,
    output: Path,
    *,
    expected_choices_sha256: str,
    circuit: Any,
) -> parity.ProbeManifest:
    """Capture one 64-shot probe only after the complete frozen decision is verified."""
    if output.exists():
        raise FileExistsError(output)
    models, raw = validation.restore_choices(
        source,
        initial,
        trained,
        calibration,
        choices,
        expected_sha256=expected_choices_sha256,
    )
    if (
        circuit.num_detectors != models[0][2].model.h.shape[0]
        or circuit.num_observables != models[0][2].model.logical.shape[0]
    ):
        raise ValueError("probe dimensions differ from frozen models")
    frozen = [
        (directory, validation._files(directory))
        for directory in (initial, trained, calibration, choices)
    ]
    sources = {
        **validation._sources(),
        str(Path(__file__).relative_to(archive.ROOT)): archive.digest(Path(__file__)),
    }
    protocols = {p: archive.digest(p) for p in checkpoints.PROTOCOLS}
    states = [training._state(t) for _, _, t in models]

    def verify() -> None:
        if (
            any(validation._files(directory) != files for directory, files in frozen)
            or any(
                archive.digest(archive.ROOT / name) != sha
                for name, sha in sources.items()
            )
            or any(archive.digest(p) != sha for p, sha in protocols.items())
            or any(
                training._state(t) != state or t.model.training
                for (_, _, t), state in zip(models, states, strict=True)
            )
        ):
            raise ValueError("probe source or frozen input changed during capture")

    return parity._capture_probes(
        circuit,
        output,
        choices_sha256=expected_choices_sha256,
        circuit_sha256=raw["inputs"]["circuit_sha256"],
        protocol_sha256=archive.digest(checkpoints.PROTOCOLS[0]),
        seed=PROBE_SEED,
        purpose="matched_physical_probe",
        verify=verify,
    )


def load_probes(
    directory: Path,
    *,
    expected_manifest_sha256: str,
    choices_sha256: str,
    circuit_sha256: str,
) -> torch.Tensor:
    """Replay the same capture on the peer host without constructing a sampler."""
    return parity._load_probes(
        directory,
        expected_manifest_sha256=expected_manifest_sha256,
        choices_sha256=choices_sha256,
        circuit_sha256=circuit_sha256,
        protocol_sha256=archive.digest(checkpoints.PROTOCOLS[0]),
        seed=PROBE_SEED,
        purpose="matched_physical_probe",
    )


def path_pairs(
    models: list[tuple[int, str, Trainer]],
    choices: dict[str, Any],
    pipeline: nn.Module,
    *,
    device: torch.device,
) -> dict[str, tuple[Correction, Correction]]:
    """Compose verified models/choices; this does not authenticate vendor weights or devices."""
    if [(s, a, t.step) for s, a, t in models] != [
        (s, a, 2000) for s in (11, 22, 33) for a in ("cnn", "mlp")
    ]:
        raise ValueError("requires all six final models in registered order")
    paths = choices["paths"]
    expected = [
        (s, a, 2000, m)
        for s in (11, 22, 33)
        for a in ("cnn", "mlp")
        for m in ("raw", "calibrated")
    ]
    if [
        (p["seed"], p["architecture"], p["step"], p["mode"]) for p in paths
    ] != expected:
        raise ValueError("requires all twelve explicitly identified frozen paths")
    native = [
        (
            f"{a}-{s}-step-2000-{row['mode']}",
            t.model,
            row["selection"]["threshold"],
            row["temperature"],
        )
        for index, (s, a, t) in enumerate(models)
        for row in paths[2 * index : 2 * index + 2]
    ]
    return parity._path_pairs(native, pipeline, device=device)


def check_paths(
    pairs: Mapping[str, tuple[Correction, Correction]],
    physical: torch.Tensor,
    global_decode: GlobalDecode,
) -> list[parity.PathEvidence]:
    return parity._check_paths(pairs, physical, global_decode, expected_paths=PATHS)


def _sources() -> dict[str, str]:
    return {
        **validation._sources(),
        str(Path(__file__).relative_to(archive.ROOT)): archive.digest(Path(__file__)),
    }


def _source_hash() -> str:
    return hashlib.sha256(json.dumps(_sources(), sort_keys=True).encode()).hexdigest()


def _precision() -> None:
    if (
        torch.is_autocast_enabled("cpu")
        or torch.is_autocast_enabled("cuda")
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
        or not torch.are_deterministic_algorithms_enabled()
    ):
        raise ValueError("receipts require deterministic FP32 without TF32 or autocast")


@torch.inference_mode()
def record_device_receipt(
    source: Path,
    initial: Path,
    trained: Path,
    calibration: Path,
    choices: Path,
    probes: Path,
    output: Path,
    *,
    host: str,
    choices_sha256: str,
    probes_sha256: str,
    ising_weights: Path,
) -> dict[str, Any]:
    """Hardware-only writer; the external trusted controller must authenticate the host.

    There is no CPU fallback or sampler. Formal source/dependency/resource/lock
    and total-deadline acceptance remains mandatory before actual execution.
    """
    if host not in ("a800-172", "a800-171"):
        raise ValueError("requires a registered A800 host")
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise ValueError("receipts require autocast disabled")
    if output.exists():
        raise FileExistsError(output)
    models, frozen = validation.restore_choices(
        source, initial, trained, calibration, choices, expected_sha256=choices_sha256
    )
    expected_torch = json.loads((archive.ARCHIVE / "manifest.json").read_text())[
        "torch"
    ]
    if str(torch.__version__) != expected_torch or torch.get_num_threads() != 1:
        raise ValueError("receipt producing environment differs from pinned protocol")
    circuit_sha = frozen["inputs"]["circuit_sha256"]
    dem_sha = frozen["inputs"]["dem_sha256"]
    if any(
        hashlib.sha256(t.model.dem_text.encode()).hexdigest() != dem_sha
        for _, _, t in models
    ):
        raise ValueError("native receipt models differ from verified canonical DEM")
    physical = load_probes(
        probes,
        expected_manifest_sha256=probes_sha256,
        choices_sha256=choices_sha256,
        circuit_sha256=circuit_sha,
    )
    sources = _sources()
    source_hash = _source_hash()
    protocols = {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS}
    files = [
        (p, validation._files(p))
        for p in (initial, trained, calibration, choices, probes)
    ]
    states = [training._state(t) for _, _, t in models]
    weight_hash = archive.digest(ising_weights)
    if weight_hash != comparison.FAST_SHA256:
        raise ValueError("receipt requires the pinned official Fast weights")
    hardware, reference = parity._device_reference(
        source, ising_weights, circuit_sha256=circuit_sha, dem_sha256=dem_sha
    )
    _precision()
    if any(p.dtype != torch.float32 for p in reference.pipeline.parameters()) or any(
        not bool(torch.isfinite(v).all())
        for v in reference.pipeline.state_dict().values()
    ):
        raise ValueError("receipt pipeline must have finite FP32 parameters")
    pipeline_hash = archive.state_digest(reference.pipeline.state_dict())
    output.mkdir(parents=True, exist_ok=False)
    with (output / "plan.json").open("x") as stream:
        stream.write(
            json.dumps(
                {
                    "host": host,
                    "choices_sha256": choices_sha256,
                    "probes_sha256": probes_sha256,
                    "source_files": sources,
                    "protocol_sha256": protocols,
                },
                indent=2,
            )
            + "\n"
        )
    plan_hash = archive.digest(output / "plan.json")
    evidence = check_paths(
        path_pairs(models, frozen, reference.pipeline, device=torch.device("cuda:0")),
        physical,
        comparison.global_decoder(reference.circuit),
    )
    _precision()
    parity._hardware_uuid(hardware)
    if (
        any(validation._files(p) != inventory for p, inventory in files)
        or _sources() != sources
        or archive.digest(ising_weights) != weight_hash
        or {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS} != protocols
        or archive.state_digest(reference.pipeline.state_dict()) != pipeline_hash
        or sorted(p.name for p in output.iterdir()) != ["plan.json"]
        or archive.digest(output / "plan.json") != plan_hash
        or any(
            training._state(t) != state or t.model.training
            for (_, _, t), state in zip(models, states, strict=True)
        )
    ):
        raise ValueError("receipt execution changed frozen inputs, source or state")
    result = {
        **asdict(
            parity.ParityReceipt(
                "matched_cuda_bit_parity",
                host,
                protocols[checkpoints.PROTOCOLS[0].name],
                choices_sha256,
                circuit_sha,
                probes_sha256,
                source_hash,
                parity.IMAGE,
                str(torch.__version__),
                pipeline_hash,
                hardware,
                evidence,
            )
        ),
        "protocols": protocols,
        "source_files": sources,
        "fast_weights_sha256": weight_hash,
        "training_sha256": frozen["training_sha256"],
        "calibration_sha256": frozen["calibration_sha256"],
    }
    with (output / "receipt.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def verify_two_hosts(
    source: Path,
    initial: Path,
    trained: Path,
    calibration: Path,
    choices: Path,
    probes: Path,
    receipts: dict[str, tuple[Path, str]],
    *,
    choices_sha256: str,
    probes_sha256: str,
) -> None:
    """Check externally frozen trusted receipts; parsing alone cannot authenticate SSH hosts."""
    _, frozen = validation.restore_choices(
        source, initial, trained, calibration, choices, expected_sha256=choices_sha256
    )
    circuit_sha = frozen["inputs"]["circuit_sha256"]
    physical = load_probes(
        probes,
        expected_manifest_sha256=probes_sha256,
        choices_sha256=choices_sha256,
        circuit_sha256=circuit_sha,
    )
    protocols = {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS}
    source_files = _sources()
    parity._verify_two_hosts(
        receipts,
        choices_sha256=choices_sha256,
        probes_sha256=probes_sha256,
        circuit_sha256=circuit_sha,
        width=physical.shape[1],
        expected_torch=json.loads((archive.ARCHIVE / "manifest.json").read_text())[
            "torch"
        ],
        purpose="matched_cuda_bit_parity",
        protocol_sha256=protocols[checkpoints.PROTOCOLS[0].name],
        source_sha256=_source_hash(),
        paths=PATHS,
    )
    for path, expected in receipts.values():
        if archive.digest(path) != expected:
            raise ValueError("receipt changed during validation")
        raw = json.loads(path.read_text())
        if (
            raw["protocols"] != protocols
            or raw["source_files"] != source_files
            or raw["fast_weights_sha256"] != comparison.FAST_SHA256
            or raw["training_sha256"] != frozen["training_sha256"]
            or raw["calibration_sha256"] != frozen["calibration_sha256"]
        ):
            raise ValueError(
                "receipt model, protocol, source or weight binding mismatch"
            )
