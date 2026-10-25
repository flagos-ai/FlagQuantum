"""Matched-model frozen probes and parity composition; no device receipt or CLI.

Complete real-device receipt and trusted execution gates remain prerequisites.
CPU or synthetic evidence returned here is software verification only.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

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
