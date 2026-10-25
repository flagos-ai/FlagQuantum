"""Container-bound zero-sample preflight for the matched study.

No formal stage runner or CLI is exposed. The trusted SSH controller must supply
and freeze the real inspect document; parsing a host label is not attestation.
Complete workflow acceptance remains required before any samples or updates.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from examples.qec import compare_ising as comparison
from examples.qec import continued_parity as device
from examples.qec import diagnose_predecoder as archive
from examples.qec import ising_layout as layout
from examples.qec import matched_checkpoints as checkpoints
from examples.qec import matched_parity as parity
from examples.qec import matched_primary as primary
from examples.qec import run_continued_assessment as coordinator


@dataclass(frozen=True)
class Options:
    ising_source: Path
    ising_weights: Path
    inspect: Path
    inspect_sha256: str
    host: str
    study: Path
    dependency_lock: Path


def _fingerprints() -> dict[str, str]:
    result = primary._sources()
    for path in (Path(__file__), Path(coordinator.__file__)):
        result[str(path.relative_to(archive.ROOT))] = archive.digest(path)
    return result


def preflight(options: Options) -> dict[str, Any]:
    """Verify the inspected container, pinned layout and reference without draws.

    The returned metadata is a software/environment preflight, never a device
    prediction receipt, formal checkpoint bundle or study launch authorization.
    """
    checkpoints._sha256(options.inspect_sha256)
    if options.inspect.is_symlink():
        raise ValueError("controller inspect must not be a symlink")
    raw = coordinator._container_scope(
        options.inspect,
        expected_sha256=options.inspect_sha256,
        host=options.host,
        readonly_paths=(
            archive.ROOT,
            options.ising_source,
            options.ising_weights,
            options.inspect,
            options.dependency_lock,
        ),
    )
    if raw["HostConfig"].get("AutoRemove") is not True:
        raise ValueError("matched containers must be automatically removed")
    # Old checkpoints/arrays are neither required nor admitted for fresh models.
    stem = "qec_ai_ising_a800_20261025/"
    coordinator._source_evidence(
        {
            stem + "a800_172/manifest.json",
            stem + "a800_172/results.json",
            stem + "requirements.lock",
        }
    )
    lock_sha = archive.digest(options.dependency_lock)
    if lock_sha != archive.digest(
        archive.ROOT / "docs/development/evidence" / stem / "requirements.lock"
    ):
        raise ValueError("producing dependency lock mismatch")
    anchor = json.loads((archive.ARCHIVE / "manifest.json").read_text())
    if (
        platform.python_version() != anchor["python"]
        or str(torch.__version__) != anchor["torch"]
    ):
        raise ValueError("preflight runtime differs from the pinned environment")
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise ValueError("preflight requires autocast disabled")
    if (
        archive.digest(options.ising_weights) != comparison.FAST_SHA256
        or comparison.source_digest(options.ising_source)
        != comparison.ISING_SOURCE_SHA256
    ):
        raise ValueError("preflight requires pinned vendor source and Fast weights")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    sources = _fingerprints()
    protocols = {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS}
    metadata = {
        p: archive.digest(p)
        for p in (archive.ARCHIVE / "manifest.json", archive.ARCHIVE / "results.json")
    }
    with torch.random.fork_rng(devices=[]):
        template, inputs = layout.build_model(options.ising_source)
        layout.verify_encoding(template)
        hardware, reference = device._device_reference(
            options.ising_source,
            options.ising_weights,
            circuit_sha256=str(inputs["circuit_sha256"]),
            dem_sha256=str(inputs["dem_sha256"]),
        )
    uuid = device._hardware_uuid(hardware)
    requests = raw["HostConfig"]["DeviceRequests"]
    if (
        len(requests) != 1
        or len(requests[0]["DeviceIDs"]) != 1
        or str(requests[0]["DeviceIDs"][0]).strip().lower() != uuid
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("controller must bind the sole visible A800 UUID")
    parity._precision()
    if any(p.dtype != torch.float32 for p in reference.pipeline.parameters()) or any(
        not bool(torch.isfinite(v).all())
        for v in reference.pipeline.state_dict().values()
    ):
        raise ValueError("preflight reference must have finite FP32 parameters")
    if hashlib.sha256(template.dem_text.encode()).hexdigest() != inputs["dem_sha256"]:
        raise ValueError("preflight native layout changed its DEM identity")
    if (
        torch.get_num_threads() != 1
        or torch.get_num_interop_threads() != 1
        or _fingerprints() != sources
        or {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS} != protocols
        or any(archive.digest(p) != sha for p, sha in metadata.items())
        or archive.digest(options.inspect) != options.inspect_sha256
        or archive.digest(options.dependency_lock) != lock_sha
        or archive.digest(options.ising_weights) != comparison.FAST_SHA256
        or comparison.source_digest(options.ising_source)
        != comparison.ISING_SOURCE_SHA256
    ):
        raise ValueError("preflight inputs, metadata, source or protocols changed")
    return {
        "purpose": "matched_container_preflight",
        "host": options.host,
        "container_id": raw["Id"],
        "controller_inspect_sha256": options.inspect_sha256,
        "protocols": protocols,
        "source_files": sources,
        "dependency_lock_sha256": lock_sha,
        "fast_weights_sha256": comparison.FAST_SHA256,
        "ising_source_sha256": comparison.ISING_SOURCE_SHA256,
        "inputs": inputs,
        "hardware": hardware,
        "pipeline_state_sha256": archive.state_digest(reference.pipeline.state_dict()),
        "new_samples": 0,
        "new_optimizer_steps": 0,
        "formal_launch_authorized": False,
    }
