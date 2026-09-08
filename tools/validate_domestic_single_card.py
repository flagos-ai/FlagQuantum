#!/usr/bin/env python3
"""Run fail-closed P0-P5 validation on one attested domestic FlagOS device.

Torch-FL must be imported before PyTorch in some distributions.  Keep this
module's top level limited to the standard library and import execution code
only after the external attestation has passed policy validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform as host_platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "domestic-single-card-certification-contract.toml"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_PLACEHOLDER_TOKENS = ("replace-with", "unknown", "unverified")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    try:
        return subprocess.run(
            ("git", *args),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unavailable"


def _source_provenance() -> dict[str, Any]:
    revision = _git_output("rev-parse", "HEAD")
    if revision == "unavailable":
        revision = os.environ.get("FLAGQUANTUM_SOURCE_REVISION", "unavailable")
    status = _git_output("status", "--porcelain")
    if status == "unavailable":
        encoded = os.environ.get("FLAGQUANTUM_SOURCE_TREE_DIRTY")
        tree_dirty = None if encoded is None else encoded == "1"
    else:
        tree_dirty = bool(status)
    return {"revision": revision, "tree_dirty": tree_dirty}


def _non_placeholder(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    lowered = value.lower()
    return not any(token in lowered for token in _PLACEHOLDER_TOKENS)


def attestation_errors(
    payload: dict[str, Any], *, expected_device: str = "flagos:0"
) -> tuple[str, ...]:
    """Return blockers in a provisioner-owned physical-device attestation."""

    errors: list[str] = []
    expected = {
        "schema": "flagquantum_domestic_single_card_attestation_v1",
        "status": "verified",
        "accelerator_class": "domestic_accelerator",
        "provider": "torch_fl",
        "logical_device": expected_device,
        "evidence_source": "provisioner_owned_runtime_probe",
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        errors.append("attestation identity is not a verified domestic Torch-FL route")
    for name in ("collected_at", "provisioner"):
        if not _non_placeholder(payload.get(name)):
            errors.append(f"attestation field {name!r} is missing")

    physical = payload.get("physical_device", {})
    for name in (
        "vendor",
        "model",
        "architecture",
        "device_identifier",
        "logical_device_name",
    ):
        if not _non_placeholder(physical.get(name)):
            errors.append(f"physical-device attestation field {name!r} is missing")

    software = payload.get("software", {})
    for name in ("driver_version", "vendor_runtime_version"):
        if not _non_placeholder(software.get(name)):
            errors.append(f"software attestation field {name!r} is missing")
    if not _COMMIT.fullmatch(str(software.get("torch_fl_revision", ""))):
        errors.append("attestation requires a full Torch-FL Git revision")
    if not _DIGEST.fullmatch(str(software.get("container_image_digest", ""))):
        errors.append("attestation requires an immutable container image digest")

    route = payload.get("route_audit", {})
    required_route = {
        "provider_internal_route_audited": True,
        "logical_to_physical_device_verified": True,
        "cpu_fallback_allowed": False,
        "cpu_fallback_observed": False,
        "host_execution_observed": False,
    }
    if any(route.get(name) is not value for name, value in required_route.items()):
        errors.append("provider route audit does not prove a no-host-fallback route")
    if not _DIGEST.fullmatch(str(route.get("physical_device_probe_sha256", ""))):
        errors.append("provider route audit requires a physical-device probe digest")
    return tuple(errors)


def _phase_errors(phases: dict[str, Any], *, device: str) -> tuple[str, ...]:
    errors: list[str] = []
    expected = {
        "p0_forward",
        "p1_expectation_gradient",
        "p2_selective_double_single",
        "p3_full_double_single",
        "p4_device_double_single",
        "p5_double_single_sgd",
    }
    if set(phases) != expected:
        errors.append(
            "domestic execution evidence does not cover the exact P0-P5 matrix"
        )
        return tuple(errors)
    for name, phase in phases.items():
        if phase.get("status") != "passed" or phase.get("device") != device:
            errors.append(f"{name} did not pass on the requested logical device")
        if phase.get("provider") != "torch_fl":
            errors.append(f"{name} escaped the Torch-FL provider boundary")
        if phase.get("logical_device_residency") is not True:
            errors.append(f"{name} lacks logical-device residency evidence")

    if phases["p0_forward"].get("flagquantum_host_fallback") is not False:
        errors.append("P0 reported a FlagQuantum host fallback")
    for name in ("p1_expectation_gradient", "p2_selective_double_single"):
        if phases[name].get("flagquantum_host_fallback") is not False:
            errors.append(f"{name} reported a FlagQuantum host fallback")
    if phases["p3_full_double_single"].get("state_host_fallback") is not False:
        errors.append("P3 state evolution reported a host fallback")
    p4 = phases["p4_device_double_single"]
    if p4.get("parameter_host_fallback") is not False:
        errors.append("P4 parameter generation reported a host fallback")
    if p4.get("state_host_fallback") is not False:
        errors.append("P4 state evolution reported a host fallback")
    p5 = phases["p5_double_single_sgd"]
    if p5.get("accelerator_float64_tensor_materialized") is not False:
        errors.append("P5 materialized an accelerator float64 tensor")
    if p5.get("tensor_grad_used") is not False or p5.get("parameter_word_count") != 2:
        errors.append("P5 did not preserve the explicit two-word optimizer boundary")
    return tuple(errors)


def evidence_errors(payload: dict[str, Any]) -> tuple[str, ...]:
    """Validate generated evidence without requiring accelerator dependencies."""

    errors: list[str] = []
    if payload.get("schema") != "flagquantum_domestic_single_card_evidence_v1":
        errors.append("unexpected domestic single-card evidence schema")
    if payload.get("status") != "passed":
        errors.append("domestic single-card evidence must have passed status")
    if payload.get("artifact_class") != "hardware_execution_candidate":
        errors.append("domestic single-card evidence artifact class drifted")
    if payload.get("distribution_semantics") != "single_device_fast_path":
        errors.append("domestic single-card evidence semantics drifted")
    source = payload.get("source", {})
    if (
        not _COMMIT.fullmatch(str(source.get("revision", "")))
        or source.get("tree_dirty") is not False
    ):
        errors.append("domestic single-card evidence requires a clean exact source")

    attestation = payload.get("attestation", {})
    errors.extend(attestation_errors(attestation.get("payload", {})))
    if not _DIGEST.fullmatch(str(attestation.get("sha256", ""))):
        errors.append("domestic single-card evidence lacks attestation identity")
    errors.extend(_phase_errors(payload.get("phases", {}), device="flagos:0"))

    claims = payload.get("claims", {})
    required_true = {
        "domestic_accelerator_hardware_exercised",
        "single_device_correctness_evidence",
        "provider_route_no_cpu_fallback_attested",
        "domestic_accelerator_certification_candidate",
    }
    required_false = {
        "hardware_certification",
        "convergence_certification",
        "flagcx_collectives_validated",
        "scalability_claim_allowed",
        "performance_claim_allowed",
        "production_claim_allowed",
    }
    if any(claims.get(name) is not True for name in required_true):
        errors.append("domestic single-card candidate claims are incomplete")
    if any(claims.get(name) is not False for name in required_false):
        errors.append("domestic single-card evidence contains a forbidden promotion")
    return tuple(dict.fromkeys(errors))


def _load_validator(module_name: str) -> Callable[..., dict[str, Any]]:
    qualified = f"tools.{module_name}" if __package__ else module_name
    return importlib.import_module(qualified).validate


def _require_runtime_identity(
    *, attestation: dict[str, Any], identity: dict[str, Any], device_name: str
) -> None:
    if (
        identity.get("provider") != "torch_fl"
        or identity.get("device_type") != "flagos"
    ):
        raise RuntimeError("FlagOS runtime identity escaped the Torch-FL provider")
    vendor = identity.get("vendor")
    expected_vendor = attestation["physical_device"]["vendor"]
    if vendor is None or str(vendor).casefold() != str(expected_vendor).casefold():
        raise RuntimeError(
            "Torch-FL runtime vendor does not match the physical-device attestation"
        )
    if device_name != attestation["logical_device"]:
        raise RuntimeError(
            "requested device does not match the attested logical device"
        )
    expected_revision = attestation["software"]["torch_fl_revision"]
    if os.environ.get("FLAGQUANTUM_TORCH_FL_COMMIT") != expected_revision:
        raise RuntimeError("provisioner Torch-FL revision does not match attestation")
    expected_image = attestation["software"]["container_image_digest"]
    if os.environ.get("FLAGQUANTUM_FLAGOS_CONTAINER_IMAGE") != expected_image:
        raise RuntimeError("provisioner container digest does not match attestation")


def validate(*, attestation_path: Path, device_name: str) -> dict[str, Any]:
    """Execute the fixed P0-P5 acceptance matrix on an attested FlagOS device."""

    contract = tomllib.loads(CONTRACT.read_text(encoding="utf-8"))
    if device_name != contract["logical_device"]:
        raise RuntimeError(
            f"contract requires logical device {contract['logical_device']!r}"
        )
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    blockers = attestation_errors(attestation, expected_device=device_name)
    if blockers:
        raise RuntimeError(
            "domestic accelerator attestation rejected: " + "; ".join(blockers)
        )
    source = _source_provenance()
    if (
        not _COMMIT.fullmatch(str(source.get("revision", "")))
        or source.get("tree_dirty") is not False
    ):
        raise RuntimeError(
            "domestic accelerator evidence requires a clean exact source revision"
        )

    try:
        torch_fl = importlib.import_module("torch_fl")
    except ImportError as exc:
        raise RuntimeError("Torch-FL is required for domestic certification") from exc
    torch = importlib.import_module("torch")
    platforms = importlib.import_module("flagquantum.compute")
    if not hasattr(torch, "flagos") or not torch.flagos.is_available():
        raise RuntimeError("Torch-FL did not expose an available flagos device")

    device = platforms.resolve_platform_device(device_name)
    runtime = platforms.get_platform_runtime("flagos")
    discovered = runtime.discover()
    index = 0 if device.index is None else int(device.index)
    if index >= len(discovered):
        raise RuntimeError("attested logical FlagOS device was not discovered")
    selected = discovered[index]
    if selected.name != attestation["physical_device"]["logical_device_name"]:
        raise RuntimeError("Torch-FL device name does not match attestation")
    identity = runtime.identity().to_dict()
    _require_runtime_identity(
        attestation=attestation, identity=identity, device_name=device_name
    )

    matrix = contract["acceptance_matrix"]
    depths = tuple(int(value) for value in matrix["depths"])
    seeds = tuple(int(value) for value in matrix["seeds"])
    steps = tuple(int(value) for value in matrix["optimizer_steps"])
    phases = {
        "p0_forward": _load_validator("validate_split_real_imag_flagos")(
            device_name=device_name, depths=depths
        ),
        "p1_expectation_gradient": _load_validator(
            "validate_split_real_imag_training_flagos"
        )(device_name=device_name, depths=depths, seeds=seeds),
        "p2_selective_double_single": _load_validator(
            "validate_split_real_imag_precision_flagos"
        )(device_name=device_name, depths=depths, seeds=seeds),
        "p3_full_double_single": _load_validator(
            "validate_split_real_imag_double_single_flagos"
        )(device_name=device_name, depths=depths, seeds=seeds),
        "p4_device_double_single": _load_validator(
            "validate_split_real_imag_device_double_single_flagos"
        )(device_name=device_name, depths=depths, seeds=seeds),
        "p5_double_single_sgd": _load_validator(
            "validate_split_real_imag_optimizer_accelerator"
        )(device_name=device_name, checkpoints=steps),
    }
    phase_blockers = _phase_errors(phases, device=device_name)
    if phase_blockers:
        raise RuntimeError(
            "domestic P0-P5 validation rejected: " + "; ".join(phase_blockers)
        )
    runtime.synchronize(device)
    memory = runtime.memory_snapshot(device)
    payload = {
        "schema": "flagquantum_domestic_single_card_evidence_v1",
        "status": "passed",
        "artifact_class": "hardware_execution_candidate",
        "validation_scope": "single_domestic_accelerator_p0_p5",
        "distribution_semantics": "single_device_fast_path",
        "device": device_name,
        "provider": "torch_fl",
        "runtime_identity": identity,
        "environment": {
            "python": host_platform.python_version(),
            "platform": host_platform.platform(),
            "torch": str(torch.__version__),
            "torch_fl_package": str(getattr(torch_fl, "__version__", "unknown")),
            "visible_device_count": len(discovered),
            "selected_device_name": selected.name,
            "selected_device_memory_bytes": selected.memory_bytes,
            "memory_after_validation": {
                "allocated_bytes": memory.allocated_bytes,
                "reserved_bytes": memory.reserved_bytes,
                "free_bytes": memory.free_bytes,
                "total_bytes": memory.total_bytes,
            },
        },
        "source": source,
        "contract": {
            "path": CONTRACT.name,
            "sha256": _sha256(CONTRACT),
            "schema": contract["schema"],
        },
        "attestation": {
            "sha256": _sha256(attestation_path),
            "payload": attestation,
        },
        "phases": phases,
        "reference_policy": {
            "cpu_complex128_oracles_allowed": True,
            "p3_host_gate_encoding_is_declared": True,
            "flagquantum_execution_host_fallback_allowed": False,
            "provider_internal_cpu_fallback_allowed": False,
        },
        "claims": {
            "domestic_accelerator_hardware_exercised": True,
            "single_device_correctness_evidence": True,
            "provider_route_no_cpu_fallback_attested": True,
            "domestic_accelerator_certification_candidate": True,
            "hardware_certification": False,
            "convergence_certification": False,
            "flagcx_collectives_validated": False,
            "scalability_claim_allowed": False,
            "performance_claim_allowed": False,
            "production_claim_allowed": False,
        },
    }
    generated_errors = evidence_errors(payload)
    if generated_errors:
        raise RuntimeError(
            "generated domestic evidence rejected: " + "; ".join(generated_errors)
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--device", default="flagos:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        payload = validate(attestation_path=args.attestation, device_name=args.device)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
