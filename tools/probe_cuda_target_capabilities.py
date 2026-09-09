#!/usr/bin/env python
"""Record a single-device CUDA statevector capability snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch

import flagquantum as fq
from flagquantum.compute import get_platform_runtime
from flagquantum.compute.cuda_target_capabilities import (
    CUDA_STATEVECTOR_WORKLOAD,
    cuda_statevector_capability_snapshot,
)
from flagquantum.core.numerics import (
    default_accuracy_requirement,
    default_precision_plan,
)
from flagquantum.runtime import run_native
from flagquantum.runtime.numerical_validation import certify_statevector_local_p0
from flagquantum.runtime.operator_probes import preflight_statevector_local_p0

ROOT = Path(__file__).resolve().parents[1]


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _git_revision() -> str:
    declared = os.environ.get("FLAGQUANTUM_SOURCE_REVISION")
    if declared:
        return declared
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def probe(*, captured_at: datetime, ttl: timedelta) -> dict[str, Any]:
    runtime = get_platform_runtime("cuda")
    devices = runtime.discover()
    if len(devices) != 1:
        raise RuntimeError(
            "CUDA snapshot requires exactly one visible device; set "
            "CUDA_VISIBLE_DEVICES to one device"
        )
    device_record = devices[0]
    device = device_record.device
    properties = torch.cuda.get_device_properties(device)
    device_uuid = str(getattr(properties, "uuid", ""))
    if not device_uuid:
        raise RuntimeError("CUDA provider did not expose a physical device UUID")

    circuit = fq.Circuit(3, device=device, dtype=torch.complex128)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rx(2, theta=0.125)
    state, _ = run_native(
        circuit,
        mode="statevector",
        device=device,
        return_plan=True,
    )
    runtime.synchronize(device)
    preflight_report = preflight_statevector_local_p0(
        device=device,
        dtype="complex128",
        provider=runtime.identity().provider,
        refresh=True,
    )
    preflight_report.require_supported()
    accuracy = default_accuracy_requirement("complex128")
    precision_plan = default_precision_plan("complex128")
    numerical_report = certify_statevector_local_p0(
        device=device,
        dtype="complex128",
        provider=runtime.identity().provider,
        accuracy_requirement=accuracy,
        precision_plan=precision_plan,
        refresh=True,
    )
    numerical_report.require_accepted()
    if state.device.type != "cuda" or state.dtype != torch.complex128:
        raise RuntimeError("CUDA statevector result lost device or complex128 dtype")
    precision = precision_plan.to_dict()
    expected_precision = {
        "complex_representation": "native_complex",
        "parameter_dtype": "float64",
        "gate_generation_dtype": "complex128",
        "state_storage_dtype": "complex128",
        "kernel_compute_dtype": "complex128",
        "reduction_dtype": "complex128",
        "gradient_dtype": "float64",
    }
    mismatches = {
        name: precision.get(name)
        for name, expected in expected_precision.items()
        if precision.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(f"CUDA precision plan mismatch: {mismatches}")

    memory = runtime.memory_snapshot(device)
    if memory.free_bytes is None:
        raise RuntimeError("CUDA provider did not expose available memory")
    identity = runtime.identity()
    source_revision = _git_revision()
    target_revision = f"sm_{properties.major}{properties.minor}"
    environment = {
        "hostname_sha256": hashlib.sha256(platform.node().encode("utf-8")).hexdigest(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "device_name": device_record.name,
        "device_uuid": device_uuid,
        "total_memory_bytes": properties.total_memory,
        "source_revision": source_revision,
    }
    environment_id = _canonical_sha256(environment)
    evidence = {
        "schema": "flagquantum.cuda_statevector_probe.v1",
        "status": "passed",
        "provider": identity.to_dict(),
        "environment": environment,
        "scope": {
            "device": str(device),
            "dtype": "complex128",
            "workload": CUDA_STATEVECTOR_WORKLOAD,
            "world_size": 1,
            "node_count": 1,
        },
        "observations": {
            "state_device": str(state.device),
            "state_dtype": str(state.dtype).removeprefix("torch."),
            "memory_available_bytes": memory.free_bytes,
            "operator_preflight": preflight_report.to_dict(),
            "numerical_validation": numerical_report.to_dict(),
            "precision_plan": precision,
        },
        "claim_blockers": [
            "hidden_cpu_fallback_not_audited",
            "multi_gpu_not_measured",
            "multi_node_not_measured",
            "production_performance_not_measured",
        ],
    }
    evidence_sha256 = _canonical_sha256(evidence)
    evidence_id = f"cuda-statevector-{evidence_sha256[:16]}"
    snapshot = cuda_statevector_capability_snapshot(
        target_id=f"cuda-gpu-{device_uuid}",
        provider_version=f"torch-{torch.__version__}-cuda-{torch.version.cuda}",
        target_revision=target_revision,
        environment_id=environment_id,
        device_id=f"cuda:{device.index}:{device_uuid}",
        memory_available_bytes=memory.free_bytes,
        evidence_id=evidence_id,
        evidence_sha256=evidence_sha256,
        captured_at=captured_at,
        ttl=ttl,
    )
    return {
        "schema": "flagquantum.cuda_target_capability_artifact.v1",
        "evidence_sha256": evidence_sha256,
        "evidence": evidence,
        "snapshot": snapshot.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ttl-minutes", type=int, default=30)
    args = parser.parse_args()
    if args.ttl_minutes < 1:
        parser.error("--ttl-minutes must be positive")
    payload = probe(
        captured_at=datetime.now(timezone.utc),
        ttl=timedelta(minutes=args.ttl_minutes),
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
