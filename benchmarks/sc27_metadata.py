"""Shared identity and tensor-accounting helpers for SC27 benchmark runners."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SOURCE_COMMIT = re.compile(r"^[0-9a-f]{40}$")

import torch


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference_errors(
    *,
    value: float,
    gradients: Iterable[float],
    reference_path: Path | None,
) -> dict[str, Any]:
    if reference_path is None:
        return {
            "reference_artifact_sha256": None,
            "value_absolute_error": None,
            "value_relative_error": None,
            "gradient_max_absolute_error": None,
            "gradient_relative_l2_error": None,
            "parameter_count_matches": None,
            "passed": False,
            "unavailable_reason": "correctness_reference_not_supplied",
        }
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    correctness = payload.get("correctness", payload)
    expected_value = float(correctness["initial_value"])
    expected_gradients = torch.tensor(
        correctness["initial_gradients"], dtype=torch.float64
    ).flatten()
    actual_gradients = torch.tensor(tuple(gradients), dtype=torch.float64).flatten()
    count_matches = actual_gradients.numel() == expected_gradients.numel()
    if not count_matches:
        return {
            "reference_artifact_sha256": file_sha256(reference_path),
            "value_absolute_error": abs(float(value) - expected_value),
            "value_relative_error": None,
            "gradient_max_absolute_error": None,
            "gradient_relative_l2_error": None,
            "parameter_count_matches": False,
            "passed": False,
            "unavailable_reason": None,
        }
    delta = actual_gradients - expected_gradients
    value_abs = abs(float(value) - expected_value)
    value_rel = value_abs / max(abs(expected_value), 1e-12)
    gradient_abs = float(torch.max(torch.abs(delta)).item())
    gradient_rel = float(
        torch.linalg.vector_norm(delta).item()
        / max(torch.linalg.vector_norm(expected_gradients).item(), 1e-12)
    )
    passed = (
        value_abs <= 1e-5
        and value_rel <= 1e-5
        and gradient_abs <= 2e-4
        and gradient_rel <= 2e-4
    )
    return {
        "reference_artifact_sha256": file_sha256(reference_path),
        "value_absolute_error": value_abs,
        "value_relative_error": value_rel,
        "gradient_max_absolute_error": gradient_abs,
        "gradient_relative_l2_error": gradient_rel,
        "parameter_count_matches": True,
        "passed": passed,
        "unavailable_reason": None,
    }


def tensor_bytes(values: Iterable[torch.Tensor]) -> int:
    return sum(int(value.numel() * value.element_size()) for value in values)


def nested_tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(nested_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(nested_tensor_bytes(item) for item in value)
    return 0


def optimizer_state_bytes(optimizer: torch.optim.Optimizer | None) -> int:
    if optimizer is None:
        return 0
    return nested_tensor_bytes(optimizer.state)


def build_optimizer(
    name: str,
    parameters: Iterable[torch.Tensor],
    *,
    learning_rate: float,
) -> torch.optim.Optimizer | None:
    values = list(parameters)
    if name == "none":
        return None
    if name == "sgd":
        return torch.optim.SGD(values, lr=learning_rate)
    if name == "adam":
        return torch.optim.Adam(values, lr=learning_rate)
    raise ValueError(f"unsupported optimizer: {name}")


def _command_output(command: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def source_commit(repo_root: Path) -> str | None:
    value = _command_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if value is not None:
        return value
    injected = os.environ.get("FQ_SC27_SOURCE_COMMIT")
    return injected if injected and SOURCE_COMMIT.fullmatch(injected) else None


def source_dirty(repo_root: Path) -> bool | None:
    value = _command_output(["git", "status", "--porcelain"], cwd=repo_root)
    if value is None:
        # Empty stdout is also a valid clean tree, so distinguish git failure.
        probe = _command_output(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_root)
        if probe == "true":
            return False
        injected = os.environ.get("FQ_SC27_SOURCE_DIRTY")
        if injected is not None:
            normalized = injected.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return None
    return bool(value)


def driver_version() -> str | None:
    return _command_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
            "--id=0",
        ]
    )


def _nvidia_smi_inventory() -> list[dict[str, str]]:
    value = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,power.limit,"
            "persistence_mode,compute_mode,clocks.max.sm,clocks.max.memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if value is None:
        return []
    rows = []
    for line in value.splitlines():
        fields = [field.strip() for field in line.split(",", 9)]
        if len(fields) == 10:
            rows.append(dict(zip(
                (
                    "physical_index", "gpu_uuid", "pci_bus_id", "name",
                    "memory_mib", "power_limit_watts", "persistence_mode",
                    "compute_mode", "max_sm_clock_mhz", "max_memory_clock_mhz",
                ),
                fields,
                strict=True,
            )))
    return rows


def gpu_identity(device_index: int) -> dict[str, Any]:
    """Return stable CUDA-device identity without assuming visible=physical index."""

    properties = torch.cuda.get_device_properties(device_index)
    visible = [
        token.strip()
        for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if token.strip()
    ]
    selector = visible[device_index] if device_index < len(visible) else str(device_index)
    inventory = _nvidia_smi_inventory()
    selected = next(
        (
            row
            for row in inventory
            if row["physical_index"] == selector
            or row["gpu_uuid"] == selector
            or row["gpu_uuid"].startswith(selector)
        ),
        None,
    )
    property_uuid = getattr(properties, "uuid", None)
    property_pci = getattr(properties, "pci_bus_id", None)
    # nvidia-smi is the canonical source for physical identity. Recent Torch
    # releases may expose pci_bus_id as an integer domain/bus component (for
    # example 16) instead of a complete PCI address, and property UUID strings
    # may omit the stable ``GPU-`` prefix.
    gpu_uuid = selected.get("gpu_uuid") if selected else (
        str(property_uuid) if property_uuid else None
    )
    pci_bus_id = selected.get("pci_bus_id") if selected else (
        str(property_pci) if property_pci else None
    )
    return {
        "cuda_visible_ordinal": int(device_index),
        "cuda_visible_selector": selector,
        "physical_index": selected.get("physical_index") if selected else None,
        "gpu_uuid": gpu_uuid,
        "pci_bus_id": pci_bus_id,
        "device_name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "multiprocessor_count": int(properties.multi_processor_count),
        "power_limit_watts": selected.get("power_limit_watts") if selected else None,
        "persistence_mode": selected.get("persistence_mode") if selected else None,
        "compute_mode": selected.get("compute_mode") if selected else None,
        "max_sm_clock_mhz": selected.get("max_sm_clock_mhz") if selected else None,
        "max_memory_clock_mhz": (
            selected.get("max_memory_clock_mhz") if selected else None
        ),
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "torch_cpu_thread_count": torch.get_num_threads(),
        "communication_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES", "NCCL_SOCKET_IFNAME", "NCCL_IB_HCA",
                "NCCL_NET_GDR_LEVEL", "NCCL_ALGO", "NCCL_PROTO",
                "NCCL_P2P_LEVEL", "OMP_NUM_THREADS",
            )
        },
        "identity_complete": bool(gpu_uuid and pci_bus_id),
    }


def topology_snapshot() -> dict[str, Any]:
    matrix = _command_output(["nvidia-smi", "topo", "-m"])
    return {
        "nvidia_smi_topology_matrix": matrix,
        "nvidia_smi_topology_sha256": (
            hashlib.sha256(matrix.encode("utf-8")).hexdigest()
            if matrix is not None else None
        ),
        "captured": matrix is not None,
    }


def _interval_union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _intersection_duration(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    first, second = _interval_union(left), _interval_union(right)
    i = j = 0
    total = 0.0
    while i < len(first) and j < len(second):
        start = max(first[i][0], second[j][0])
        end = min(first[i][1], second[j][1])
        total += max(0.0, end - start)
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    return total


def cuda_profile_metrics(
    profile: Any, *, wall_seconds: float, logical_communication_bytes: int
) -> dict[str, Any]:
    """Summarize a separate CUDA-profiler step without contaminating timings."""

    cuda_events = [
        event
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and event.time_range.end > event.time_range.start
    ]
    communication = [
        event
        for event in cuda_events
        if "nccl" in event.name.lower() or "c10d" in event.name.lower()
    ]
    gradient_collectives = [
        event
        for event in communication
        if "allreduce" in event.name.lower() or "all_reduce" in event.name.lower()
    ]
    compute = [event for event in cuda_events if event not in communication]
    intervals = lambda events: [
        (float(event.time_range.start), float(event.time_range.end))
        for event in events
    ]
    all_union = _interval_union(intervals(cuda_events))
    communication_union = _interval_union(intervals(communication))
    compute_union = _interval_union(intervals(compute))
    communication_us = _duration(communication_union)
    compute_us = _duration(compute_union)
    overlap_us = _intersection_duration(communication_union, compute_union)
    gradient_collective_union = _interval_union(intervals(gradient_collectives))
    gradient_collective_us = _duration(gradient_collective_union)
    gradient_collective_overlap_us = _intersection_duration(
        gradient_collective_union, compute_union
    )
    exposed_us = max(0.0, communication_us - overlap_us)
    return {
        "measurement": "one_additional_post_protocol_cuda_profiler_training_step",
        "included_in_timing_samples": False,
        "kernel_count": len(cuda_events),
        "communication_kernel_count": len(communication),
        "profile_wall_seconds": float(wall_seconds),
        "gpu_busy_seconds_union": _duration(all_union) / 1e6,
        "compute_device_seconds_union": compute_us / 1e6,
        "communication_device_seconds_union": communication_us / 1e6,
        "communication_compute_overlap_seconds": overlap_us / 1e6,
        "exposed_communication_device_seconds": exposed_us / 1e6,
        "overlap_fraction_of_communication": (
            overlap_us / communication_us if communication_us > 0 else 0.0
        ),
        "logical_communication_bytes": int(logical_communication_bytes),
        "effective_logical_bandwidth_bytes_per_communication_second": (
            logical_communication_bytes / (communication_us / 1e6)
            if communication_us > 0 else None
        ),
        "communication_kernels_observed": bool(communication),
        "gradient_all_reduce_kernel_count": len(gradient_collectives),
        "gradient_all_reduce_device_seconds_union": gradient_collective_us / 1e6,
        "gradient_all_reduce_compute_overlap_seconds": (
            gradient_collective_overlap_us / 1e6
        ),
        "gradient_all_reduce_overlap_fraction": (
            gradient_collective_overlap_us / gradient_collective_us
            if gradient_collective_us > 0 else 0.0
        ),
    }


def source_identity(
    *,
    repo_root: Path,
    workload: dict[str, Any],
    container_digest: str | None,
    raw_log_sha256: str | None,
) -> dict[str, Any]:
    return {
        "commit": source_commit(repo_root),
        "source_dirty": source_dirty(repo_root),
        "workload_sha256": canonical_sha256(workload),
        "command": shlex.join(sys.argv),
        "container_digest": container_digest
        or os.environ.get("FQ_SC27_CONTAINER_DIGEST"),
        "raw_log_sha256": raw_log_sha256,
        "hostname": platform.node(),
    }


def parameter_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    flattened = [value.detach().reshape(-1).to(device="cpu") for value in parameters]
    if not flattened:
        return torch.empty(0)
    return torch.cat(flattened)
