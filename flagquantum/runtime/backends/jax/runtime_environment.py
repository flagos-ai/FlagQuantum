# ruff: noqa: F401, F821
"""JAX device discovery, runtime initialization, and readiness policy."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Callable, Mapping, Sequence

from ....core.ir import CircuitIR, ensure_circuit_ir
from ...distributed.backend_policy import (
    DistributedBackendPolicy,
    resolve_distributed_backend_policy,
)
from .common import communication_tier as _communication_tier
from .common import env_int as _env_int
from .common import node_count as _node_count
from .common import product_int as _product
from .common import rank_for_wire as _rank_for_wire
from .common import split_contiguous as _split_contiguous
from .release_policy import (
    attach_evidence_contract as _attach_distributed_evidence_contract,
)
from .release_policy import (
    attach_mps_backward_readiness as _attach_mps_backward_readiness,
)
from .release_policy import (
    attach_statevector_claimability as _attach_statevector_claimability,
)


def _resolve_policy(
    *,
    distributed_backend_policy: DistributedBackendPolicy | None = None,
    distributed_profile: str | None = None,
    jax_backend: str | None = None,
    torch_backend: str | None = None,
) -> DistributedBackendPolicy:
    policy = distributed_backend_policy or resolve_distributed_backend_policy(
        profile=distributed_profile
    )
    if jax_backend is None and torch_backend is None:
        return policy
    source = dict(policy.source)
    if jax_backend is not None:
        source["runtime_jax_backend"] = str(jax_backend)
    if torch_backend is not None:
        source["runtime_torch_backend"] = str(torch_backend)
    return replace(
        policy,
        jax_backend=str(jax_backend or policy.jax_backend),
        torch_backend=str(torch_backend or policy.torch_backend),
        source=source,
    )


def _resolve_world_size(
    world_size: int | None, policy: DistributedBackendPolicy
) -> int:
    if world_size is not None:
        return max(1, int(world_size))
    env_world_size = _env_int("WORLD_SIZE", 0)
    if policy.profile == "production" and env_world_size > 0:
        return max(1, env_world_size)
    if policy.profile == "development" and policy.local_world_size > 1:
        return max(1, int(policy.local_world_size))
    return 1


def _resolve_local_world_size(
    local_world_size: int | None,
    *,
    world_size: int,
    policy: DistributedBackendPolicy,
) -> int:
    if local_world_size is not None:
        return max(1, min(int(world_size), int(local_world_size)))
    if policy.local_world_size > 1:
        return max(1, min(int(world_size), int(policy.local_world_size)))
    return max(1, int(world_size))


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import error is environment-specific.
        raise RuntimeError(
            "JAX sharded statevector execution requires torch for FlagQuantum compatibility."
        ) from exc
    return torch


def _require_jax() -> tuple[Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:  # pragma: no cover - import error is environment-specific.
        raise RuntimeError("JAX sharded statevector execution requires jax.") from exc
    return jax, jnp


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _parse_device_ids(value: Any) -> int | tuple[int, ...] | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return tuple(int(item) for item in value)
    text = str(value).strip()
    if not text:
        return None
    if "," not in text:
        return int(text)
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip() != "":
            return str(value)
    return None


def initialize_jax_distributed(
    *,
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
    local_device_ids: int | Sequence[int] | str | None = None,
    cluster_detection_method: str | None = None,
    initialization_timeout: int = 300,
    heartbeat_timeout_seconds: int = 100,
    shutdown_timeout_seconds: int = 300,
    coordinator_bind_address: str | None = None,
) -> dict[str, Any]:
    """Initialize JAX multi-process runtime from explicit args or env.

    This must be called before any JAX computation or device enumeration in a
    multi-process run. Single-process calls are no-ops and keep local fast paths
    free from distributed startup.
    """

    jax, _ = _require_jax()
    try:
        existing_process_count = int(jax.process_count())
        existing_process_index = int(jax.process_index())
    except (
        Exception
    ):  # pragma: no cover - partially initialized runtimes are environment-specific.
        existing_process_count = 1
        existing_process_index = 0
    if existing_process_count > 1:
        return {
            "initialized": False,
            "already_initialized": True,
            "process_count": existing_process_count,
            "process_index": existing_process_index,
            "local_device_count": len(tuple(jax.local_devices())),
            "global_device_count": len(tuple(jax.devices())),
        }

    resolved_num_processes = (
        int(num_processes)
        if num_processes is not None
        else _env_optional_int("FQ_JAX_NUM_PROCESSES")
        or _env_optional_int("JAX_NUM_PROCESSES")
        or _env_optional_int("WORLD_SIZE")
    )
    if resolved_num_processes is None or int(resolved_num_processes) <= 1:
        return {
            "initialized": False,
            "already_initialized": False,
            "process_count": existing_process_count,
            "process_index": existing_process_index,
            "local_device_count": len(tuple(jax.local_devices())),
            "global_device_count": len(tuple(jax.devices())),
        }

    resolved_process_id = (
        int(process_id)
        if process_id is not None
        else _env_optional_int("FQ_JAX_PROCESS_ID")
        or _env_optional_int("JAX_PROCESS_ID")
        or _env_optional_int("RANK")
    )
    if resolved_process_id is None:
        raise RuntimeError(
            "JAX distributed initialization requires process_id, FQ_JAX_PROCESS_ID, JAX_PROCESS_ID, or RANK."
        )

    resolved_coordinator = coordinator_address or _env_first(
        "FQ_JAX_COORDINATOR_ADDRESS", "JAX_COORDINATOR_ADDRESS"
    )
    if resolved_coordinator is None:
        master_addr = _env_first("MASTER_ADDR")
        master_port = _env_optional_int("FQ_JAX_COORDINATOR_PORT") or _env_optional_int(
            "JAX_COORDINATOR_PORT"
        )
        if master_addr is not None and master_port is None:
            torch_port = _env_optional_int("MASTER_PORT")
            master_port = int(torch_port) + 1000 if torch_port is not None else 12355
        if master_addr is not None and master_port is not None:
            resolved_coordinator = f"{master_addr}:{int(master_port)}"
    if resolved_coordinator is None and not cluster_detection_method:
        raise RuntimeError(
            "JAX distributed initialization requires coordinator_address/FQ_JAX_COORDINATOR_ADDRESS "
            "or a cluster_detection_method supported by JAX."
        )

    resolved_local_device_ids = _parse_device_ids(
        local_device_ids
        if local_device_ids is not None
        else _env_first("FQ_JAX_LOCAL_DEVICE_IDS", "JAX_LOCAL_DEVICE_IDS")
    )
    cluster_detection_method = cluster_detection_method or _env_first(
        "FQ_JAX_CLUSTER_DETECTION_METHOD", "JAX_CLUSTER_DETECTION_METHOD"
    )
    jax.distributed.initialize(
        coordinator_address=resolved_coordinator,
        num_processes=int(resolved_num_processes),
        process_id=int(resolved_process_id),
        local_device_ids=resolved_local_device_ids,
        cluster_detection_method=cluster_detection_method,
        initialization_timeout=int(initialization_timeout),
        heartbeat_timeout_seconds=int(heartbeat_timeout_seconds),
        shutdown_timeout_seconds=int(shutdown_timeout_seconds),
        coordinator_bind_address=coordinator_bind_address,
    )
    return {
        "initialized": True,
        "already_initialized": False,
        "coordinator_address": resolved_coordinator,
        "num_processes": int(resolved_num_processes),
        "process_id": int(resolved_process_id),
        "local_device_ids": resolved_local_device_ids,
        "cluster_detection_method": cluster_detection_method,
        "process_count": int(jax.process_count()),
        "process_index": int(jax.process_index()),
        "local_device_count": len(tuple(jax.local_devices())),
        "global_device_count": len(tuple(jax.devices())),
    }


def _jax_pmap_device_assignment(
    world_size: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[int, ...]]:
    jax, _ = _require_jax()
    global_devices = tuple(jax.devices())
    if len(global_devices) < int(world_size):
        raise RuntimeError(
            f"JAX pmap statevector backward requires at least {int(world_size)} global JAX devices, "
            f"but only {len(global_devices)} are visible."
        )
    pmap_devices = tuple(global_devices[: int(world_size)])
    local_devices_all = tuple(jax.local_devices())
    local_rank_indices: list[int] = []
    local_devices: list[Any] = []
    for index, device in enumerate(pmap_devices):
        if any(device == local_device for local_device in local_devices_all):
            local_rank_indices.append(int(index))
            local_devices.append(device)
    if not local_rank_indices:
        raise RuntimeError(
            "This process does not own any device in the selected JAX pmap device assignment. "
            "Check world_size, local_device_ids, and JAX distributed initialization."
        )
    return pmap_devices, tuple(local_devices), tuple(local_rank_indices)


def _jax_complex_dtype(complex_bytes: int) -> Any:
    jax, jnp = _require_jax()
    if int(complex_bytes) == 16:
        jax.config.update("jax_enable_x64", True)
        return jnp.complex128
    return jnp.complex64


def _jax_real_dtype(complex_bytes: int) -> Any:
    jax, jnp = _require_jax()
    if int(complex_bytes) == 16:
        jax.config.update("jax_enable_x64", True)
        return jnp.float64
    return jnp.float32


def _torch_complex_dtype(complex_bytes: int) -> Any:
    torch = _require_torch()
    return torch.complex128 if int(complex_bytes) == 16 else torch.complex64


def _jax_array_device_name(array: Any) -> str:
    device = getattr(array, "device", None)
    if callable(device):
        try:
            device = device()
        except TypeError:
            pass
    return str(device)


def _resolve_jax_device(device: str | None = None) -> Any | None:
    if device is None or str(device).lower() in {"", "auto"}:
        return None
    jax, _ = _require_jax()
    requested = str(device).lower()
    platform = (
        "gpu"
        if requested.startswith(("cuda", "gpu"))
        else "cpu" if requested == "cpu" else requested
    )
    devices = jax.devices(platform)
    if not devices:
        raise RuntimeError(f"No JAX devices found for platform {platform!r}.")
    return devices[0]


def _jnp_device_put(array: Any, device: Any | None) -> Any:
    if device is None:
        return array
    jax, _ = _require_jax()
    return jax.device_put(array, device)


def _jax_array_nbytes(array: Any) -> int:
    return int(
        getattr(array, "size", 0)
        * getattr(getattr(array, "dtype", None), "itemsize", 0)
    )


def _resolve_collective_backend(
    collective_backend: str, policy: DistributedBackendPolicy
) -> str:
    backend = str(collective_backend).lower()
    if backend in {"auto", ""}:
        if policy.profile == "production" and policy.jax_backend in {
            "pmap",
            "shard_map",
        }:
            return "pmap"
        return "local_simulated"
    if backend in {"local", "local_simulated", "local_simulated_psum"}:
        return "local_simulated"
    if backend in {"pmap", "jax_pmap", "jax_pmap_psum"}:
        return "pmap"
    if backend in {"shard_map", "jax_shard_map"}:
        return "shard_map"
    raise ValueError(
        "collective_backend must be 'auto', 'local_simulated', 'pmap', or 'shard_map'."
    )


def _resolve_jax_backward_backend(
    backward_backend: str, policy: DistributedBackendPolicy
) -> str:
    backend = str(backward_backend).lower()
    if backend in {"auto", ""}:
        if policy.profile == "production" and policy.jax_backend in {
            "pmap",
            "jax_pmap",
        }:
            return "pmap"
        if policy.profile == "production" and policy.jax_backend in {
            "shard_map",
            "jax_shard_map",
        }:
            return "shard_map"
        return "local_simulated"
    if backend in {"local", "local_simulated", "local_simulated_backward"}:
        return "local_simulated"
    if backend in {"pmap", "jax_pmap", "jax_pmap_backward"}:
        return "pmap"
    if backend in {"shard_map", "jax_shard_map", "jax_shard_map_backward"}:
        return "shard_map"
    raise ValueError(
        "backward_backend must be 'auto', 'local_simulated', 'pmap', or 'shard_map'."
    )


def _jax_device_count_summary() -> dict[str, Any]:
    jax, _ = _require_jax()
    try:
        process_count = int(jax.process_count())
        process_index = int(jax.process_index())
    except (
        Exception
    ):  # pragma: no cover - older JAX or partially initialized distributed runtime.
        process_count = 1
        process_index = 0
    return {
        "local_device_count": len(tuple(jax.local_devices())),
        "global_device_count": len(tuple(jax.devices())),
        "process_count": process_count,
        "process_index": process_index,
    }


def _require_jax_production_backward_ready(
    *,
    mode: str,
    backend: str,
    world_size: int,
    blockers: Sequence[str],
) -> dict[str, Any]:
    backend = str(backend)
    device_summary = _jax_device_count_summary()
    if backend == "local_simulated":
        return {
            "backend": backend,
            "execution": "local_simulated_backward",
            "device_summary": device_summary,
            "blockers": tuple(blockers),
        }
    if blockers:
        raise RuntimeError(
            f"JAX {backend} {mode} backward is not production-ready for this workload: "
            f"{tuple(blockers)}. FlagQuantum refuses to run a replicated or local fallback."
        )
    if int(device_summary["global_device_count"]) < int(world_size):
        raise RuntimeError(
            f"JAX {backend} {mode} backward requires at least {int(world_size)} global JAX devices in this "
            f"runtime, but only {device_summary['global_device_count']} are visible. This path fails "
            "closed instead of simulating production collectives."
        )
    if backend == "shard_map":
        if int(device_summary["process_count"]) > 1:
            raise RuntimeError(
                f"JAX shard_map {mode} backward across multiple JAX processes requires global-array input "
                "construction and partitioned host callbacks that are not wired yet. Use backward_backend='pmap' "
                "for multi-process JAX execution; shard_map fails closed instead of replaying a local full state."
            )
        if int(device_summary["local_device_count"]) < int(world_size):
            raise RuntimeError(
                f"JAX shard_map {mode} backward requires at least {int(world_size)} local JAX devices for the "
                f"single-process mesh, but only {device_summary['local_device_count']} are visible."
            )
        return {
            "backend": backend,
            "execution": "jax_shard_map_backward",
            "device_summary": device_summary,
            "blockers": (),
        }
    if backend != "pmap":
        raise RuntimeError(
            f"Unknown JAX production backward backend {backend!r}; expected 'pmap' or 'shard_map'."
        )
    if int(device_summary["local_device_count"]) <= 0:
        raise RuntimeError(
            "JAX pmap backward requires at least one local JAX device on every participating process."
        )
    return {
        "backend": backend,
        "execution": "jax_pmap_backward",
        "device_summary": device_summary,
        "blockers": (),
    }
