"""Process-group lifecycle and placement context for distributed execution."""

from __future__ import annotations

import datetime
import os
import socket
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ...providers.platform import get_platform_runtime
from .flagos_runtime import activate_flagos_device, is_flagos_request
from .identity import DistributedIdentity, build_distributed_identity


@dataclass(frozen=True)
class TorchDistributedContext:
    """torch.distributed execution context used by FlagQuantum distributed modes."""

    rank: int
    world_size: int
    local_rank: int
    backend: str
    device: torch.device
    initialized: bool
    initialized_by_flagquantum: bool = False
    node_rank: int = 0
    local_world_size: int = 1
    node_count: int = 1
    hostname: str = ""
    identity: DistributedIdentity | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "local_world_size": self.local_world_size,
            "node_rank": self.node_rank,
            "node_count": self.node_count,
            "hostname": self.hostname,
            "backend": self.backend,
            "device": str(self.device),
            "initialized": self.initialized,
            "initialized_by_flagquantum": self.initialized_by_flagquantum,
            "distributed_identity": (
                self.identity.to_dict() if self.identity is not None else None
            ),
        }


def torch_distributed_is_available() -> bool:
    """Return whether torch.distributed can be used in this runtime."""

    return bool(dist.is_available())


def _default_device_type() -> str:
    return "cuda" if get_platform_runtime("cuda").is_available() else "cpu"


def _infer_backend(
    device: torch.device | str | None = None,
    backend: str | None = None,
    *,
    logical_device_type: str | None = None,
) -> str:
    explicit = str(backend).strip().lower() if backend is not None else None
    device_type = logical_device_type or torch.device(
        device or _default_device_type()
    ).type
    if device_type == "flagos":
        if explicit not in (None, "flagos"):
            raise ValueError(
                "flagos distributed execution requires backend='flagos'; "
                f"received {backend!r}"
            )
        return "flagos"
    if explicit == "flagos":
        raise ValueError("backend='flagos' requires device='flagos:<local_rank>'")
    if explicit is not None:
        return explicit
    return (
        "nccl"
        if device_type == "cuda" and get_platform_runtime("cuda").is_available()
        else "gloo"
    )


def _normalized_active_backend() -> str:
    return str(dist.get_backend()).strip().lower()


def _resolve_local_world_size(world_size: int) -> int:
    raw = os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("NPROC_PER_NODE")
    if raw is None:
        return max(1, int(world_size))
    return max(1, min(int(world_size), int(raw)))


def _resolve_node_rank(rank: int, local_world_size: int) -> int:
    raw = os.environ.get("GROUP_RANK") or os.environ.get("NODE_RANK")
    if raw is not None:
        return max(0, int(raw))
    return int(rank) // max(1, int(local_world_size))


def _node_count(world_size: int, local_world_size: int) -> int:
    return max(
        1,
        (max(1, int(world_size)) + max(1, int(local_world_size)) - 1)
        // max(1, int(local_world_size)),
    )


def _rank_node(rank: int, local_world_size: int) -> int:
    return int(rank) // max(1, int(local_world_size))


def _communication_tier(src_rank: int, dst_rank: int, *, local_world_size: int) -> str:
    return (
        "intra_node"
        if _rank_node(src_rank, local_world_size)
        == _rank_node(dst_rank, local_world_size)
        else "inter_node"
    )


def _rank_placement_summary(
    context: TorchDistributedContext | None, *, world_size: int | None = None
) -> dict[str, Any]:
    resolved_world_size = int(
        world_size if world_size is not None else (context.world_size if context else 1)
    )
    local_world_size = context.local_world_size if context else resolved_world_size
    node_count = (
        context.node_count
        if context
        else _node_count(resolved_world_size, local_world_size)
    )
    return {
        "rank": context.rank if context else 0,
        "local_rank": context.local_rank if context else 0,
        "world_size": resolved_world_size,
        "local_world_size": local_world_size,
        "node_rank": context.node_rank if context else 0,
        "node_count": node_count,
        "hostname": context.hostname if context else "",
        "device": str(context.device) if context else "cpu",
        "distributed_identity": (
            context.identity.to_dict()
            if context is not None and context.identity is not None
            else None
        ),
    }


def init_torch_distributed(
    *,
    backend: str | None = None,
    init_method: str = "env://",
    rank: int | None = None,
    world_size: int | None = None,
    local_rank: int | None = None,
    device: torch.device | str | None = None,
    force_initialize: bool = False,
    timeout_seconds: float | None = None,
) -> TorchDistributedContext:
    """Initialize or attach to a torch.distributed process group."""

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", env_rank))
    world_size = int(world_size if world_size is not None else env_world_size)
    rank = int(rank if rank is not None else env_rank)
    local_rank = int(local_rank if local_rank is not None else env_local_rank)
    local_world_size = _resolve_local_world_size(world_size)
    node_rank = _resolve_node_rank(rank, local_world_size)
    node_count = _node_count(world_size, local_world_size)
    hostname = socket.gethostname()
    platform_identity: Mapping[str, Any] = {}
    flagos_requested = is_flagos_request(device, backend, local_rank=local_rank)
    if flagos_requested:
        resolved_device, platform_identity = activate_flagos_device(local_rank)
    else:
        resolved_device = torch.device(device or _default_device_type())
    if resolved_device.type == "cuda":
        torch.cuda.set_device(local_rank)
        resolved_device = torch.device("cuda", local_rank)
    resolved_backend = _infer_backend(
        resolved_device,
        backend,
        logical_device_type="flagos" if flagos_requested else resolved_device.type,
    )

    if dist.is_initialized():
        active_backend = _normalized_active_backend()
        if active_backend != resolved_backend:
            raise RuntimeError(
                "active torch.distributed backend does not match the requested "
                f"route: active={active_backend!r}, requested={resolved_backend!r}"
            )

    initialized_by_flagquantum = False
    if (world_size > 1 or force_initialize) and not dist.is_initialized():
        dist.init_process_group(
            backend=resolved_backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=(
                None
                if timeout_seconds is None
                else datetime.timedelta(seconds=float(timeout_seconds))
            ),
        )
        initialized_by_flagquantum = True
    if dist.is_initialized():
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())
        local_world_size = _resolve_local_world_size(world_size)
        node_rank = _resolve_node_rank(rank, local_world_size)
        node_count = _node_count(world_size, local_world_size)
    identity = build_distributed_identity(
        outer_backend=resolved_backend,
        logical_device=(
            f"flagos:{local_rank}" if flagos_requested else str(resolved_device)
        ),
        rank=rank,
        world_size=world_size,
        process_group_initialized=dist.is_initialized(),
        platform_identity=platform_identity,
    )
    return TorchDistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        backend=resolved_backend,
        device=resolved_device,
        initialized=dist.is_initialized(),
        initialized_by_flagquantum=initialized_by_flagquantum,
        node_rank=node_rank,
        local_world_size=local_world_size,
        node_count=node_count,
        hostname=hostname,
        identity=identity,
    )


def destroy_torch_distributed() -> None:
    """Destroy the active torch.distributed process group if initialized."""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
