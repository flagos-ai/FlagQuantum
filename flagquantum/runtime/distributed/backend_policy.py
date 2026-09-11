"""Canonical development/production distributed backend policy.

FlagQuantum keeps one user-facing API while allowing local development and
production deployments to use different distributed backends:

- development: single-process CPU simulation, JAX pmap-on-local-devices intent,
  PyTorch LocalTensor simulation;
- production: real multi-device JAX pmap and torch.distributed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch
import torch.distributed as dist

_PROFILE_ENV = "FQ_DISTRIBUTED_PROFILE"
_JAX_BACKEND_ENV = "FQ_JAX_DISTRIBUTED_BACKEND"
_TORCH_BACKEND_ENV = "FQ_TORCH_DISTRIBUTED_BACKEND"
_LOCAL_WORLD_SIZE_ENV = "FQ_LOCAL_WORLD_SIZE"


@dataclass(frozen=True)
class DistributedBackendPolicy:
    """Resolved distributed backend policy shared by development and production."""

    profile: str
    jax_backend: str
    torch_backend: str
    local_world_size: int
    requires_torchrun: bool
    requires_gpu: bool
    source: Mapping[str, str | None]

    @property
    def effective_world_size(self) -> int:
        """Rank count selected by the active development/production profile."""

        if self.profile == "production":
            if dist.is_available() and dist.is_initialized():
                return max(1, int(dist.get_world_size()))
            return max(1, int(self.source.get("WORLD_SIZE") or "1"))
        return max(1, int(self.local_world_size))

    def summary(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "jax_backend": self.jax_backend,
            "torch_backend": self.torch_backend,
            "local_world_size": self.local_world_size,
            "requires_torchrun": self.requires_torchrun,
            "requires_gpu": self.requires_gpu,
            "source": dict(self.source),
        }


@dataclass(frozen=True)
class LocalTensorShard:
    """One rank-local shard for LocalTensor development simulation."""

    rank: int
    start: int
    stop: int
    tensor: torch.Tensor

    def summary(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "start": self.start,
            "stop": self.stop,
            "shape": tuple(self.tensor.shape),
            "device": str(self.tensor.device),
            "dtype": str(self.tensor.dtype),
        }


@dataclass(frozen=True)
class LocalTensor:
    """Single-process PyTorch tensor sharding simulator.

    LocalTensor is intentionally a development backend: it models rank
    ownership and gather/scatter behavior without creating a process group.
    """

    shards: tuple[LocalTensorShard, ...]
    dim: int = 0

    @property
    def world_size(self) -> int:
        return len(self.shards)

    @classmethod
    def from_tensor(
        cls, tensor: torch.Tensor, *, world_size: int, dim: int = 0
    ) -> "LocalTensor":
        world_size = max(1, int(world_size))
        dim = int(dim)
        size = int(tensor.shape[dim])
        shards = []
        for rank in range(world_size):
            start = rank * size // world_size
            stop = (rank + 1) * size // world_size
            slices = [slice(None)] * tensor.ndim
            slices[dim] = slice(start, stop)
            shards.append(
                LocalTensorShard(
                    rank=rank,
                    start=start,
                    stop=stop,
                    tensor=tensor[tuple(slices)].clone(),
                )
            )
        return cls(shards=tuple(shards), dim=dim)

    def full_tensor(self) -> torch.Tensor:
        return torch.cat([shard.tensor for shard in self.shards], dim=self.dim)

    def map_shards(self, fn: Any) -> "LocalTensor":
        return LocalTensor(
            shards=tuple(
                LocalTensorShard(
                    rank=shard.rank,
                    start=shard.start,
                    stop=shard.stop,
                    tensor=fn(shard.tensor, shard.rank),
                )
                for shard in self.shards
            ),
            dim=self.dim,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "backend": "local_tensor",
            "world_size": self.world_size,
            "dim": self.dim,
            "shards": tuple(shard.summary() for shard in self.shards),
        }


def _env(name: str, env: Mapping[str, str] | None) -> str | None:
    return (env or os.environ).get(name)


def _normalize_profile(value: str | None) -> str:
    profile = (value or "auto").strip().lower()
    aliases = {
        "dev": "development",
        "local": "development",
        "cpu": "development",
        "prod": "production",
        "cluster": "production",
        "gpu": "production",
    }
    profile = aliases.get(profile, profile)
    if profile not in {"auto", "development", "production"}:
        raise ValueError(f"{_PROFILE_ENV} must be auto, development, or production.")
    return profile


def _auto_profile(env: Mapping[str, str] | None = None) -> str:
    world_size = int(_env("WORLD_SIZE", env) or "1")
    if world_size > 1 or (dist.is_available() and dist.is_initialized()):
        return "production"
    return "development"


def resolve_distributed_backend_policy(
    *,
    profile: str | None = None,
    env: Mapping[str, str] | None = None,
) -> DistributedBackendPolicy:
    """Resolve development/production distributed backend policy from env."""

    requested = _normalize_profile(profile or _env(_PROFILE_ENV, env))
    resolved_profile = _auto_profile(env) if requested == "auto" else requested
    local_world_size = max(
        1, int(_env(_LOCAL_WORLD_SIZE_ENV, env) or _env("WORLD_SIZE", env) or "1")
    )

    jax_override = _env(_JAX_BACKEND_ENV, env)
    torch_override = _env(_TORCH_BACKEND_ENV, env)
    if resolved_profile == "development":
        jax_backend = jax_override or "pmap_local_cpu"
        torch_backend = torch_override or "local_tensor"
        requires_torchrun = False
        requires_gpu = False
    else:
        jax_backend = jax_override or "pmap"
        torch_backend = torch_override or "torch_distributed"
        requires_torchrun = True
        requires_gpu = True

    return DistributedBackendPolicy(
        profile=resolved_profile,
        jax_backend=jax_backend,
        torch_backend=torch_backend,
        local_world_size=local_world_size,
        requires_torchrun=requires_torchrun,
        requires_gpu=requires_gpu,
        source={
            _PROFILE_ENV: profile or _env(_PROFILE_ENV, env),
            _JAX_BACKEND_ENV: jax_override,
            _TORCH_BACKEND_ENV: torch_override,
            _LOCAL_WORLD_SIZE_ENV: _env(_LOCAL_WORLD_SIZE_ENV, env),
            "WORLD_SIZE": _env("WORLD_SIZE", env),
        },
    )


def _resolve_backend_policy(options: dict[str, Any]) -> DistributedBackendPolicy:
    backend_policy = options.pop("distributed_backend_policy", None)
    distributed_profile = options.pop("distributed_profile", None)
    jax_backend = options.pop("jax_backend", None)
    torch_backend = options.pop("torch_backend", None)
    if backend_policy is not None:
        if not isinstance(backend_policy, DistributedBackendPolicy):
            raise TypeError(
                "distributed_backend_policy must be a DistributedBackendPolicy"
            )
        policy = backend_policy
    else:
        policy = resolve_distributed_backend_policy(profile=distributed_profile)
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


def _should_use_torch_distributed(
    distributed_executor: str,
    policy: DistributedBackendPolicy,
    *,
    world_size: int,
) -> bool:
    if distributed_executor == "torch":
        return True
    if distributed_executor not in {"auto", None}:
        return False
    if dist.is_available() and dist.is_initialized():
        return True
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return (
        policy.profile == "production"
        and policy.torch_backend == "torch_distributed"
        and max(int(world_size), env_world_size) > 1
    )


def distributed_backend_env_help() -> dict[str, str]:
    """Return supported environment variables for backend switching."""

    return {
        _PROFILE_ENV: "auto | development | production",
        _JAX_BACKEND_ENV: "development default: pmap_local_cpu; production default: pmap",
        _TORCH_BACKEND_ENV: "development default: local_tensor; production default: torch_distributed",
        _LOCAL_WORLD_SIZE_ENV: "local simulated rank count for development mode",
    }


__all__ = [
    "DistributedBackendPolicy",
    "LocalTensor",
    "LocalTensorShard",
    "distributed_backend_env_help",
    "resolve_distributed_backend_policy",
]
