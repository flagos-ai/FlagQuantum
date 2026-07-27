"""Backend-neutral distributed execution protocols and records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from ...core.ir import CircuitIR


@dataclass(frozen=True)
class DistributedExecutionRequest:
    program: CircuitIR
    mode: str
    world_size: int
    options: Mapping[str, Any]


@dataclass(frozen=True)
class DistributedExecutionRecord:
    backend: str
    mode: str
    world_size: int
    result: Any
    execution: Mapping[str, Any]


@runtime_checkable
class DistributedExecutor(Protocol):
    """Executor boundary; release policy classifies returned records later."""

    backend: str

    def execute(
        self, request: DistributedExecutionRequest
    ) -> DistributedExecutionRecord: ...
