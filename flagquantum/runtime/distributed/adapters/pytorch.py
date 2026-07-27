"""Mandatory PyTorch-native baseline for the backend-neutral executor API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..protocols import (
    DistributedExecutionRecord,
    DistributedExecutionRequest,
)


@dataclass(frozen=True)
class PyTorchDistributedExecutor:
    runner: Callable[..., Any]
    backend: str = "pytorch"

    def execute(
        self, request: DistributedExecutionRequest
    ) -> DistributedExecutionRecord:
        result = self.runner(
            request.program,
            mode=request.mode,
            world_size=request.world_size,
            **dict(request.options),
        )
        summary = result.summary() if hasattr(result, "summary") else {}
        return DistributedExecutionRecord(
            backend=self.backend,
            mode=request.mode,
            world_size=request.world_size,
            result=result,
            execution=dict(summary),
        )
