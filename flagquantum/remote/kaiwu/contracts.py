"""Experimental FlagQuantum-owned contracts for Kaiwu Ising tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

KaiwuTaskMode = Literal["optimization", "sampling"]
KaiwuJobStatus = Literal[
    "queued", "running", "succeeded", "failed", "cancelled", "unknown"
]
FrozenIsingMatrix = tuple[tuple[float, ...], ...]

KAIWU_TASK_RECEIPT_SCHEMA = "flagquantum.kaiwu-task.v1"


@dataclass(frozen=True)
class KaiwuTaskReceipt:
    """Credential-free identity required to recover one Kaiwu task."""

    task_name: str
    matrix_sha256: str
    matrix_size: int
    mode: KaiwuTaskMode
    requested_samples: int
    project_no: str | None
    submitted_at: str
    provider_task_id: str | None = None
    provider_target: str | None = None
    schema: str = KAIWU_TASK_RECEIPT_SCHEMA


@dataclass(frozen=True)
class KaiwuTaskResult:
    """Validated spins and energies returned by one Kaiwu task."""

    receipt: KaiwuTaskReceipt
    samples: tuple[tuple[int, ...], ...]
    energies: tuple[float, ...]
    raw_status: str
    metadata: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )


class KaiwuTaskClient(Protocol):
    """Narrow transport interface implemented by the future Kaiwu SDK adapter."""

    def submit(
        self,
        matrix: FrozenIsingMatrix,
        *,
        task_name: str,
        mode: KaiwuTaskMode,
        requested_samples: int,
        project_no: str | None,
    ) -> KaiwuTaskReceipt: ...

    def query_status(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> str: ...

    def fetch_result(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> KaiwuTaskResult: ...


MatrixInput = Sequence[Sequence[float]]


__all__ = (
    "FrozenIsingMatrix",
    "KAIWU_TASK_RECEIPT_SCHEMA",
    "KaiwuJobStatus",
    "KaiwuTaskClient",
    "KaiwuTaskMode",
    "KaiwuTaskReceipt",
    "KaiwuTaskResult",
    "MatrixInput",
)
