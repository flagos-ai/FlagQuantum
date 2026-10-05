"""Recoverable Kaiwu-specific remote jobs with credential-free receipts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from .contracts import (
    KAIWU_TASK_RECEIPT_SCHEMA,
    FrozenIsingMatrix,
    KaiwuJobStatus,
    KaiwuTaskClient,
    KaiwuTaskMode,
    KaiwuTaskReceipt,
    KaiwuTaskResult,
    MatrixInput,
)


def _freeze_matrix(matrix: MatrixInput) -> FrozenIsingMatrix:
    try:
        rows: list[tuple[float, ...]] = []
        for row in matrix:
            values: list[float] = []
            for value in row:
                if isinstance(value, bool):
                    raise ValueError("boolean coefficients are not supported")
                numeric = float(value)
                values.append(0.0 if numeric == 0.0 else numeric)
            rows.append(tuple(values))
        frozen = tuple(rows)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Kaiwu matrix must be a rectangular real numeric array"
        ) from exc
    if not frozen or any(len(row) != len(frozen) for row in frozen):
        raise ValueError("Kaiwu matrix must be non-empty and square")
    if any(not math.isfinite(value) for row in frozen for value in row):
        raise ValueError("Kaiwu matrix must contain only finite values")
    if any(
        frozen[row][column] != frozen[column][row]
        for row in range(len(frozen))
        for column in range(row)
    ):
        raise ValueError("Kaiwu matrix must be exactly symmetric")
    return frozen


def _matrix_sha256(matrix: FrozenIsingMatrix) -> str:
    encoded = json.dumps(matrix, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalize_status(raw_status: str) -> KaiwuJobStatus:
    value = raw_status.strip().lower()
    if value in {"finished", "completed", "done", "success", "succeed", "succeeded"}:
        return "succeeded"
    if value in {"failed", "failure", "error"}:
        return "failed"
    if value in {"cancelled", "canceled", "stopped"}:
        return "cancelled"
    if value in {"queued", "pending", "waiting", "created", "submitted"}:
        return "queued"
    if value in {"running", "executing", "processing"}:
        return "running"
    return "unknown"


def _validate_receipt(
    receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
) -> KaiwuTaskReceipt:
    if receipt.schema != KAIWU_TASK_RECEIPT_SCHEMA:
        raise ValueError("Unsupported Kaiwu task receipt schema")
    if not receipt.task_name.strip():
        raise ValueError("Kaiwu task receipt has an empty task name")
    if receipt.matrix_size != len(matrix):
        raise ValueError("Kaiwu task receipt matrix size does not match its input")
    if receipt.matrix_sha256 != _matrix_sha256(matrix):
        raise ValueError("Kaiwu task receipt matrix identity does not match its input")
    if receipt.mode not in {"optimization", "sampling"}:
        raise ValueError("Kaiwu task receipt has an unsupported task mode")
    if receipt.requested_samples <= 0:
        raise ValueError("Kaiwu task receipt requested_samples must be positive")
    return receipt


def _recompute_energy(matrix: FrozenIsingMatrix, sample: tuple[int, ...]) -> float:
    return -sum(
        sample[row] * matrix[row][column] * sample[column]
        for row in range(len(matrix))
        for column in range(len(matrix))
    )


def _validate_result(
    result: KaiwuTaskResult,
    receipt: KaiwuTaskReceipt,
    matrix: FrozenIsingMatrix,
) -> KaiwuTaskResult:
    if result.receipt != receipt:
        raise RuntimeError("Kaiwu result does not match the submitted task receipt")
    if _normalize_status(result.raw_status) != "succeeded":
        raise RuntimeError("Kaiwu result carries a non-success provider status")
    if result.metadata.get("fallback_occurred") is not False:
        raise RuntimeError(
            "Kaiwu result must declare fallback_occurred=false explicitly"
        )
    if not result.samples:
        raise RuntimeError("Kaiwu result contains no samples")
    if len(result.samples) != len(result.energies):
        raise RuntimeError("Kaiwu result sample and energy counts differ")
    if receipt.mode == "sampling" and len(result.samples) != receipt.requested_samples:
        raise RuntimeError("Kaiwu sampling result count differs from the request")
    for sample, energy in zip(result.samples, result.energies, strict=True):
        if len(sample) != receipt.matrix_size:
            raise RuntimeError("Kaiwu result sample width differs from the matrix")
        if any(spin not in {-1, 1} for spin in sample):
            raise RuntimeError("Kaiwu result samples must contain only -1 or +1")
        if not math.isfinite(energy):
            raise RuntimeError("Kaiwu result energies must be finite")
        if not math.isclose(
            energy,
            _recompute_energy(matrix, sample),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError("Kaiwu result energy failed independent recomputation")
    return result


class KaiwuRemoteJob:
    """One submitted Kaiwu task; status and result never resubmit it."""

    def __init__(
        self,
        receipt: KaiwuTaskReceipt,
        matrix: FrozenIsingMatrix,
        client: KaiwuTaskClient,
    ) -> None:
        self._receipt = _validate_receipt(receipt, matrix)
        self._matrix = matrix
        self._client = client
        self._raw_status: str | None = None

    @property
    def receipt(self) -> KaiwuTaskReceipt:
        return self._receipt

    @property
    def raw_status(self) -> str | None:
        return self._raw_status

    def status(self) -> KaiwuJobStatus:
        self._raw_status = self._client.query_status(self._receipt, self._matrix)
        return _normalize_status(self._raw_status)

    def result(self) -> KaiwuTaskResult:
        status = self.status()
        if status != "succeeded":
            raise RuntimeError(
                f"Kaiwu task {self._receipt.task_name!r} is {status}; "
                "no result fetched"
            )
        return _validate_result(
            self._client.fetch_result(self._receipt, self._matrix),
            self._receipt,
            self._matrix,
        )

    def wait(
        self, *, timeout: float = 3600.0, poll_interval: float = 60.0
    ) -> KaiwuTaskResult:
        """Poll to completion; timeout does not cancel or resubmit the task."""

        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and non-negative")
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")
        deadline = time.monotonic() + timeout
        while True:
            status = self.status()
            if status == "succeeded":
                return _validate_result(
                    self._client.fetch_result(self._receipt, self._matrix),
                    self._receipt,
                    self._matrix,
                )
            if status in {"failed", "cancelled"}:
                raise RuntimeError(
                    f"Kaiwu task {self._receipt.task_name!r} is {status} "
                    f"({self._raw_status})"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Kaiwu task {self._receipt.task_name!r} is incomplete; "
                    "no cancellation or resubmission occurred"
                )
            time.sleep(min(poll_interval, remaining))

    def save(self, path: str | Path) -> None:
        """Persist the matrix identity needed by Kaiwu recovery without secrets."""

        payload = {
            "receipt": asdict(self._receipt),
            "matrix": self._matrix,
        }
        encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)


def submit_kaiwu_task(
    matrix: MatrixInput,
    *,
    client: KaiwuTaskClient,
    task_name: str,
    mode: KaiwuTaskMode = "optimization",
    requested_samples: int = 10,
    project_no: str | None = None,
) -> KaiwuRemoteJob:
    """Submit exactly once and return a detached Kaiwu job."""

    if not isinstance(task_name, str) or not task_name.strip():
        raise ValueError("task_name must be a non-empty string")
    if mode not in {"optimization", "sampling"}:
        raise ValueError("mode must be 'optimization' or 'sampling'")
    if type(requested_samples) is not int or requested_samples <= 0:
        raise ValueError("requested_samples must be a positive integer")
    if mode == "sampling" and not 10 <= requested_samples <= 2000:
        raise ValueError("Kaiwu sampling requested_samples must be between 10 and 2000")
    if project_no is not None and (
        not isinstance(project_no, str) or not project_no.strip()
    ):
        raise ValueError("project_no must be a non-empty string or None")
    frozen = _freeze_matrix(matrix)
    receipt = client.submit(
        frozen,
        task_name=task_name.strip(),
        mode=mode,
        requested_samples=requested_samples,
        project_no=project_no.strip() if project_no is not None else None,
    )
    expected_project = project_no.strip() if project_no is not None else None
    if (
        receipt.task_name != task_name.strip()
        or receipt.mode != mode
        or receipt.requested_samples != requested_samples
        or receipt.project_no != expected_project
    ):
        raise RuntimeError("Kaiwu submission receipt does not match the request")
    return KaiwuRemoteJob(receipt, frozen, client)


def restore_kaiwu_job(path: str | Path, *, client: KaiwuTaskClient) -> KaiwuRemoteJob:
    """Restore a task without submitting a replacement."""

    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"matrix", "receipt"}:
        raise ValueError("Invalid Kaiwu task receipt bundle")
    if not isinstance(raw["receipt"], dict):
        raise ValueError("Invalid Kaiwu task receipt fields")
    try:
        receipt = KaiwuTaskReceipt(**raw["receipt"])
    except TypeError as exc:
        raise ValueError("Invalid Kaiwu task receipt fields") from exc
    matrix = _freeze_matrix(cast(MatrixInput, raw["matrix"]))
    return KaiwuRemoteJob(receipt, matrix, client)


def new_receipt(
    *,
    task_name: str,
    matrix: FrozenIsingMatrix,
    mode: KaiwuTaskMode,
    requested_samples: int,
    project_no: str | None,
    provider_task_id: str | None = None,
    provider_target: str | None = None,
) -> KaiwuTaskReceipt:
    """Build the receipt returned by a Kaiwu task client implementation."""

    return KaiwuTaskReceipt(
        task_name=task_name,
        matrix_sha256=_matrix_sha256(matrix),
        matrix_size=len(matrix),
        mode=mode,
        requested_samples=requested_samples,
        project_no=project_no,
        submitted_at=datetime.now(timezone.utc).isoformat(),
        provider_task_id=provider_task_id,
        provider_target=provider_target,
    )
