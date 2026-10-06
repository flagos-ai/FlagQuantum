"""Recoverable Kaiwu-specific remote jobs with credential-free receipts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import stat
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from numbers import Real
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


def _validate_private_directory(path: Path, *, description: str) -> None:
    try:
        metadata = path.lstat()
    except OSError:
        raise ValueError(
            f"{description} must be an existing private, non-symlink directory"
        ) from None
    if (
        path.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_mode & 0o077
    ):
        raise ValueError(
            f"{description} must be an existing private, non-symlink directory"
        )


def _verify_open_directory_binding(
    path: Path,
    descriptor_metadata: os.stat_result,
    *,
    description: str,
) -> None:
    try:
        visible = path.lstat()
    except OSError:
        raise ValueError(f"{description} changed during receipt access") from None
    if (
        path.is_symlink()
        or not stat.S_ISDIR(visible.st_mode)
        or visible.st_mode & 0o077
        or (visible.st_dev, visible.st_ino)
        != (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
    ):
        raise ValueError(f"{description} changed during receipt access")


def _open_private_directory(
    path: Path, *, description: str
) -> tuple[int, os.stat_result]:
    _validate_private_directory(path, description=description)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise ValueError(f"{description} changed during receipt access") from None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & 0o077:
            raise ValueError(f"{description} changed during receipt access")
        _verify_open_directory_binding(path, metadata, description=description)
        return descriptor, metadata
    except BaseException:
        os.close(descriptor)
        raise


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting ambiguous duplicate fields."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Kaiwu receipt JSON contains duplicate object keys")
        result[key] = value
    return result


def _write_private_json_exclusive(path: str | Path, payload: object) -> None:
    destination = Path(path)
    parent = destination.parent
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    directory_descriptor, directory_metadata = _open_private_directory(
        parent, description="Kaiwu receipt parent"
    )
    temporary_name = f".{destination.name}.{secrets.token_hex(16)}.tmp"
    temporary_descriptor: int | None = None
    published = False
    try:
        temporary_descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        with os.fdopen(temporary_descriptor, "wb") as stream:
            temporary_descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _verify_open_directory_binding(
            parent,
            directory_metadata,
            description="Kaiwu receipt parent",
        )
        os.link(
            temporary_name,
            destination.name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        published = True
        _verify_open_directory_binding(
            parent,
            directory_metadata,
            description="Kaiwu receipt parent",
        )
        if os.fstat(directory_descriptor).st_mode & 0o077:
            raise ValueError("Kaiwu receipt parent changed during receipt access")
        os.fsync(directory_descriptor)
        published = False
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        if published:
            try:
                os.unlink(destination.name, dir_fd=directory_descriptor)
                os.fsync(directory_descriptor)
            except OSError:
                pass
        try:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        finally:
            os.close(directory_descriptor)


def _read_private_json(path: str | Path) -> Any:
    source = Path(path)
    directory_descriptor, directory_metadata = _open_private_directory(
        source.parent, description="Kaiwu receipt parent"
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            descriptor = os.open(source.name, flags, dir_fd=directory_descriptor)
        except OSError:
            raise ValueError(
                "Kaiwu receipt must be a private, regular, non-symlink file"
            ) from None
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
                raise ValueError(
                    "Kaiwu receipt must be a private, regular, non-symlink file"
                )
            with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
                descriptor = -1
                result = json.load(
                    stream, object_pairs_hook=_reject_duplicate_json_keys
                )
            _verify_open_directory_binding(
                source.parent,
                directory_metadata,
                description="Kaiwu receipt parent",
            )
            return result
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    finally:
        os.close(directory_descriptor)


def _freeze_matrix(matrix: MatrixInput) -> FrozenIsingMatrix:
    try:
        rows: list[tuple[float, ...]] = []
        for row in matrix:
            values: list[float] = []
            for value in row:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError("coefficients must be real numeric scalars")
                numeric = float(value)
                values.append(0.0 if numeric == 0.0 else numeric)
            rows.append(tuple(values))
        frozen = tuple(rows)
    except Exception:
        raise ValueError(
            "Kaiwu matrix must be a rectangular real numeric array"
        ) from None
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
    if (
        not isinstance(receipt.task_name, str)
        or not receipt.task_name.strip()
        or receipt.task_name != receipt.task_name.strip()
        or not receipt.task_name.isprintable()
    ):
        raise ValueError("Kaiwu task receipt has an invalid task name")
    if type(receipt.matrix_size) is not int or receipt.matrix_size != len(matrix):
        raise ValueError("Kaiwu task receipt matrix size does not match its input")
    if receipt.matrix_sha256 != _matrix_sha256(matrix):
        raise ValueError("Kaiwu task receipt matrix identity does not match its input")
    if not isinstance(receipt.mode, str) or receipt.mode not in {
        "optimization",
        "sampling",
    }:
        raise ValueError("Kaiwu task receipt has an unsupported task mode")
    if type(receipt.requested_samples) is not int or receipt.requested_samples <= 0:
        raise ValueError("Kaiwu task receipt requested_samples must be positive")
    if receipt.mode == "sampling" and not 10 <= receipt.requested_samples <= 2000:
        raise ValueError(
            "Kaiwu sampling receipt requested_samples must be between 10 and 2000"
        )
    if receipt.project_no is not None and (
        not isinstance(receipt.project_no, str)
        or not receipt.project_no.strip()
        or receipt.project_no != receipt.project_no.strip()
        or not receipt.project_no.isprintable()
    ):
        raise ValueError("Kaiwu task receipt has an invalid project number")
    try:
        submitted_at = datetime.fromisoformat(receipt.submitted_at)
    except (TypeError, ValueError):
        submitted_at = None
    if submitted_at is None or submitted_at.utcoffset() != timedelta(0):
        raise ValueError("Kaiwu task receipt must have an aware UTC submission time")
    for field_name in ("provider_task_id", "provider_target"):
        value = getattr(receipt, field_name)
        if value is not None and (
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
            or not value.isprintable()
        ):
            raise ValueError(f"Kaiwu task receipt has an invalid {field_name}")
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
    if not isinstance(result, KaiwuTaskResult):
        raise RuntimeError("Kaiwu client returned an invalid result object")
    if result.receipt != receipt:
        raise RuntimeError("Kaiwu result does not match the submitted task receipt")
    if (
        not isinstance(result.raw_status, str)
        or not result.raw_status.strip()
        or not result.raw_status.isprintable()
    ):
        raise RuntimeError("Kaiwu result carries an invalid provider status")
    if _normalize_status(result.raw_status) != "succeeded":
        raise RuntimeError("Kaiwu result carries a non-success provider status")
    if not isinstance(result.metadata, Mapping):
        raise RuntimeError("Kaiwu result metadata must be a mapping")
    if result.metadata.get("fallback_occurred") is not False:
        raise RuntimeError(
            "Kaiwu result must declare fallback_occurred=false explicitly"
        )
    if not isinstance(result.samples, tuple) or not result.samples:
        raise RuntimeError("Kaiwu result contains no samples")
    if not isinstance(result.energies, tuple):
        raise RuntimeError("Kaiwu result energies must be a tuple")
    if len(result.samples) != len(result.energies):
        raise RuntimeError("Kaiwu result sample and energy counts differ")
    if receipt.mode == "sampling" and len(result.samples) != receipt.requested_samples:
        raise RuntimeError("Kaiwu sampling result count differs from the request")
    for sample, energy in zip(result.samples, result.energies, strict=True):
        if not isinstance(sample, tuple):
            raise RuntimeError("Kaiwu result samples must be tuples")
        if len(sample) != receipt.matrix_size:
            raise RuntimeError("Kaiwu result sample width differs from the matrix")
        if any(type(spin) is not int or spin not in {-1, 1} for spin in sample):
            raise RuntimeError("Kaiwu result samples must contain only -1 or +1")
        if isinstance(energy, bool) or not isinstance(energy, Real):
            raise RuntimeError("Kaiwu result energies must be finite real numbers")
        if not math.isfinite(float(energy)):
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
        raw_status = self._client.query_status(self._receipt, self._matrix)
        if (
            not isinstance(raw_status, str)
            or not raw_status.strip()
            or not raw_status.isprintable()
        ):
            raise RuntimeError("Kaiwu client returned an invalid provider status")
        self._raw_status = raw_status
        return _normalize_status(self._raw_status)

    def result(self) -> KaiwuTaskResult:
        status = self.status()
        if status != "succeeded":
            raise RuntimeError(
                f"Kaiwu task {self._receipt.task_name!r} is {status}; no result fetched"
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

        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, Real)
            or not math.isfinite(float(timeout))
            or timeout < 0
        ):
            raise ValueError("timeout must be finite and non-negative")
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, Real)
            or not math.isfinite(float(poll_interval))
            or poll_interval <= 0
        ):
            raise ValueError("poll_interval must be finite and positive")
        numeric_timeout = float(timeout)
        numeric_poll_interval = float(poll_interval)
        deadline = time.monotonic() + numeric_timeout
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
            time.sleep(min(numeric_poll_interval, remaining))

    def save(self, path: str | Path) -> None:
        """Persist the matrix identity needed by Kaiwu recovery without secrets."""

        payload = {
            "receipt": asdict(self._receipt),
            "matrix": self._matrix,
        }
        _write_private_json_exclusive(path, payload)


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

    if (
        not isinstance(task_name, str)
        or not task_name.strip()
        or not task_name.strip().isprintable()
    ):
        raise ValueError("task_name must be a non-empty printable string")
    if not isinstance(mode, str) or mode not in {"optimization", "sampling"}:
        raise ValueError("mode must be 'optimization' or 'sampling'")
    if type(requested_samples) is not int or requested_samples <= 0:
        raise ValueError("requested_samples must be a positive integer")
    if mode == "sampling" and not 10 <= requested_samples <= 2000:
        raise ValueError("Kaiwu sampling requested_samples must be between 10 and 2000")
    if project_no is not None and (
        not isinstance(project_no, str)
        or not project_no.strip()
        or not project_no.strip().isprintable()
    ):
        raise ValueError("project_no must be a non-empty printable string or None")
    frozen = _freeze_matrix(matrix)
    receipt = client.submit(
        frozen,
        task_name=task_name.strip(),
        mode=mode,
        requested_samples=requested_samples,
        project_no=project_no.strip() if project_no is not None else None,
    )
    if not isinstance(receipt, KaiwuTaskReceipt):
        raise RuntimeError("Kaiwu client returned an invalid task receipt")
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

    try:
        raw = _read_private_json(path)
    except json.JSONDecodeError:
        raise ValueError("Invalid Kaiwu task receipt bundle") from None
    if not isinstance(raw, dict) or set(raw) != {"matrix", "receipt"}:
        raise ValueError("Invalid Kaiwu task receipt bundle")
    if not isinstance(raw["receipt"], dict):
        raise ValueError("Invalid Kaiwu task receipt fields")
    try:
        receipt = KaiwuTaskReceipt(**raw["receipt"])
    except TypeError:
        raise ValueError("Invalid Kaiwu task receipt fields") from None
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
