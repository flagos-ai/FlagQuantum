"""Pinned Kaiwu 1.3.1 client built only from documented SDK behavior."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any

from ._credentials import KaiwuCredentials
from .contracts import (
    FrozenIsingMatrix,
    KaiwuTaskMode,
    KaiwuTaskReceipt,
    KaiwuTaskResult,
)
from .jobs import (
    _freeze_matrix,
    _open_private_directory,
    _read_private_json,
    _verify_open_directory_binding,
    _write_private_json_exclusive,
    new_receipt,
)
from .sdk import (
    KaiwuSDKEnvironment,
    KaiwuSDKError,
    KaiwuSDKUnavailableError,
    _initialize_preflighted_kaiwu_license,
    _preflight_kaiwu_sdk,
)

_SAFE_SCHEMA_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}")
_MAX_SCHEMA_FIELDS = 64
_MAX_SCHEMA_SEQUENCE_TYPES = 64
_MAX_SCHEMA_DIMENSIONS = 16
_PINNED_SDK_VERSION = "1.3.1"
_PINNED_1_3_1_TASK_MODES: dict[KaiwuTaskMode, str] = {
    "optimization": "quota",
    "sampling": "sample",
}


class KaiwuSDKClient:
    """Kaiwu 1.3.1 task client using documented checkpoint idempotency.

    The SDK documents ``task_name + ising_matrix`` as task identity and returns
    ``None`` from ``solve`` while that task is incomplete. This client calls
    ``solve`` once for initial submission and again only to query that same
    identity. It records only a value-free structural description of the
    documented ``get_task_result`` dictionary and does not interpret its
    undocumented fields.

    Consequently, provider task IDs and provider-reported target labels remain
    unavailable. Real acceptance must stay closed until a pinned SDK response
    can establish those fields.
    """

    _checkpoint_lock = RLock()

    def __init__(
        self,
        *,
        checkpoint_dir: str | Path,
        credentials: KaiwuCredentials | None = None,
        expected_version: str = _PINNED_SDK_VERSION,
        interval_minutes: int = 1,
    ) -> None:
        candidate = Path(checkpoint_dir).expanduser()
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError(
                "checkpoint_dir must be an existing, non-symlink directory"
            )
        if candidate.stat().st_mode & 0o077:
            raise ValueError("checkpoint_dir must not be accessible by group or others")
        descriptor, checkpoint_metadata = _open_private_directory(
            candidate, description="checkpoint_dir"
        )
        try:
            path = candidate.resolve(strict=True)
            _verify_open_directory_binding(
                candidate,
                checkpoint_metadata,
                description="checkpoint_dir",
            )
            resolved_metadata = path.lstat()
            if (
                path.is_symlink()
                or not stat.S_ISDIR(resolved_metadata.st_mode)
                or resolved_metadata.st_mode & 0o077
                or (resolved_metadata.st_dev, resolved_metadata.st_ino)
                != (checkpoint_metadata.st_dev, checkpoint_metadata.st_ino)
            ):
                raise ValueError("checkpoint_dir changed during initialization")
        finally:
            os.close(descriptor)
        if type(interval_minutes) is not int or interval_minutes < 1:
            raise ValueError("interval_minutes must be an integer of at least one")
        if expected_version != _PINNED_SDK_VERSION:
            raise ValueError(
                "KaiwuSDKClient supports only the pinned Kaiwu 1.3.1 contract"
            )
        try:
            current_checkpoint = path.lstat()
        except OSError:
            raise ValueError("checkpoint_dir changed during initialization") from None
        if (
            path.is_symlink()
            or not stat.S_ISDIR(current_checkpoint.st_mode)
            or current_checkpoint.st_mode & 0o077
            or (current_checkpoint.st_dev, current_checkpoint.st_ino)
            != (checkpoint_metadata.st_dev, checkpoint_metadata.st_ino)
        ):
            raise ValueError("checkpoint_dir changed during initialization")
        module = _preflight_kaiwu_sdk(expected_version=expected_version)
        common = getattr(module, "common", None)
        manager = getattr(common, "CheckpointManager", None)
        cim = getattr(module, "cim", None)
        optimizer_type = getattr(cim, "CIMOptimizer", None)
        if manager is None or not hasattr(manager, "save_dir"):
            raise KaiwuSDKUnavailableError(
                "Kaiwu SDK does not expose CheckpointManager.save_dir"
            )
        if not callable(optimizer_type):
            raise KaiwuSDKUnavailableError("Kaiwu SDK does not expose cim.CIMOptimizer")
        environment = _initialize_preflighted_kaiwu_license(
            module,
            credentials,
            expected_version=expected_version,
        )
        self._module = module
        self._checkpoint_dir = path
        self._checkpoint_identity = (
            checkpoint_metadata.st_dev,
            checkpoint_metadata.st_ino,
        )
        self._interval_minutes = interval_minutes
        self._environment = environment
        self._optimizers: dict[str, Any] = {}
        self._solutions: dict[str, tuple[tuple[int, ...], ...]] = {}
        self._result_schemas: dict[str, dict[str, Any]] = {}

    @property
    def environment(self) -> KaiwuSDKEnvironment:
        return self._environment

    @staticmethod
    def _identity_key(receipt: KaiwuTaskReceipt) -> str:
        return f"{receipt.task_name}:{receipt.matrix_sha256}"

    def recovery_receipt_path(self, receipt: KaiwuTaskReceipt) -> Path:
        """Return the deterministic, credential-free recovery bundle path."""

        digest = hashlib.sha256(self._identity_key(receipt).encode()).hexdigest()
        return self._checkpoint_dir / f"flagquantum-kaiwu-{digest}.json"

    def _load_or_persist_receipt(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> KaiwuTaskReceipt:
        path = self.recovery_receipt_path(receipt)
        with self._checkpoint_lock:
            self._validate_checkpoint_binding()
            if path.exists() or path.is_symlink():
                stored = self._load_recovery_receipt(path, receipt, matrix)
                self._validate_checkpoint_binding()
                return stored
            try:
                _write_private_json_exclusive(
                    path, {"receipt": asdict(receipt), "matrix": matrix}
                )
                self._validate_checkpoint_binding()
                return receipt
            except FileExistsError:
                stored = self._load_recovery_receipt(path, receipt, matrix)
                self._validate_checkpoint_binding()
                return stored

    def _validate_checkpoint_binding(self) -> None:
        try:
            metadata = self._checkpoint_dir.lstat()
        except OSError:
            raise KaiwuSDKError("Kaiwu checkpoint directory binding changed") from None
        if (
            self._checkpoint_dir.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_mode & 0o077
            or (metadata.st_dev, metadata.st_ino) != self._checkpoint_identity
        ):
            raise KaiwuSDKError("Kaiwu checkpoint directory binding changed")

    @staticmethod
    def _load_recovery_receipt(
        path: Path, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> KaiwuTaskReceipt:
        try:
            raw = _read_private_json(path)
            if not isinstance(raw, dict) or set(raw) != {"receipt", "matrix"}:
                raise ValueError("unexpected recovery bundle fields")
            stored = KaiwuTaskReceipt(**raw["receipt"])
            stored_matrix = _freeze_matrix(raw["matrix"])
            submitted_at = datetime.fromisoformat(stored.submitted_at)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise KaiwuSDKError("Existing Kaiwu recovery receipt is invalid") from None
        if (
            submitted_at.utcoffset() != timedelta(0)
            or stored.provider_task_id is not None
            or stored.provider_target is not None
        ):
            raise KaiwuSDKError("Existing Kaiwu recovery receipt is invalid")
        comparable = (
            "task_name",
            "matrix_sha256",
            "matrix_size",
            "mode",
            "requested_samples",
            "project_no",
            "schema",
        )
        if stored_matrix != matrix or any(
            getattr(stored, field) != getattr(receipt, field) for field in comparable
        ):
            raise KaiwuSDKError(
                "Existing Kaiwu recovery receipt conflicts with the request"
            )
        return stored

    @contextmanager
    def _checkpoint_context(self) -> Iterator[None]:
        common = getattr(self._module, "common", None)
        manager = getattr(common, "CheckpointManager", None)
        if manager is None or not hasattr(manager, "save_dir"):
            raise KaiwuSDKError("Kaiwu SDK does not expose CheckpointManager.save_dir")
        with self._checkpoint_lock:
            self._validate_checkpoint_binding()
            previous = manager.save_dir
            manager.save_dir = str(self._checkpoint_dir)
            try:
                yield
            finally:
                manager.save_dir = previous
                self._validate_checkpoint_binding()

    def _optimizer(self, receipt: KaiwuTaskReceipt) -> Any:
        key = self._identity_key(receipt)
        existing = self._optimizers.get(key)
        if existing is not None:
            return existing
        cim = getattr(self._module, "cim", None)
        optimizer_type = getattr(cim, "CIMOptimizer", None)
        if not callable(optimizer_type):
            raise KaiwuSDKError("Kaiwu SDK does not expose cim.CIMOptimizer")
        task_mode = _PINNED_1_3_1_TASK_MODES.get(receipt.mode)
        if task_mode is None:
            raise KaiwuSDKError("Kaiwu task receipt has an unsupported mode")
        options: dict[str, Any] = {
            "task_name": receipt.task_name,
            "wait": False,
            "interval": self._interval_minutes,
            # Kaiwu 1.3.1 documents the legacy values ``quota`` and ``sample``.
            # The 1.4.1 ``optimization``/``sampling`` vocabulary belongs in a
            # separately pinned adapter rather than a runtime guess here.
            "task_mode": task_mode,
            "sample_number": receipt.requested_samples,
        }
        if receipt.project_no is not None:
            options["project_no"] = receipt.project_no
        try:
            with self._checkpoint_context():
                optimizer = optimizer_type(**options)
        except Exception:
            raise KaiwuSDKError(
                "Kaiwu CIM optimizer initialization failed; vendor details were redacted"
            ) from None
        self._optimizers[key] = optimizer
        return optimizer

    def _solve_identity(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> tuple[tuple[int, ...], ...] | None:
        self._validate_checkpoint_binding()
        stored = self._load_recovery_receipt(
            self.recovery_receipt_path(receipt), receipt, matrix
        )
        self._validate_checkpoint_binding()
        if stored != receipt:
            raise KaiwuSDKError(
                "Kaiwu task receipt differs from its authoritative recovery bundle"
            )
        key = self._identity_key(receipt)
        cached = self._solutions.get(key)
        if cached is not None:
            return cached
        try:
            import numpy as np
        except ImportError:
            raise KaiwuSDKError("Kaiwu SDK client requires NumPy") from None
        optimizer = self._optimizer(receipt)
        try:
            with self._checkpoint_context():
                raw_solution = optimizer.solve(np.asarray(matrix, dtype=np.float64))
        except Exception:
            raise KaiwuSDKError(
                "Kaiwu CIM task operation failed; vendor details were redacted"
            ) from None
        if raw_solution is None:
            return None
        array = np.asarray(raw_solution)
        if array.ndim != 2 or array.shape[1] != receipt.matrix_size:
            raise KaiwuSDKError("Kaiwu CIM returned an invalid solution shape")
        if not bool(np.all((array == -1) | (array == 1))):
            raise KaiwuSDKError("Kaiwu CIM returned values outside the spin domain")
        solutions = tuple(tuple(int(spin) for spin in row) for row in array.tolist())
        self._solutions[key] = solutions
        return solutions

    def submit(
        self,
        matrix: FrozenIsingMatrix,
        *,
        task_name: str,
        mode: KaiwuTaskMode,
        requested_samples: int,
        project_no: str | None,
    ) -> KaiwuTaskReceipt:
        receipt = new_receipt(
            task_name=task_name,
            matrix=matrix,
            mode=mode,
            requested_samples=requested_samples,
            project_no=project_no,
        )
        receipt = self._load_or_persist_receipt(receipt, matrix)
        self._solve_identity(receipt, matrix)
        return receipt

    def query_status(self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix) -> str:
        return (
            "Completed"
            if self._solve_identity(receipt, matrix) is not None
            else "Pending"
        )

    def fetch_result(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> KaiwuTaskResult:
        samples = self._solve_identity(receipt, matrix)
        if samples is None:
            raise KaiwuSDKError("Kaiwu CIM task is incomplete")
        result_schema = self._inspect_result_schema(receipt, matrix)
        energies = tuple(self._energy(matrix, sample) for sample in samples)
        return KaiwuTaskResult(
            receipt=receipt,
            samples=samples,
            energies=energies,
            raw_status="Completed",
            metadata={
                "fallback_occurred": False,
                "kaiwu_sdk_version": self._environment.sdk_version,
                "task_identity_basis": "task_name+ising_matrix",
                "provider_task_id_available": False,
                "provider_target_available": False,
                "provider_result_schema": result_schema,
            },
        )

    def _inspect_result_schema(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> dict[str, Any]:
        """Describe documented result fields without retaining provider values."""

        key = self._identity_key(receipt)
        cached = self._result_schemas.get(key)
        if cached is not None:
            return cached
        optimizer = self._optimizer(receipt)
        getter = getattr(optimizer, "get_task_result", None)
        if not callable(getter):
            schema = {"available": False, "reason": "method_unavailable"}
            self._result_schemas[key] = schema
            return schema
        try:
            import numpy as np

            with self._checkpoint_context():
                raw = getter(np.asarray(matrix, dtype=np.float64))
            result_description = self._describe_value(raw, include_mapping_fields=True)
        except Exception:
            schema = {"available": False, "reason": "inspection_failed"}
            self._result_schemas[key] = schema
            return schema
        schema = {
            "available": True,
            "result": result_description,
        }
        self._result_schemas[key] = schema
        return schema

    @classmethod
    def _describe_value(
        cls, value: Any, *, include_mapping_fields: bool = False
    ) -> dict[str, Any]:
        """Return JSON-safe type/shape metadata with no raw provider values."""

        value_type = f"{type(value).__module__}.{type(value).__qualname__}"
        if _SAFE_SCHEMA_NAME.fullmatch(value_type) is None:
            value_type = "unavailable"
        if isinstance(value, dict):
            field_count = len(value)
            description: dict[str, Any] = {
                "type": value_type,
                "length": field_count,
            }
            if field_count > _MAX_SCHEMA_FIELDS:
                description.update(
                    {
                        "string_keys": None,
                        "fields_safe": False,
                        "field_limit": _MAX_SCHEMA_FIELDS,
                    }
                )
                return description
            if not all(isinstance(key, str) for key in value):
                description.update({"string_keys": False, "fields_safe": False})
                return description
            fields = sorted(value)
            description["string_keys"] = True
            fields_safe = all(
                _SAFE_SCHEMA_NAME.fullmatch(field) is not None for field in fields
            )
            description["fields_safe"] = fields_safe
            if not fields_safe:
                description["field_limit"] = _MAX_SCHEMA_FIELDS
                return description
            description["fields"] = fields
            if include_mapping_fields:
                description["field_schemas"] = {
                    field: cls._describe_value(value[field]) for field in fields
                }
            return description
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None and dtype is not None:
            shape_safe = True
            try:
                raw_shape = list(
                    itertools.islice(iter(shape), _MAX_SCHEMA_DIMENSIONS + 1)
                )
                if len(raw_shape) > _MAX_SCHEMA_DIMENSIONS:
                    raise ValueError("shape has too many dimensions")
                normalized_shape = [int(size) for size in raw_shape]
            except (TypeError, ValueError):
                normalized_shape = []
                shape_safe = False
            dtype_name = str(dtype)
            dtype_safe = _SAFE_SCHEMA_NAME.fullmatch(dtype_name) is not None
            return {
                "type": value_type,
                "shape": normalized_shape,
                "shape_safe": shape_safe,
                "dtype": dtype_name if dtype_safe else "unavailable",
                "dtype_safe": dtype_safe,
            }
        if isinstance(value, (list, tuple)):
            inspected = value[:_MAX_SCHEMA_SEQUENCE_TYPES]
            return {
                "type": value_type,
                "length": len(value),
                "element_types": sorted(
                    {
                        f"{type(element).__module__}.{type(element).__qualname__}"
                        for element in inspected
                        if _SAFE_SCHEMA_NAME.fullmatch(
                            f"{type(element).__module__}.{type(element).__qualname__}"
                        )
                        is not None
                    }
                ),
                "element_type_scan_limit": _MAX_SCHEMA_SEQUENCE_TYPES,
            }
        if isinstance(value, (str, bytes)):
            return {"type": value_type, "length": len(value)}
        return {"type": value_type}

    @staticmethod
    def _energy(matrix: FrozenIsingMatrix, sample: tuple[int, ...]) -> float:
        energy = -sum(
            sample[row] * matrix[row][column] * sample[column]
            for row in range(len(matrix))
            for column in range(len(matrix))
        )
        if not math.isfinite(energy):
            raise KaiwuSDKError("Kaiwu result energy is not finite")
        return float(energy)
