"""Pinned Kaiwu 1.3.1 client built only from documented SDK behavior."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
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
from .jobs import new_receipt
from .sdk import (
    KaiwuSDKEnvironment,
    KaiwuSDKError,
    _load_kaiwu_module,
    initialize_kaiwu_license,
)


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
        expected_version: str = "1.3.1",
        interval_minutes: int = 1,
    ) -> None:
        path = Path(checkpoint_dir).expanduser().resolve()
        if not path.is_dir():
            raise ValueError("checkpoint_dir must be an existing directory")
        if type(interval_minutes) is not int or interval_minutes < 1:
            raise ValueError("interval_minutes must be an integer of at least one")
        environment = initialize_kaiwu_license(
            credentials,
            expected_version=expected_version,
        )
        self._module = _load_kaiwu_module()
        self._checkpoint_dir = path
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
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                stored = KaiwuTaskReceipt(**raw["receipt"])
                stored_matrix = tuple(
                    tuple(float(value) for value in row) for row in raw["matrix"]
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                raise KaiwuSDKError(
                    "Existing Kaiwu recovery receipt is invalid"
                ) from None
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
                getattr(stored, field) != getattr(receipt, field)
                for field in comparable
            ):
                raise KaiwuSDKError(
                    "Existing Kaiwu recovery receipt conflicts with the request"
                )
            return stored

        encoded = (
            json.dumps(
                {"receipt": asdict(receipt), "matrix": matrix},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
        return receipt

    @contextmanager
    def _checkpoint_context(self) -> Iterator[None]:
        common = getattr(self._module, "common", None)
        manager = getattr(common, "CheckpointManager", None)
        if manager is None or not hasattr(manager, "save_dir"):
            raise KaiwuSDKError("Kaiwu SDK does not expose CheckpointManager.save_dir")
        with self._checkpoint_lock:
            previous = manager.save_dir
            manager.save_dir = str(self._checkpoint_dir)
            try:
                yield
            finally:
                manager.save_dir = previous

    def _optimizer(self, receipt: KaiwuTaskReceipt) -> Any:
        key = self._identity_key(receipt)
        existing = self._optimizers.get(key)
        if existing is not None:
            return existing
        cim = getattr(self._module, "cim", None)
        optimizer_type = getattr(cim, "CIMOptimizer", None)
        if not callable(optimizer_type):
            raise KaiwuSDKError("Kaiwu SDK does not expose cim.CIMOptimizer")
        task_mode_type = getattr(cim, "TaskMode", None)
        task_mode = getattr(task_mode_type, receipt.mode.upper(), receipt.mode)
        options: dict[str, Any] = {
            "task_name": receipt.task_name,
            "wait": False,
            "interval": self._interval_minutes,
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
        except Exception:
            schema = {"available": False, "reason": "inspection_failed"}
            self._result_schemas[key] = schema
            return schema
        schema = {
            "available": True,
            "result": self._describe_value(raw, include_mapping_fields=True),
        }
        self._result_schemas[key] = schema
        return schema

    @classmethod
    def _describe_value(
        cls, value: Any, *, include_mapping_fields: bool = False
    ) -> dict[str, Any]:
        """Return JSON-safe type/shape metadata with no raw provider values."""

        value_type = f"{type(value).__module__}.{type(value).__qualname__}"
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                return {"type": value_type, "length": len(value), "string_keys": False}
            fields = sorted(value)
            description: dict[str, Any] = {
                "type": value_type,
                "length": len(value),
                "string_keys": True,
                "fields": fields,
            }
            if include_mapping_fields:
                description["field_schemas"] = {
                    field: cls._describe_value(value[field]) for field in fields
                }
            return description
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None and dtype is not None:
            try:
                normalized_shape = [int(size) for size in shape]
            except (TypeError, ValueError):
                normalized_shape = []
            return {
                "type": value_type,
                "shape": normalized_shape,
                "dtype": str(dtype),
            }
        if isinstance(value, (list, tuple)):
            return {
                "type": value_type,
                "length": len(value),
                "element_types": sorted(
                    {
                        f"{type(element).__module__}.{type(element).__qualname__}"
                        for element in value
                    }
                ),
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
