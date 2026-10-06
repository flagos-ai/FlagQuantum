"""Bounded synchronous sampler compatibility for Kaiwu PyTorch Plugin."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from threading import RLock
from typing import TYPE_CHECKING, cast

from ...remote.kaiwu import (
    KaiwuRemoteJob,
    KaiwuTaskClient,
    KaiwuTaskReceipt,
    KaiwuTaskResult,
    submit_kaiwu_task,
)
from .matrix import (
    IntegerPrecisionReport,
    KaiwuInteropError,
    MatrixLike,
    canonicalize_ising_matrix,
    prepare_integer_precision,
)

if TYPE_CHECKING:
    import numpy as np
    import torch


@dataclass(frozen=True, slots=True)
class KaiwuTransferRecord:
    """One observed host-side matrix/sample transfer through the sampler."""

    input_type: str
    input_device: str
    input_dtype: str
    matrix_shape: tuple[int, int]
    original_matrix_sha256: str
    submission_matrix_sha256: str
    canonical_device: str
    canonical_dtype: str
    submission_storage: str
    returned_storage: str
    returned_dtype: str
    returned_shape: tuple[int, int]
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class KaiwuPrecisionEvidence:
    """One original matrix's explicit reduction and submitted identity."""

    original_matrix_sha256: str
    submission_matrix_sha256: str
    source_type: str
    source_dtype: str
    normalized_dtype: str
    normalized_min: float
    normalized_max: float
    symmetry_normalization: str
    rounding_policy: str
    scale_factor: float
    target_min: int
    target_max: int
    max_abs_error: float
    mean_abs_error: float


class KaiwuSampler:
    """Expose ``solve(ising_matrix)`` while delegating tasks to Remote.

    The adapter is synchronous because Kaiwu PyTorch Plugin currently calls
    ``sampler.solve(...)`` synchronously. Waiting is bounded, identical matrices
    are deduplicated within this sampler instance, and a hard remote-call budget
    fails before an excess submission. Precision reduction is disabled unless
    ``integer_target_range`` is explicitly provided.
    """

    def __init__(
        self,
        *,
        client: KaiwuTaskClient,
        task_name: str,
        project_no: str | None = None,
        requested_samples: int = 10,
        timeout: float = 3600.0,
        poll_interval: float = 60.0,
        max_remote_calls: int = 1,
        integer_target_range: tuple[int, int] | None = None,
    ) -> None:
        if (
            not isinstance(task_name, str)
            or not task_name.strip()
            or not task_name.strip().isprintable()
        ):
            raise ValueError("task_name must be a non-empty printable string")
        if type(max_remote_calls) is not int or max_remote_calls <= 0:
            raise ValueError("max_remote_calls must be a positive integer")
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
        if type(requested_samples) is not int or not 10 <= requested_samples <= 2000:
            raise ValueError("requested_samples must be between 10 and 2000")
        if integer_target_range is not None and (
            not isinstance(integer_target_range, tuple)
            or len(integer_target_range) != 2
        ):
            raise ValueError("integer_target_range must be a two-integer tuple or None")
        if integer_target_range is not None:
            target_min, target_max = integer_target_range
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in integer_target_range
            ):
                raise ValueError("integer_target_range values must be integers")
            if target_min >= 0 or target_max <= 0:
                raise ValueError("integer_target_range must straddle zero")
        if project_no is not None and (
            not isinstance(project_no, str)
            or not project_no.strip()
            or not project_no.strip().isprintable()
        ):
            raise ValueError("project_no must be a non-empty printable string or None")

        self._client = client
        self._task_name = task_name.strip()
        self._project_no = project_no.strip() if project_no is not None else None
        self._requested_samples = requested_samples
        self._timeout = float(timeout)
        self._poll_interval = float(poll_interval)
        self._max_remote_calls = max_remote_calls
        self._integer_target_range = integer_target_range
        self._solve_lock = RLock()
        self._cache: dict[str, tuple[tuple[int, ...], ...]] = {}
        self._jobs: dict[str, KaiwuRemoteJob] = {}
        self._receipts: list[KaiwuTaskReceipt] = []
        self._remote_call_count = 0
        self._last_job: KaiwuRemoteJob | None = None
        self._last_result: KaiwuTaskResult | None = None
        self._last_precision_report: IntegerPrecisionReport | None = None
        self._precision_reports: dict[str, IntegerPrecisionReport] = {}
        self._precision_evidence: dict[str, KaiwuPrecisionEvidence] = {}
        self._transfer_records: list[KaiwuTransferRecord] = []

    @property
    def remote_call_count(self) -> int:
        return self._remote_call_count

    @property
    def client(self) -> KaiwuTaskClient:
        """Return the Remote client bound to this draft interoperability adapter."""

        return self._client

    @property
    def receipts(self) -> tuple[KaiwuTaskReceipt, ...]:
        return tuple(self._receipts)

    @property
    def last_job(self) -> KaiwuRemoteJob | None:
        return self._last_job

    @property
    def last_result(self) -> KaiwuTaskResult | None:
        return self._last_result

    @property
    def last_precision_report(self) -> IntegerPrecisionReport | None:
        return self._last_precision_report

    @property
    def precision_reports(self) -> tuple[IntegerPrecisionReport, ...]:
        """Return one report for each distinct original matrix encountered."""

        return tuple(self._precision_reports.values())

    @property
    def precision_evidence(self) -> tuple[KaiwuPrecisionEvidence, ...]:
        """Return identity-bound precision evidence for each original matrix."""

        return tuple(self._precision_evidence.values())

    @property
    def transfer_records(self) -> tuple[KaiwuTransferRecord, ...]:
        """Return completed CPU-side matrix and sample boundary observations."""

        return tuple(self._transfer_records)

    def solve(self, ising_matrix: object) -> np.ndarray:
        """Serialize one bounded solve transaction and return ``int8`` spins.

        Serializing the complete transaction makes cache lookup, task
        registration, budget accounting, waiting, and evidence publication one
        atomic operation per sampler instance. Concurrent calls therefore
        cannot submit the same matrix twice or race past the declared budget.
        """

        with self._solve_lock:
            return self._solve_locked(ising_matrix)

    def _solve_locked(self, ising_matrix: object) -> np.ndarray:
        """Submit one unique matrix while ``_solve_lock`` is held."""

        try:
            import numpy as np
        except (
            ImportError
        ) as exc:  # pragma: no cover - plugin environments require NumPy
            raise KaiwuInteropError(
                "KaiwuSampler requires NumPy from the Kaiwu PyTorch Plugin environment"
            ) from exc

        canonical = canonicalize_ising_matrix(cast(MatrixLike, ising_matrix))
        input_device = str(getattr(ising_matrix, "device", "cpu"))
        input_dtype = str(getattr(ising_matrix, "dtype", type(ising_matrix).__name__))
        input_type = (
            f"{type(ising_matrix).__module__}.{type(ising_matrix).__qualname__}"
        )
        self._last_precision_report = None
        report: IntegerPrecisionReport | None = None
        if self._integer_target_range is not None:
            target_min, target_max = self._integer_target_range
            report = prepare_integer_precision(
                canonical,
                target_min=target_min,
                target_max=target_max,
            )
            submission = report.quantized
            self._last_precision_report = report
        else:
            submission = canonical

        original_key = hashlib.sha256(
            json.dumps(
                canonical.tolist(), separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        matrix_rows = tuple(tuple(float(value) for value in row) for row in submission)
        cache_key = hashlib.sha256(
            json.dumps(matrix_rows, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            if report is not None:
                self._precision_reports.setdefault(original_key, report)
                self._retain_precision_evidence(
                    original_key,
                    cache_key,
                    source_type=input_type,
                    source_dtype=input_dtype,
                    report=report,
                )
            output: np.ndarray = np.asarray(cached, dtype=np.int8).copy()
            self._record_transfer(
                input_type=input_type,
                input_device=input_device,
                input_dtype=input_dtype,
                canonical=canonical,
                original_matrix_sha256=original_key,
                submission_matrix_sha256=cache_key,
                output=output,
                cache_hit=True,
            )
            return output
        job = self._jobs.get(cache_key)
        if job is None:
            if self._remote_call_count >= self._max_remote_calls:
                raise RuntimeError(
                    "KaiwuSampler remote-call budget exhausted before submission"
                )

            # Reserve the slot before crossing the client boundary. A client
            # exception may be indeterminate: the provider could already have
            # observed the task identity, so releasing the slot would permit a
            # later matrix to exceed the declared quota ceiling.
            self._remote_call_count += 1
            job = submit_kaiwu_task(
                matrix_rows,
                client=self._client,
                task_name=f"{self._task_name}-{cache_key[:12]}",
                mode="sampling",
                requested_samples=self._requested_samples,
                project_no=self._project_no,
            )
            self._jobs[cache_key] = job
            self._receipts.append(job.receipt)
        if report is not None:
            self._precision_reports.setdefault(original_key, report)
            self._retain_precision_evidence(
                original_key,
                cache_key,
                source_type=input_type,
                source_dtype=input_dtype,
                report=report,
            )
        self._last_job = job
        result = job.wait(timeout=self._timeout, poll_interval=self._poll_interval)
        self._last_result = result
        self._cache[cache_key] = result.samples
        output = cast("np.ndarray", np.asarray(result.samples, dtype=np.int8).copy())
        self._record_transfer(
            input_type=input_type,
            input_device=input_device,
            input_dtype=input_dtype,
            canonical=canonical,
            original_matrix_sha256=original_key,
            submission_matrix_sha256=cache_key,
            output=output,
            cache_hit=False,
        )
        return output

    def _retain_precision_evidence(
        self,
        original_matrix_sha256: str,
        submission_matrix_sha256: str,
        *,
        source_type: str,
        source_dtype: str,
        report: IntegerPrecisionReport,
    ) -> None:
        self._precision_evidence.setdefault(
            original_matrix_sha256,
            KaiwuPrecisionEvidence(
                original_matrix_sha256=original_matrix_sha256,
                submission_matrix_sha256=submission_matrix_sha256,
                source_type=source_type,
                source_dtype=source_dtype,
                normalized_dtype=report.normalized_dtype,
                normalized_min=report.normalized_min,
                normalized_max=report.normalized_max,
                symmetry_normalization=report.symmetry_normalization,
                rounding_policy=report.rounding_policy,
                scale_factor=report.scale_factor,
                target_min=report.target_min,
                target_max=report.target_max,
                max_abs_error=report.max_abs_error,
                mean_abs_error=report.mean_abs_error,
            ),
        )

    def _record_transfer(
        self,
        *,
        input_type: str,
        input_device: str,
        input_dtype: str,
        canonical: torch.Tensor,
        original_matrix_sha256: str,
        submission_matrix_sha256: str,
        output: np.ndarray,
        cache_hit: bool,
    ) -> None:
        canonical_shape = cast(tuple[int, int], tuple(canonical.shape))
        output_shape = cast(tuple[int, int], tuple(output.shape))
        self._transfer_records.append(
            KaiwuTransferRecord(
                input_type=input_type,
                input_device=input_device,
                input_dtype=input_dtype,
                matrix_shape=canonical_shape,
                original_matrix_sha256=original_matrix_sha256,
                submission_matrix_sha256=submission_matrix_sha256,
                canonical_device=str(canonical.device),
                canonical_dtype=str(canonical.dtype),
                submission_storage="cpu_python_tuple",
                returned_storage="cpu_numpy",
                returned_dtype=str(output.dtype),
                returned_shape=output_shape,
                cache_hit=cache_hit,
            )
        )
