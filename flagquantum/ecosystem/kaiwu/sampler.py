"""Bounded synchronous sampler compatibility for Kaiwu PyTorch Plugin."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
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
    canonical_device: str
    canonical_dtype: str
    submission_storage: str
    returned_storage: str
    returned_dtype: str
    returned_shape: tuple[int, int]
    cache_hit: bool


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
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError("task_name must be a non-empty string")
        if type(max_remote_calls) is not int or max_remote_calls <= 0:
            raise ValueError("max_remote_calls must be a positive integer")
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and non-negative")
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")
        if type(requested_samples) is not int or not 10 <= requested_samples <= 2000:
            raise ValueError("requested_samples must be between 10 and 2000")
        if integer_target_range is not None and (
            not isinstance(integer_target_range, tuple)
            or len(integer_target_range) != 2
        ):
            raise ValueError("integer_target_range must be a two-integer tuple or None")

        self._client = client
        self._task_name = task_name.strip()
        self._project_no = project_no
        self._requested_samples = requested_samples
        self._timeout = float(timeout)
        self._poll_interval = float(poll_interval)
        self._max_remote_calls = max_remote_calls
        self._integer_target_range = integer_target_range
        self._cache: dict[str, tuple[tuple[int, ...], ...]] = {}
        self._jobs: dict[str, KaiwuRemoteJob] = {}
        self._receipts: list[KaiwuTaskReceipt] = []
        self._remote_call_count = 0
        self._last_job: KaiwuRemoteJob | None = None
        self._last_result: KaiwuTaskResult | None = None
        self._last_precision_report: IntegerPrecisionReport | None = None
        self._precision_reports: dict[str, IntegerPrecisionReport] = {}
        self._transfer_records: list[KaiwuTransferRecord] = []

    @property
    def remote_call_count(self) -> int:
        return self._remote_call_count

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
    def transfer_records(self) -> tuple[KaiwuTransferRecord, ...]:
        """Return completed CPU-side matrix and sample boundary observations."""

        return tuple(self._transfer_records)

    def solve(self, ising_matrix: object) -> np.ndarray:
        """Submit one unique matrix and return an ``int8`` NumPy spin array."""

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
            output: np.ndarray = np.asarray(cached, dtype=np.int8).copy()
            self._record_transfer(
                input_type=input_type,
                input_device=input_device,
                input_dtype=input_dtype,
                canonical=canonical,
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

            job = submit_kaiwu_task(
                matrix_rows,
                client=self._client,
                task_name=f"{self._task_name}-{cache_key[:12]}",
                mode="sampling",
                requested_samples=self._requested_samples,
                project_no=self._project_no,
            )
            self._remote_call_count += 1
            self._jobs[cache_key] = job
            self._receipts.append(job.receipt)
        if report is not None:
            self._precision_reports.setdefault(original_key, report)
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
            output=output,
            cache_hit=False,
        )
        return output

    def _record_transfer(
        self,
        *,
        input_type: str,
        input_device: str,
        input_dtype: str,
        canonical: torch.Tensor,
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
                canonical_device=str(canonical.device),
                canonical_dtype=str(canonical.dtype),
                submission_storage="cpu_python_tuple",
                returned_storage="cpu_numpy",
                returned_dtype=str(output.dtype),
                returned_shape=output_shape,
                cache_hit=cache_hit,
            )
        )
