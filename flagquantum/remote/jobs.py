"""Detached remote jobs with explicit polling and credential-free receipts."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..errors import CapabilityError, ExecutionError
from ..runtime.result import ExecutionResult
from .compute._native_job import NativeJiudingJobClient
from .compute.jiuding import JiudingClient
from .qpu.contracts import (
    ProviderTaskHandle,
    build_result_metadata,
    validate_deployment_result,
)
from .qpu.quafu import QuafuProvider

if TYPE_CHECKING:
    from ..circuit import Circuit
    from ..core.ir import CircuitIR
    from ..observables import OutputRequest

JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled", "unknown"]


@dataclass(frozen=True)
class _Receipt:
    id: str
    target: str
    n_wires: int
    compiler: str | None = None
    target_qubits: tuple[int, ...] = ()
    name: str | None = None
    output_name: str | None = None
    workspace: str | None = None
    submission_identity: dict[str, str] = field(default_factory=dict)
    native_receipt: dict[str, Any] | None = None
    schema: str = "flagquantum.remote-job.v1"


def _normalize(raw: str) -> JobStatus:
    value = raw.lower()
    if value in {"finished", "completed", "done", "succeed", "succeeded"}:
        return "succeeded"
    if value in {"failed", "error"}:
        return "failed"
    if value in {"cancelled", "canceled", "stopped"}:
        return "cancelled"
    if value in {
        "queued",
        "pending",
        "scheduling",
        "starting",
        "transpiled",
        "waiting",
    }:
        return "queued"
    if value in {"running", "executing", "transpiling"}:
        return "running"
    return "unknown"


class RemoteJob:
    """One remotely submitted job; obtain through submit or restore_job.

    No method resubmits a job. Status and result calls may wait for network I/O,
    but only wait() polls for execution completion. raw_status holds the last
    queried provider status, or None before the first query.
    """

    def __init__(
        self,
        receipt: _Receipt,
        client: QuafuProvider | JiudingClient,
        native_receipt: dict[str, Any] | None = None,
    ) -> None:
        self._receipt = receipt
        self._client = client
        self._native_receipt = native_receipt
        self._raw_status: str | None = None

    @property
    def id(self) -> str:
        return self._receipt.id

    @property
    def target(self) -> str:
        return self._receipt.target

    @property
    def raw_status(self) -> str | None:
        return self._raw_status

    def _handle(self) -> ProviderTaskHandle:
        return ProviderTaskHandle(
            "quafu",
            self.id,
            self.target.partition(":")[2],
            self._receipt.submission_identity,
        )

    def _jiuding_receipt(self) -> dict[str, Any]:
        assert isinstance(self._client, JiudingClient)
        if self._native_receipt is None:
            self._native_receipt = self._client.restore_receipt(self.id)
        return self._native_receipt

    def status(self) -> JobStatus:
        """Query once and normalize state; unrecognized states remain unknown."""
        if isinstance(self._client, QuafuProvider):
            self._raw_status = self._client.query_status(self._handle())
            return _normalize(self._raw_status)
        jobs = self._client.status(self._jiuding_receipt())
        # An experiment may have previous attempts: inspect the submitted job only.
        states = [str(j.get("status", "")) for j in jobs if str(j.get("id")) == self.id]
        self._raw_status = ", ".join(states) if states else "Missing"
        return _normalize(states[0]) if len(states) == 1 else "unknown"

    def result(self) -> ExecutionResult:
        """Fetch a finished result once; raise ExecutionError when not successful."""
        state = self.status()
        if state != "succeeded":
            raise ExecutionError(
                f"Job {self.id} is {state} ({self.raw_status}); no result fetched"
            )
        if isinstance(self._client, QuafuProvider):
            from .qpu.execution import counts_result

            native = validate_deployment_result(
                self._client.fetch_result(self._handle())
            )
            r = self._receipt
            if any(len(str(key)) != r.n_wires for key in native.counts):
                raise ExecutionError(
                    "Returned count width differs from the submitted circuit"
                )
            return counts_result(
                native,
                n_wires=r.n_wires,
                output_name=r.output_name,
                compiler=r.compiler,
                target=r.target,
                target_qubits=r.target_qubits,
                name=r.name,
            )
        if isinstance(self._client, NativeJiudingJobClient):
            return self._client.read_result(self._jiuding_receipt())

        from .compute._managed_program import read_managed_result
        from .compute._program_submission import decode_job_result

        receipt = self._jiuding_receipt()
        value = decode_job_result(read_managed_result(self._client, receipt), receipt)
        if not isinstance(value, ExecutionResult):
            raise ExecutionError("Jiuding program did not return an ExecutionResult")
        return value

    def wait(
        self, timeout: float = 300.0, poll_interval: float = 3.0
    ) -> ExecutionResult:
        """Poll until completion; timeout leaves the remote job running.

        The deadline bounds polling; an in-flight network request is additionally
        bounded by its provider timeout. No background thread is started.
        """
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")
        deadline = time.monotonic() + timeout
        while True:
            state = self.status()
            if state == "succeeded":
                try:
                    return self.result()
                except RuntimeError as exc:
                    # Providers may report completion before results become readable.
                    if "has no result yet" not in str(exc):
                        raise
            if state in {"failed", "cancelled"}:
                raise ExecutionError(f"Job {self.id} is {state} ({self.raw_status})")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Job {self.id} is not complete; no cancellation or resubmission"
                )
            time.sleep(min(poll_interval, remaining))

    def cancel(self) -> None:
        """Request cancellation; query status to confirm the remote outcome."""
        if isinstance(self._client, QuafuProvider):
            self._client.cancel(self._handle())
        else:
            receipt = self._jiuding_receipt()
            attempts = self._client.status(receipt)
            if any(
                str(attempt.get("id")) != self.id
                and _normalize(str(attempt.get("status", "")))
                not in {"succeeded", "failed", "cancelled"}
                for attempt in attempts
            ):
                raise CapabilityError(
                    "Jiuding experiment has other active attempts; cancel through the provider"
                )
            self._client.cancel(receipt)

    def save(self, path: str | Path) -> None:
        """Save a private JSON receipt without credentials; never overwrite a file."""
        encoded = json.dumps(asdict(self._receipt), indent=2) + "\n"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)


def submit(
    program: Circuit | CircuitIR,
    *,
    target: str,
    outputs: OutputRequest | Sequence[OutputRequest] | None = None,
    shots: int | None = None,
    compiler: str | None = None,
    target_qubits: Sequence[int] | None = None,
    name: str | None = None,
    image: str | None = None,
    workspace: str | None = None,
    project: str | None = None,
    queue: str | None = None,
) -> RemoteJob:
    """Submit once and return without waiting for remote execution.

    Quafu supports full-register counts. Jiuding uses native HTTP batch execution and
    requires an image; credentials come from the existing provider configuration.
    Submission includes preparation and network I/O. Unsupported output requests
    fail before a task is submitted. Save the returned job for later restoration.
    """
    from ..core.ir import ensure_circuit_ir

    ir = ensure_circuit_ir(program)
    provider, separator, backend = target.partition(":")
    if not separator or not backend.strip() or provider not in {"quafu", "jiuding"}:
        raise ValueError("target must name a quafu or jiuding backend")
    if provider == "jiuding":
        if compiler is not None or target_qubits is not None or name is not None:
            raise TypeError(
                "Jiuding submission does not accept compiler, target_qubits or name"
            )
        if not isinstance(image, str) or not image.strip():
            raise ValueError("Jiuding batch submission requires image")
        if workspace is not None:
            raise TypeError("Native jobs use project and queue, not workspace")
        selected_project = project or os.environ.get("JIUDING_PROJECT")
        if not selected_project:
            raise ValueError(
                "Jiuding submission requires project='project-set.project'"
            )
        client = NativeJiudingJobClient(project=selected_project, queue=queue)
        native = client.submit_program(
            ir, target=target, image=image, outputs=outputs, shots=shots
        )
        return RemoteJob(
            _Receipt(str(native["jobId"]), target, ir.n_wires, native_receipt=native),
            client,
            native,
        )
    if any(value is not None for value in (image, workspace, project, queue)):
        raise TypeError(
            "image, project, queue and workspace apply only to Jiuding submission"
        )
    return _submit_quafu(
        ir,
        target=target,
        outputs=outputs,
        shots=shots,
        compiler=compiler,
        target_qubits=target_qubits,
        name=name,
    )


def _submit_quafu(
    ir: CircuitIR,
    *,
    target: str,
    outputs: OutputRequest | Sequence[OutputRequest] | None,
    shots: int | None,
    compiler: str | None,
    target_qubits: Sequence[int] | None,
    name: str | None,
) -> RemoteJob:
    from ..deployment import CloudBackendProfile, create_deployment_package
    from .qpu.execution import validate_quafu_output
    from .qpu.quafu import _service_options

    output = validate_quafu_output(ir, outputs)
    if output.kind != "counts":
        raise CapabilityError(
            "Detached Quafu jobs support full-register counts; use fq.run for expectations"
        )
    if type(shots) is not int or shots <= 0 or shots % 1024:
        raise ValueError("Quafu shots must be a positive multiple of 1024")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        raise ValueError("name must be a non-empty string")
    package_options: dict[str, Any] = {}
    if compiler is None:
        if ir.metadata.get("execution_target"):
            raise ValueError("service compilation requires an unbound logical circuit")
        options = _service_options(
            ir.n_wires, () if target_qubits is None else target_qubits
        )
        package_options = {
            "backend": CloudBackendProfile(
                provider="quafu",
                name=target.partition(":")[2].strip(),
                n_wires=ir.n_wires,
            ),
            "optimize": False,
            "metadata": {
                "provider_options": options,
                "compilation_location": "service",
                "service_compiler": "quarkcircuit",
            },
        }
        mapping = tuple(options["target_qubits"])
    else:
        from .._api import compile

        ir = compile(ir, compiler=compiler, target=target, target_qubits=target_qubits)
        mapping = tuple(
            ir.metadata.get("execution_target", {}).get("target_qubits", ())
        )
    package = create_deployment_package(
        ir,
        shots=shots,
        name=name.strip() if name else "flagquantum_job",
        **package_options,
    )
    client = QuafuProvider(reverse_result_bits=compiler is None)
    handle = client.submit(package)
    return RemoteJob(
        _Receipt(
            handle.task_id,
            target,
            ir.n_wires,
            compiler,
            mapping,
            name,
            output.name,
            submission_identity=build_result_metadata(handle),
        ),
        client,
    )


def restore_job(path: str | Path) -> RemoteJob:
    """Reconnect using a saved receipt; never submit or deserialize executable code."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != "flagquantum.remote-job.v1":
        raise ValueError("Unsupported remote job receipt schema")
    try:
        receipt = _Receipt(**raw)
    except TypeError as exc:
        raise ValueError("Invalid remote job receipt fields") from exc
    if not isinstance(receipt.id, str) or not receipt.id.strip():
        raise ValueError("Receipt must contain a job ID")
    if type(receipt.n_wires) is not int or receipt.n_wires <= 0:
        raise ValueError("Receipt must contain a positive circuit width")
    if not isinstance(receipt.target, str):
        raise ValueError("Invalid receipt target")
    for value in (
        receipt.compiler,
        receipt.name,
        receipt.output_name,
        receipt.workspace,
    ):
        if value is not None and not isinstance(value, str):
            raise ValueError("Invalid receipt text field")
    from .qpu.quafu import _service_options

    _service_options(receipt.n_wires, receipt.target_qubits)
    provider, separator, backend = receipt.target.partition(":")
    if not separator or not backend.strip():
        raise ValueError("Invalid receipt target")
    if provider == "quafu":
        if not isinstance(receipt.submission_identity, dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in receipt.submission_identity.items()
        ):
            raise ValueError("Invalid submission identity")
        build_result_metadata(
            ProviderTaskHandle(
                "quafu", receipt.id, backend, receipt.submission_identity
            )
        )
        return RemoteJob(
            receipt, QuafuProvider(reverse_result_bits=receipt.compiler is None)
        )
    if provider == "jiuding":
        native = receipt.native_receipt
        if native is not None:
            if (
                not isinstance(native, dict)
                or native.get("jobId") != receipt.id
                or native.get("artifact_transport") != "job_logs"
                or not isinstance(native.get("project"), str)
                or not isinstance(native.get("queue"), str)
            ):
                raise ValueError("Invalid native job receipt")
            return RemoteJob(
                receipt,
                NativeJiudingJobClient(
                    project=native["project"], queue=native["queue"]
                ),
                native,
            )
        return RemoteJob(receipt, JiudingClient(workspace=receipt.workspace))
    raise ValueError("Unsupported receipt provider")


__all__ = ("JobStatus", "RemoteJob", "restore_job", "submit")
