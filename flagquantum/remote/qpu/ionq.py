"""IonQ trapped-ion provider: discovery, preflight, submission, and results.

IonQ chooses a gate set per job rather than per device, so the adapter takes the
gate set from its caller and reads the device width and simulator marking from
the vendor's own listing. A device that declares no gate names leaves the
profile's `basis_gates` empty instead of filling it from a hard-coded list, and
the compiler then requires the caller to legalize the program against the gate
set the job actually names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...compiler import CouplingMap
from ...deployment.cloud import CloudBackendProfile, DeploymentPackage
from .contracts import (
    DeploymentResult,
    ProviderSubmissionPreview,
    ProviderTaskHandle,
    QuantumProvider,
    build_result_metadata,
    build_submission_receipt,
    preflight_submission,
)
from .declaration import (
    connectivity_edges,
    declared_count,
    declared_flag,
    declared_gate_names,
    declared_text,
    declared_value,
)
from .http import ProviderCredentials, QuantumCloudTransport, UrllibTransport
from .result_parsing import _outcome_counts

IONQ_API_BASE = "https://api.ionq.co/v0.3"
IONQ_PROVIDER = "ionq"

#: The two gate sets the IonQ API accepts for a job. ``qis`` is the vendor's
#: named gate set; ``native`` names the device's own physical gates.
IONQ_GATE_SETS = ("qis", "native")

_IONQ_STATUS = {
    "completed": "Finished",
    "canceled": "Cancelled",
    "cancelled": "Cancelled",
    "failed": "Failed",
    "ready": "Running",
    "submitted": "Running",
    "running": "Running",
    "waiting": "Running",
    "paused": "Paused",
}

_CONNECTIVITY_NAMES = (
    "connectivity",
    "connectivity_graph",
    "connectivityGraph",
    "coupling_map",
    "couplingMap",
)

#: The listing names the simulation targets directly rather than flagging them,
#: so a name match is the only signal a listing carries.
_SIMULATOR_MARKERS = ("simulator", "sim.", "emulator")


def ionq_backend_profile(
    declaration: Any,
    *,
    gateset: str = "qis",
    basis_gates: Sequence[str] = (),
) -> CloudBackendProfile:
    """Build a fail-closed backend profile from one IonQ listing row.

    ``gateset`` is the gate set the caller will submit with, because IonQ
    selects it per job. ``basis_gates`` is the caller's authoritative gate
    declaration and overrides whatever the listing carries.
    """

    owner = "IonQ"
    if gateset not in IONQ_GATE_SETS:
        raise ValueError(
            f"gateset must be one of {', '.join(IONQ_GATE_SETS)}, got {gateset!r}"
        )
    name = declared_text(
        declaration, "backend", "name", "id", owner=owner, fact="backend name"
    )
    n_qubits = declared_count(
        declaration, "qubits", "n_qubits", "nqubits", owner=owner, fact="qubit count"
    )
    declared_gates = declared_gate_names(
        declared_value(
            declaration, "gates", "gateset_gates", "basis_gates", default=None
        ),
        owner=owner,
    )
    raw_connectivity = declared_value(declaration, *_CONNECTIVITY_NAMES, default=None)
    coupling_map = None
    if raw_connectivity is not None:
        coupling_map = CouplingMap(
            n_wires=n_qubits,
            edges=connectivity_edges(raw_connectivity, owner=owner, n_qubits=n_qubits),
        )
    status = declared_value(declaration, "status", "state", default=None)
    lowered = name.lower()
    declared_simulator = declared_value(
        declaration, "is_simulator", "isSimulator", default=None
    )
    if declared_simulator is None:
        is_simulator = any(marker in lowered for marker in _SIMULATOR_MARKERS)
        simulator_source = "backend_name"
    else:
        is_simulator = declared_flag(
            declaration,
            "is_simulator",
            "isSimulator",
            owner=owner,
            fact="simulator marker",
            default=False,
        )
        simulator_source = "declared"
    return CloudBackendProfile(
        provider=IONQ_PROVIDER,
        name=name,
        n_qubits=n_qubits,
        basis_gates=(
            declared_gate_names(basis_gates, owner=owner)
            if basis_gates
            else declared_gates
        ),
        coupling_map=coupling_map,
        supports_openqasm=True,
        supports_dynamic_circuits=False,
        is_simulator=is_simulator,
        metadata={
            "gateset": gateset,
            "capability_source": "IonQ backend listing",
            "declared_status": None if status is None else str(status),
            "simulator_source": simulator_source,
            "connectivity_declared": coupling_map is not None,
        },
    )


def _program_format(qasm_version: float) -> str:
    if qasm_version == 3.0:
        return "qasm3"
    if qasm_version == 2.0:
        return "qasm2"
    raise ValueError(f"IonQ submission does not accept QASM {qasm_version:g}")


class IonQProvider(QuantumProvider):
    """Submit FlagQuantum OpenQASM packages to one IonQ gate set.

    Parameters:
        base_url: API root, overridable for a compatible control plane.
        credentials: API key material; ``token`` becomes the IonQ API key.
        gateset: One of :data:`IONQ_GATE_SETS`.
        transport: HTTP transport, replaced in tests.
        shots_limit: Refuse a job above the account's declared shot budget.

    Raises:
        ValueError: A declared listing row, gate set, or QASM version is
            unusable.
        RuntimeError: A submission would not be accepted as written, or a
            result payload carries no outcomes.
    """

    provider = IONQ_PROVIDER

    def __init__(
        self,
        *,
        base_url: str = IONQ_API_BASE,
        credentials: ProviderCredentials | None = None,
        gateset: str = "qis",
        basis_gates: Sequence[str] = (),
        transport: QuantumCloudTransport | None = None,
        timeout: float = 30.0,
        shots_limit: int | None = None,
    ) -> None:
        if gateset not in IONQ_GATE_SETS:
            raise ValueError(
                f"gateset must be one of {', '.join(IONQ_GATE_SETS)}, got {gateset!r}"
            )
        self.base_url = base_url.rstrip("/")
        self.credentials = credentials or ProviderCredentials()
        self.gateset = gateset
        self.basis_gates = tuple(basis_gates)
        self.transport = transport or UrllibTransport()
        self.timeout = float(timeout)
        self.shots_limit = shots_limit

    def _url(self, path: str, **values: Any) -> str:
        return self.base_url + path.format(**values)

    def _headers(self) -> dict[str, str]:
        headers = dict(self.credentials.extra_headers)
        if self.credentials.token:
            headers.setdefault("Authorization", f"apiKey {self.credentials.token}")
        return headers

    def list_devices(
        self, n_qubits: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        payload = self.transport.get_json(
            self._url("/backends"), self._headers(), self.timeout
        )
        rows = declared_value(payload, "backends", "data", default=payload)
        if isinstance(rows, Mapping):
            raise ValueError("IonQ backend listing is a mapping, not a list of rows")
        if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
            raise ValueError("IonQ backend listing does not contain rows")
        profiles = []
        for row in rows:
            profile = ionq_backend_profile(
                row, gateset=self.gateset, basis_gates=self.basis_gates
            )
            if n_qubits is not None and profile.n_qubits < n_qubits:
                continue
            profiles.append(profile)
        return tuple(profiles)

    def dry_run(self, package: DeploymentPackage) -> ProviderSubmissionPreview:
        """Validate locally and expose the program without creating a job."""

        preview = preflight_submission(
            package,
            provider=self.provider,
            qasm_versions=(2.0, 3.0),
            dynamic_circuits_supported=False,
        )
        blockers = list(preview.blockers)
        if self.shots_limit is not None and package.shots > self.shots_limit:
            blockers.append("shots_exceed_the_declared_ionq_shot_budget")
        return ProviderSubmissionPreview(
            provider=preview.provider,
            backend=preview.backend,
            shots=preview.shots,
            program_format=preview.program_format,
            program=preview.program,
            compatible=not blockers,
            blockers=tuple(blockers),
        )

    def _submit_payload(self, package: DeploymentPackage) -> dict[str, Any]:
        return {
            "target": package.backend.name,
            "name": package.name,
            "shots": package.shots,
            "gateset": self.gateset,
            "input": {
                "format": _program_format(package.qasm_version),
                "value": package.qasm,
            },
        }

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        preview = self.dry_run(package)
        if not preview.compatible:
            raise RuntimeError("IonQ preflight failed: " + ", ".join(preview.blockers))
        response = self.transport.post_json(
            self._url("/jobs"),
            self._submit_payload(package),
            self._headers(),
            self.timeout,
        )
        task_id = declared_value(response, "id", "job_id", "task_id", default=None)
        if task_id is None:
            raise RuntimeError("IonQ submit response does not contain a job id.")
        return ProviderTaskHandle(
            provider=self.provider,
            task_id=str(task_id),
            backend_name=package.backend.name,
            payload=build_submission_receipt(
                package, {"shots": package.shots, **dict(response)}
            ),
        )

    def query_status(self, handle: ProviderTaskHandle) -> str:
        response = self.transport.get_json(
            self._url("/jobs/{task_id}", task_id=handle.task_id),
            self._headers(),
            self.timeout,
        )
        status = declared_value(response, "status", "state", default=None)
        if status is None:
            raise RuntimeError("IonQ job response does not contain a status.")
        lowered = str(status).lower()
        return _IONQ_STATUS.get(lowered, str(status))

    def cancel(self, handle: ProviderTaskHandle) -> Any:
        """Ask IonQ to cancel the job and report the resulting status."""

        response = self.transport.post_json(
            self._url("/jobs/{task_id}/cancel", task_id=handle.task_id),
            {},
            self._headers(),
            self.timeout,
        )
        return declared_value(response, "status", "state", default="Cancelled")

    def fetch_result(self, handle: ProviderTaskHandle) -> DeploymentResult:
        response = self.transport.get_json(
            self._url("/jobs/{task_id}/results", task_id=handle.task_id),
            self._headers(),
            self.timeout,
        )
        shots = int(handle.payload.get("shots", 0))
        if shots <= 0:
            raise RuntimeError("IonQ result cannot be read without its shot count.")
        counts = _outcome_counts(
            dict(response),
            provider="IonQ",
            keys=("probabilities", "counts", "histogram"),
            shots=shots,
            outcome_key=_ionq_outcome,
        )
        return DeploymentResult(
            handle=handle,
            counts=counts,
            shots=sum(counts.values()),
            metadata=build_result_metadata(
                handle, {"gateset": self.gateset, "result_format": "probabilities"}
            ),
        )


def _ionq_outcome(outcome: Any) -> str:
    bits = str(outcome).strip()
    if not bits or any(bit not in "01" for bit in bits):
        raise RuntimeError(f"IonQ result contains invalid outcome {outcome!r}")
    return bits


__all__ = (
    "IONQ_API_BASE",
    "IONQ_GATE_SETS",
    "IonQProvider",
    "ionq_backend_profile",
)
