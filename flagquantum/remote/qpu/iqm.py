"""IQM superconducting provider: discovery, preflight, submission, and results.

IQM devices are superconducting lattices, so unlike a trapped-ion machine the
coupling map is a real constraint and the vendor publishes it. This adapter
reads that lattice and passes it through as a Compiler ``CouplingMap``, so a
program that needs routing is routed against the declared device rather than
against an assumed one. A device row that declares no connectivity therefore
produces a profile with no coupling map: the absence is reported, and the
Compiler then has no lattice to route against.

IQM results are reported as a set of bitstrings, and the bitstring order is the
vendor's; this adapter does not reverse it.
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
    declared_flag,
    declared_gate_names,
    declared_text,
    declared_value,
    declared_width,
)
from .http import ProviderCredentials, QuantumCloudTransport, UrllibTransport
from .result_parsing import _outcome_counts

IQM_API_BASE = "https://resonance.iqm.tech/v1"
IQM_PROVIDER = "iqm"

#: IQM's own physical gate names. A caller may override the setting, but a
#: device row that names its gates is the authority this adapter returns.
IQM_NATIVE_GATES = ("prx", "cz")

_IQM_STATUS = {
    "completed": "Finished",
    "failed": "Failed",
    "aborted": "Failed",
    "cancelled": "Cancelled",
    "canceled": "Cancelled",
    "pending": "Running",
    "queued": "Running",
    "ready": "Running",
    "running": "Running",
    "executing": "Running",
}

_CONNECTIVITY_NAMES = (
    "connectivity",
    "coupling_map",
    "couplingMap",
    "connections",
)
_DEVICE_TYPE_NAMES = ("type", "device_type", "category")
_SIMULATOR_TYPES = ("simulator", "emulator", "sim")


def iqm_backend_profile(
    declaration: Any,
    *,
    basis_gates: Sequence[str] = (),
) -> CloudBackendProfile:
    """Build a fail-closed backend profile from one IQM device declaration.

    The connectivity the device declares becomes the profile's coupling map. A
    device whose lattice is absent from the declaration is reported as having no
    declared connectivity rather than being given a default topology, and the
    profile's own ``coupling_map`` is then ``None``.

    An IQM listing states its width as ``qubits``, the list of physical qubit
    positions, so the width is the length of that list. A listing that states a
    count instead is read the same way, and which one answered is recorded in
    the profile metadata.
    """

    owner = "IQM"
    name = declared_text(
        declaration, "name", "device_name", "id", owner=owner, fact="device name"
    )
    n_qubits, width_source = declared_width(
        declaration, "qubits", "n_qubits", "nqubits", owner=owner
    )
    declared_gates = declared_gate_names(
        declared_value(
            declaration, "gates", "basis_gates", "native_gates", default=None
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
    declared_type = declared_value(declaration, *_DEVICE_TYPE_NAMES, default=None)
    declared_simulator = declared_value(
        declaration, "is_simulator", "simulator", default=None
    )
    if declared_simulator is not None:
        simulator = declared_flag(
            declaration,
            "is_simulator",
            "simulator",
            owner=owner,
            fact="simulator marker",
            default=False,
        )
        simulator_source = "declared"
    elif declared_type is not None:
        simulator = str(declared_type).strip().lower() in _SIMULATOR_TYPES
        simulator_source = "device_type"
    else:
        simulator = "simulator" in name.lower() or "emulator" in name.lower()
        simulator_source = "device_name"
    return CloudBackendProfile(
        provider=IQM_PROVIDER,
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
        is_simulator=simulator,
        metadata={
            "capability_source": "IQM device listing",
            "declared_type": None if declared_type is None else str(declared_type),
            "simulator_source": simulator_source,
            "qubit_count_source": width_source,
            "connectivity_declared": coupling_map is not None,
            "declared_coupling_count": (
                0 if coupling_map is None else len(coupling_map.edges)
            ),
        },
    )


class IQMProvider(QuantumProvider):
    """Submit FlagQuantum OpenQASM 3 packages to one IQM device.

    Parameters:
        base_url: API root, overridable for a compatible control plane.
        credentials: Bearer token material.
        device: Device to target, or ``None`` for submission-independent use.
        basis_gates: The device's native gate set, as declared by the caller.
        transport: HTTP transport, replaced in tests.

    Raises:
        ValueError: A device row, gate declaration, or QASM version is unusable.
        RuntimeError: A submission would not be accepted as written, or a result
            payload carries no outcomes.
    """

    provider = IQM_PROVIDER

    def __init__(
        self,
        *,
        base_url: str = IQM_API_BASE,
        credentials: ProviderCredentials | None = None,
        device: str | None = None,
        basis_gates: Sequence[str] = (),
        transport: QuantumCloudTransport | None = None,
        timeout: float = 30.0,
        shots_limit: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.credentials = credentials or ProviderCredentials()
        self.device = device
        self.basis_gates = tuple(basis_gates)
        self.transport = transport or UrllibTransport()
        self.timeout = float(timeout)
        self.shots_limit = shots_limit

    def _url(self, path: str, **values: Any) -> str:
        return self.base_url + path.format(**values)

    def _headers(self) -> dict[str, str]:
        headers = dict(self.credentials.extra_headers)
        if self.credentials.token:
            headers.setdefault("Authorization", f"Bearer {self.credentials.token}")
        return headers

    def list_devices(
        self, n_qubits: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        payload = self.transport.get_json(
            self._url("/devices"), self._headers(), self.timeout
        )
        rows = declared_value(payload, "devices", "data", default=payload)
        if isinstance(rows, Mapping):
            raise ValueError("IQM device listing is a mapping, not a list of rows")
        if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
            raise ValueError("IQM device listing does not contain rows")
        profiles = []
        for row in rows:
            profile = iqm_backend_profile(row, basis_gates=self.basis_gates)
            if n_qubits is not None and profile.n_qubits < n_qubits:
                continue
            profiles.append(profile)
        return tuple(profiles)

    def dry_run(self, package: DeploymentPackage) -> ProviderSubmissionPreview:
        """Validate locally and expose the program without creating a job."""

        preview = preflight_submission(
            package,
            provider=self.provider,
            backend_name=self.device,
            qasm_versions=(3.0,),
            dynamic_circuits_supported=False,
        )
        blockers = list(preview.blockers)
        if self.shots_limit is not None and package.shots > self.shots_limit:
            blockers.append("shots_exceed_the_declared_iqm_shot_budget")
        return ProviderSubmissionPreview(
            provider=preview.provider,
            backend=preview.backend,
            shots=preview.shots,
            program_format=preview.program_format,
            program=preview.program,
            compatible=not blockers,
            blockers=tuple(blockers),
        )

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        preview = self.dry_run(package)
        if not preview.compatible:
            raise RuntimeError("IQM preflight failed: " + ", ".join(preview.blockers))
        response = self.transport.post_json(
            self._url("/jobs"),
            {
                "name": package.name,
                "device": package.backend.name,
                "shots": package.shots,
                "qasm": package.qasm,
            },
            self._headers(),
            self.timeout,
        )
        task_id = declared_value(response, "id", "job_id", "task_id", default=None)
        if task_id is None:
            raise RuntimeError("IQM submit response does not contain a job id.")
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
            raise RuntimeError("IQM job response does not contain a status.")
        lowered = str(status).lower()
        return _IQM_STATUS.get(lowered, str(status))

    def cancel(self, handle: ProviderTaskHandle) -> Any:
        """Ask IQM to cancel the job and report the resulting status."""

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
            raise RuntimeError("IQM result cannot be read without its shot count.")
        counts = _outcome_counts(
            dict(response),
            provider="IQM",
            keys=("counts", "measurements", "histogram", "probabilities"),
            shots=shots,
            outcome_key=_iqm_outcome,
        )
        return DeploymentResult(
            handle=handle,
            counts=counts,
            shots=sum(counts.values()),
            metadata=build_result_metadata(handle, {"bitstring_order": "as_returned"}),
        )


def _iqm_outcome(outcome: Any) -> str:
    if isinstance(outcome, str | bytes):
        bits = str(outcome).strip()
    elif isinstance(outcome, Sequence):
        if not all(bit in (0, 1, "0", "1") for bit in outcome):
            raise RuntimeError(f"IQM result contains invalid outcome {outcome!r}")
        bits = "".join(str(bit) for bit in outcome)
    else:
        raise RuntimeError(f"IQM result contains invalid outcome {outcome!r}")
    if not bits or any(bit not in "01" for bit in bits):
        raise RuntimeError(f"IQM result contains invalid outcome {outcome!r}")
    return bits


__all__ = (
    "IQM_API_BASE",
    "IQM_NATIVE_GATES",
    "IQMProvider",
    "iqm_backend_profile",
)
