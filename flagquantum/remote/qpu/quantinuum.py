"""Quantinuum H-series provider: discovery, preflight, submission, and results.

The Quantinuum cloud API is a QASM/QIR job service rather than an OpenQASM 3
one: a job declares its ``language``, and the adapter maps a package's QASM
version onto the language it can submit. The adapter maps only the versions it
has a language for and refuses the rest, so an unsupported version fails before
a job exists rather than at the vendor's door.

H-series machines share one trap, so the vendor lists no connectivity and this
adapter declares none. A caller that needs routing for a Quantinuum target
supplies the coupling map it was given, rather than having one invented here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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
    declared_count,
    declared_flag,
    declared_gate_names,
    declared_text,
    declared_value,
)
from .http import ProviderCredentials, QuantumCloudTransport, UrllibTransport
from .result_parsing import _outcome_counts, _tally_outcomes

QUANTINUUM_API_BASE = "https://api.quantinuum.com/v1"
QUANTINUUM_PROVIDER = "quantinuum"

#: The submission languages this adapter maps a package onto. A version absent
#: from the mapping is refused rather than translated.
QUANTINUUM_LANGUAGES: Mapping[float, str] = {2.0: "QASM"}

_QUANTINUUM_STATUS = {
    "completed": "Finished",
    "failed": "Failed",
    "error": "Failed",
    "cancelled": "Cancelled",
    "canceled": "Cancelled",
    "queued": "Running",
    "running": "Running",
    "submitted": "Running",
    "pending": "Running",
}

_SIMULATOR_MARKERS = ("sim", "emulator")


def quantinuum_backend_profile(
    declaration: Any,
    *,
    n_qubits: int | None = None,
    basis_gates: Sequence[str] = (),
    is_simulator: bool | None = None,
) -> CloudBackendProfile:
    """Build a fail-closed backend profile from one Quantinuum machine row.

    An API version that lists machine names only is accepted as a bare string;
    ``n_qubits`` then supplies the width, which such a listing does not carry. A
    row that declares its own width, or a caller that passes ``is_simulator``,
    wins over the inference below.
    """

    owner = "Quantinuum"
    row = {"name": declaration} if isinstance(declaration, str) else declaration
    name = declared_text(row, "name", "machine", "id", owner=owner, fact="machine name")
    width = declared_value(row, "n_qubits", "nqubits", "qubits", default=None)
    if width is None:
        if n_qubits is None:
            raise ValueError(
                "Quantinuum declaration does not expose a qubit count; pass "
                "n_qubits for an API version that lists machine names only"
            )
        row = {**dict(row), "n_qubits": n_qubits}
        width_source = "caller"
    else:
        width_source = "declaration"
    gate_names = declared_gate_names(
        declared_value(row, "gates", "basis_gates", default=None), owner=owner
    )
    # Quantinuum publishes the emulator of a machine under its own name with a
    # trailing ``E``, and marks no listing field for it.
    declared_simulator = declared_value(row, "is_simulator", default=None)
    if is_simulator is not None:
        if type(is_simulator) is not bool:
            raise ValueError("Quantinuum is_simulator must be a boolean")
        simulator = is_simulator
        simulator_source = "caller"
    elif declared_simulator is None:
        lowered = name.lower()
        simulator = lowered.endswith("e") or any(
            marker in lowered for marker in _SIMULATOR_MARKERS
        )
        simulator_source = "machine_name"
    else:
        simulator = declared_flag(
            row,
            "is_simulator",
            owner=owner,
            fact="simulator marker",
            default=False,
        )
        simulator_source = "declared"
    status = declared_value(row, "status", "state", default=None)
    return CloudBackendProfile(
        provider=QUANTINUUM_PROVIDER,
        name=name,
        n_qubits=declared_count(
            row, "n_qubits", "nqubits", "qubits", owner=owner, fact="qubit count"
        ),
        basis_gates=(
            declared_gate_names(basis_gates, owner=owner) if basis_gates else gate_names
        ),
        coupling_map=None,
        supports_openqasm=True,
        supports_dynamic_circuits=False,
        is_simulator=simulator,
        metadata={
            "capability_source": "Quantinuum machine listing",
            "declared_status": None if status is None else str(status),
            "qubit_count_source": width_source,
            "simulator_source": simulator_source,
            "connectivity_declared": False,
        },
    )


def _language(qasm_version: float) -> str:
    try:
        return QUANTINUUM_LANGUAGES[qasm_version]
    except KeyError:
        raise ValueError(
            f"Quantinuum submission does not accept QASM {qasm_version:g}; it "
            "accepts " + ", ".join(f"{value:g}" for value in QUANTINUUM_LANGUAGES)
        ) from None


def _quantinuum_counts(payload: Mapping[str, Any], *, shots: int) -> dict[str, int]:
    """Read a Quantinuum result, which reports one entry per returned shot.

    A per-shot list is tallied as returned: the adapter does not reorder or
    reverse a bitstring, because a result that was silently reversed would be
    indistinguishable from a correct one.
    """

    entries = declared_value(payload, "results", "data", default=payload)
    if isinstance(entries, Mapping):
        return _outcome_counts(
            dict(entries),
            provider="Quantinuum",
            keys=("counts", "histogram", "probabilities", "c"),
            shots=shots,
            outcome_key=_quantinuum_outcome,
        )
    if isinstance(entries, str | bytes) or not isinstance(entries, Sequence):
        raise RuntimeError("Quantinuum result response does not contain outcomes.")
    registers: list[Any] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RuntimeError("Quantinuum result entry must be a classical register")
        register = declared_value(entry, "c", "m", "register", default=None)
        if register is None:
            raise RuntimeError("Quantinuum result entry does not contain a register")
        registers.append(register)
    if registers and isinstance(registers[0], Mapping):
        for register in registers:
            if not isinstance(register, Mapping):
                raise RuntimeError(
                    "Quantinuum result mixes per-shot and aggregated registers"
                )
        merged: dict[str, int] = {}
        for register in registers:
            for outcome, value in register.items():
                key = _quantinuum_outcome(outcome)
                merged[key] = merged.get(key, 0) + int(round(float(value)))
        return merged
    outcomes: list[Any] = []
    for register in registers:
        if isinstance(register, str | bytes) or not isinstance(register, Sequence):
            raise RuntimeError("Quantinuum result register must hold bitstrings")
        outcomes.extend(register)
    return _tally_outcomes(
        outcomes, provider="Quantinuum", outcome_key=_quantinuum_outcome
    )


def _quantinuum_outcome(outcome: Any) -> str:
    bits = str(outcome).strip()
    if not bits or any(bit not in "01" for bit in bits):
        raise RuntimeError(f"Quantinuum result contains invalid outcome {outcome!r}")
    return bits


class QuantinuumProvider(QuantumProvider):
    """Submit FlagQuantum OpenQASM packages to one Quantinuum H-series machine.

    Parameters:
        base_url: API root, overridable for a compatible control plane.
        credentials: Bearer token material.
        machine: Machine to target, or ``None`` for submission-independent use.
        machine_width: Width to use for an API version that lists names only.
        basis_gates: The machine's native gate set, as declared by the caller.
        transport: HTTP transport, replaced in tests.

    Raises:
        ValueError: A machine row, QASM version, or gate declaration is unusable.
        RuntimeError: A submission would not be accepted as written, or a result
            payload carries no outcomes.
    """

    provider = QUANTINUUM_PROVIDER

    def __init__(
        self,
        *,
        base_url: str = QUANTINUUM_API_BASE,
        credentials: ProviderCredentials | None = None,
        machine: str | None = None,
        machine_width: int | None = None,
        basis_gates: Sequence[str] = (),
        transport: QuantumCloudTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.credentials = credentials or ProviderCredentials()
        self.machine = machine
        self.machine_width = machine_width
        self.basis_gates = tuple(basis_gates)
        self.transport = transport or UrllibTransport()
        self.timeout = float(timeout)

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
            self._url("/machines"), self._headers(), self.timeout
        )
        rows = declared_value(payload, "machines", "data", default=payload)
        if isinstance(rows, Mapping):
            raise ValueError("Quantinuum machine listing is a mapping, not rows")
        if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
            raise ValueError("Quantinuum machine listing does not contain rows")
        profiles = []
        for row in rows:
            profile = quantinuum_backend_profile(
                row,
                n_qubits=self.machine_width,
                basis_gates=self.basis_gates,
            )
            if n_qubits is not None and profile.n_qubits < n_qubits:
                continue
            profiles.append(profile)
        return tuple(profiles)

    def dry_run(self, package: DeploymentPackage) -> ProviderSubmissionPreview:
        """Validate locally and expose the program without creating a job."""

        return preflight_submission(
            package,
            provider=self.provider,
            backend_name=self.machine,
            qasm_versions=tuple(QUANTINUUM_LANGUAGES),
            dynamic_circuits_supported=False,
        )

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        preview = self.dry_run(package)
        if not preview.compatible:
            raise RuntimeError(
                "Quantinuum preflight failed: " + ", ".join(preview.blockers)
            )
        response = self.transport.post_json(
            self._url("/job"),
            {
                "name": package.name,
                "machine": package.backend.name,
                "language": _language(package.qasm_version),
                "program": package.qasm,
                "count": package.shots,
            },
            self._headers(),
            self.timeout,
        )
        task_id = declared_value(response, "job", "id", "job_id", default=None)
        if task_id is None:
            raise RuntimeError("Quantinuum submit response does not contain a job id.")
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
            self._url("/job/{task_id}", task_id=handle.task_id),
            self._headers(),
            self.timeout,
        )
        status = declared_value(response, "status", "state", default=None)
        if status is None:
            raise RuntimeError("Quantinuum job response does not contain a status.")
        lowered = str(status).lower()
        return _QUANTINUUM_STATUS.get(lowered, str(status))

    def cancel(self, handle: ProviderTaskHandle) -> Any:
        """Ask Quantinuum to cancel the job and report the resulting status."""

        response = self.transport.put_json(
            self._url("/job/{task_id}/cancel", task_id=handle.task_id),
            {},
            self._headers(),
            self.timeout,
        )
        return declared_value(response, "status", "state", default="Cancelled")

    def fetch_result(self, handle: ProviderTaskHandle) -> DeploymentResult:
        response = self.transport.get_json(
            self._url("/job/{task_id}/results", task_id=handle.task_id),
            self._headers(),
            self.timeout,
        )
        shots = int(handle.payload.get("shots", 0))
        if shots <= 0:
            raise RuntimeError(
                "Quantinuum result cannot be read without its shot count."
            )
        counts = _quantinuum_counts(dict(response), shots=shots)
        return DeploymentResult(
            handle=handle,
            counts=counts,
            shots=sum(counts.values()),
            metadata=build_result_metadata(handle, {"bitstring_order": "as_returned"}),
        )


__all__ = (
    "QUANTINUUM_API_BASE",
    "QUANTINUUM_LANGUAGES",
    "QuantinuumProvider",
    "quantinuum_backend_profile",
)
