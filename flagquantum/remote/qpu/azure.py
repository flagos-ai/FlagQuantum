"""Azure Quantum QIR submission adapter."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ...compiler import CouplingMap
from ...deployment.cloud import (
    CloudBackendProfile,
    DeploymentPackage,
    validate_deployment_package,
)
from .contracts import (
    DeploymentResult,
    ProviderTaskHandle,
    QuantumProvider,
    build_result_metadata,
    build_submission_receipt,
    validate_deployment_result,
)

_BIT = re.compile(r"[01]")


@dataclass(frozen=True)
class AzureSubmissionPreview:
    """Validated, non-submitting description of an Azure Quantum job."""

    target_id: str
    backend: CloudBackendProfile
    shots: int
    program: str
    compatible: bool
    blockers: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "backend": self.backend.name,
            "shots": self.shots,
            "source_format": "openqasm-3",
            "submission_format": "qir",
            "compatible": self.compatible,
            "blockers": self.blockers,
        }


def _attribute(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        found = getattr(value, name, default)
        if found is not default:
            return found
    return default


def azure_backend_profile(
    target: Any,
    *,
    n_wires: int,
    basis_gates: Sequence[str] = (),
    coupling_map: CouplingMap | None = None,
) -> CloudBackendProfile:
    """Build a fail-closed backend profile from an Azure target-like object."""

    width = int(n_wires)
    if width <= 0:
        raise ValueError("Azure Quantum target width must be positive")
    name = str(_attribute(target, "name", "id", "target_id", default="")).strip()
    if not name:
        raise ValueError("Azure Quantum target does not expose an id")
    provider_id = str(_attribute(target, "provider_id", "provider", default="")).strip()
    lowered = f"{provider_id} {name}".lower()
    return CloudBackendProfile(
        provider="azure-quantum",
        name=name,
        n_qubits=width,
        basis_gates=tuple(str(gate).lower() for gate in basis_gates),
        coupling_map=coupling_map,
        supports_openqasm=True,
        supports_dynamic_circuits=False,
        is_simulator=any(
            marker in lowered for marker in (".sim.", "simulator", "emulator")
        ),
        metadata={
            "target_id": name,
            "device_provider": provider_id,
            "capability_source": "explicit Azure Quantum adapter configuration",
            "submission_format": "qir",
        },
    )


def _job_id(job: Any) -> str:
    identifier = _attribute(job, "id", "job_id")
    if callable(identifier):
        identifier = identifier()
    if identifier is None:
        identifier = _attribute(getattr(job, "details", None), "id", "job_id")
    if identifier is None:
        raise RuntimeError("Azure Quantum job does not expose an id")
    return str(identifier)


def _bitstring(value: Any) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        bits = "".join(str(int(bit)) for bit in value)
    else:
        text = str(value).strip()
        bits = "".join(_BIT.findall(text))
        if not bits:
            bits = text
    if not bits or any(bit not in "01" for bit in bits):
        raise RuntimeError(f"Azure Quantum result contains invalid outcome {value!r}")
    return bits


def _probabilities_to_counts(
    probabilities: Mapping[Any, Any], shots: int
) -> dict[str, int]:
    weighted: list[tuple[str, float, int, float]] = []
    for outcome, raw_probability in probabilities.items():
        probability = float(raw_probability)
        if not math.isfinite(probability) or probability < 0:
            raise RuntimeError("Azure Quantum result contains an invalid probability")
        exact = probability * shots
        base = math.floor(exact)
        weighted.append((_bitstring(outcome), probability, base, exact - base))
    total_probability = sum(item[1] for item in weighted)
    if not math.isclose(total_probability, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError("Azure Quantum probabilities do not sum to one")
    remaining = shots - sum(item[2] for item in weighted)
    ranked = sorted(
        range(len(weighted)), key=lambda index: (-weighted[index][3], index)
    )
    increments = set(ranked[:remaining])
    return {
        outcome: base + (index in increments)
        for index, (outcome, _probability, base, _remainder) in enumerate(weighted)
    }


def _azure_counts(payload: Any, *, shots: int) -> dict[str, int]:
    current = payload
    if isinstance(current, Mapping):
        for key in ("counts", "histogram", "probabilities", "results"):
            candidate = current.get(key)
            if isinstance(candidate, Mapping):
                current = candidate
                break
    if not isinstance(current, Mapping) or not current:
        raise RuntimeError("Azure Quantum result does not contain outcomes")
    numeric = {outcome: float(value) for outcome, value in current.items()}
    if all(value.is_integer() and value >= 0 for value in numeric.values()):
        counts = {_bitstring(key): int(value) for key, value in numeric.items()}
        if sum(counts.values()) == shots:
            return counts
    return _probabilities_to_counts(numeric, shots)


class AzureQuantumProvider(QuantumProvider):
    """Submit FlagQuantum OpenQASM 3 packages to one Azure Quantum target."""

    provider = "azure-quantum"

    def __init__(
        self,
        workspace: Any,
        target: str | Any,
        *,
        n_wires: int,
        basis_gates: Sequence[str] = (),
        coupling_map: CouplingMap | None = None,
        program_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if isinstance(workspace, str):
            try:
                from qdk.azure import Workspace
            except ImportError as exc:
                raise ImportError(
                    "AzureQuantumProvider requires qdk[azure] when a resource id "
                    "is supplied"
                ) from exc
            workspace = Workspace(resource_id=workspace)
        self.workspace = workspace
        self.target = (
            workspace.get_targets(target) if isinstance(target, str) else target
        )
        self.backend = azure_backend_profile(
            self.target,
            n_wires=n_wires,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
        )
        self._program_factory = program_factory

    @property
    def target_id(self) -> str:
        return str(self.backend.metadata["target_id"])

    def discover_backends(
        self, n_wires: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        if n_wires is not None and self.backend.n_qubits < int(n_wires):
            return ()
        return (self.backend,)

    def dry_run(self, package: DeploymentPackage) -> AzureSubmissionPreview:
        """Validate locally and expose the source without submitting a paid job."""

        blockers: list[str] = []
        try:
            validate_deployment_package(package)
        except Exception as exc:
            blockers.append(f"invalid_deployment_package:{exc}")
        if package.backend.provider != self.provider:
            blockers.append("backend_provider_is_not_azure_quantum")
        package_target = str(package.backend.metadata.get("target_id", ""))
        if package.backend.name != self.target_id or (
            package_target and package_target != self.target_id
        ):
            blockers.append("deployment_package_targets_a_different_target")
        if package.qasm_version != 3.0:
            blockers.append("azure_quantum_provider_requires_openqasm_3")
        if package.backend.supports_dynamic_circuits:
            blockers.append("azure_quantum_dynamic_circuits_are_not_supported")
        return AzureSubmissionPreview(
            target_id=self.target_id,
            backend=package.backend,
            shots=package.shots,
            program=package.qasm,
            compatible=not blockers,
            blockers=tuple(blockers),
        )

    def _program(self, source: str) -> Any:
        if self._program_factory is not None:
            return self._program_factory(source)
        try:
            from qdk.openqasm import compile as compile_openqasm
        except ImportError as exc:
            raise ImportError(
                "Azure Quantum submission requires qdk with OpenQASM support"
            ) from exc
        return compile_openqasm(source)

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        preview = self.dry_run(package)
        if not preview.compatible:
            raise RuntimeError(
                "Azure Quantum preflight failed: " + ", ".join(preview.blockers)
            )
        job = self.target.submit(
            self._program(preview.program), package.name, shots=preview.shots
        )
        return ProviderTaskHandle(
            provider=self.provider,
            task_id=_job_id(job),
            backend_name=package.backend.name,
            payload=build_submission_receipt(
                package, {"job": job, "shots": package.shots}
            ),
        )

    def _job(self, handle: ProviderTaskHandle) -> Any:
        job = handle.payload.get("job")
        if job is not None:
            return job
        getter = getattr(self.workspace, "get_job", None)
        if not callable(getter):
            raise RuntimeError("Azure Quantum job object is missing from the handle")
        return getter(handle.task_id)

    def query_status(self, handle: ProviderTaskHandle) -> str:
        job = self._job(handle)
        refresh = getattr(job, "refresh", None)
        if callable(refresh):
            refresh()
        status = str(_attribute(getattr(job, "details", job), "status", default=""))
        return {
            "succeeded": "Finished",
            "failed": "Failed",
            "cancelled": "Cancelled",
            "canceled": "Cancelled",
            "waiting": "Running",
            "executing": "Running",
            "finishing": "Running",
        }.get(status.lower(), status or "Running")

    def cancel(self, handle: ProviderTaskHandle) -> Any:
        """Request cancellation through the Azure Quantum job object."""

        job = self._job(handle)
        cancel = getattr(job, "cancel", None)
        if not callable(cancel):
            raise RuntimeError("Azure Quantum job does not support cancellation")
        return cancel()

    def fetch_result(self, handle: ProviderTaskHandle) -> DeploymentResult:
        job = self._job(handle)
        get_results = getattr(job, "get_results", None)
        if not callable(get_results):
            raise RuntimeError("Azure Quantum job does not expose results")
        raw = get_results()
        expected_shots = int(handle.payload.get("shots", 0))
        if expected_shots <= 0:
            expected_shots = int(
                _attribute(getattr(job, "details", job), "shots", default=0)
            )
        if expected_shots <= 0:
            raise RuntimeError("Azure Quantum submission receipt is missing shots")
        counts = _azure_counts(raw, shots=expected_shots)
        return DeploymentResult(
            handle=handle,
            counts=counts,
            shots=sum(counts.values()),
            metadata=build_result_metadata(
                handle,
                {
                    "target_id": self.target_id,
                    "device_provider": self.backend.metadata["device_provider"],
                    "submission_format": "qir",
                },
            ),
        )

    def run(self, package: DeploymentPackage) -> DeploymentResult:
        """Submit and block through ``get_results()``, matching QDK semantics."""

        return validate_deployment_result(self.fetch_result(self.submit(package)))


__all__ = (
    "AzureQuantumProvider",
    "AzureSubmissionPreview",
    "azure_backend_profile",
)
