"""Amazon Braket backend discovery, preflight, submission, and results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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
from .result_parsing import _normalize_counts


@dataclass(frozen=True)
class BraketSubmissionPreview:
    """Validated, non-submitting description of an Amazon Braket task."""

    device_arn: str
    backend: CloudBackendProfile
    shots: int
    program: str
    compatible: bool
    blockers: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        return {
            "device_arn": self.device_arn,
            "backend": self.backend.name,
            "shots": self.shots,
            "program_format": "openqasm-3",
            "dynamic_dialect": self.backend.dynamic_dialect,
            "compatible": self.compatible,
            "blockers": self.blockers,
        }


def _property_value(value: Any, *path: str, default: Any = None) -> Any:
    current = value
    for key in path:
        if isinstance(current, Mapping):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
        if current is default:
            break
    return current


def _first_property(value: Any, paths: tuple[tuple[str, ...], ...]) -> Any:
    marker = object()
    for path in paths:
        found = _property_value(value, *path, default=marker)
        if found is not marker and found is not None:
            return found
    return None


def _braket_coupling_map(properties: Any, n_wires: int) -> Any:
    from ...compiler import CouplingMap

    graph = _first_property(
        properties,
        (
            ("paradigm", "connectivity", "connectivityGraph"),
            ("paradigm", "connectivity", "connectivity_graph"),
        ),
    )
    edges: set[tuple[int, int]] = set()
    if isinstance(graph, Mapping):
        for source, targets in graph.items():
            for target in targets or ():
                left, right = sorted((int(source), int(target)))
                if left != right:
                    edges.add((left, right))
    if not edges:
        return None
    return CouplingMap(n_wires=n_wires, edges=tuple(sorted(edges)))


def _braket_dynamic_groups(properties: Any) -> tuple[tuple[int, ...], ...]:
    raw = _first_property(
        properties,
        (
            ("paradigm", "dynamicQubitGroups"),
            ("paradigm", "dynamic_qubit_groups"),
            ("provider", "dynamicQubitGroups"),
            ("provider", "dynamic_qubit_groups"),
        ),
    )
    if not raw:
        return ()
    return tuple(tuple(int(wire) for wire in group) for group in raw)


def braket_backend_profile(
    device: Any,
    *,
    dynamic_qubit_groups: tuple[tuple[int, ...], ...] | None = None,
) -> CloudBackendProfile:
    """Build a fail-closed backend profile from an ``AwsDevice``-like object."""

    properties = getattr(device, "properties", None)
    if properties is None:
        raise ValueError("Braket device does not expose properties")
    n_wires = _first_property(
        properties,
        (
            ("paradigm", "qubitCount"),
            ("paradigm", "qubit_count"),
            ("service", "qubitCount"),
        ),
    )
    if n_wires is None:
        raise ValueError("Braket device properties do not declare a qubit count")
    n_wires = int(n_wires)
    basis = (
        _first_property(
            properties,
            (
                ("paradigm", "nativeGateSet"),
                ("paradigm", "native_gate_set"),
            ),
        )
        or ()
    )
    arn = str(getattr(device, "arn", getattr(device, "device_arn", "")))
    name = str(getattr(device, "name", arn or "braket-device"))
    provider_name = str(
        _first_property(properties, (("provider", "providerName"),))
        or _first_property(properties, (("provider", "provider_name"),))
        or ""
    )
    is_iqm = "iqm" in f"{provider_name} {name} {arn}".lower()
    groups = (
        tuple(tuple(int(wire) for wire in group) for group in dynamic_qubit_groups)
        if dynamic_qubit_groups is not None
        else _braket_dynamic_groups(properties)
    )
    metadata: dict[str, object] = {
        "device_arn": arn,
        "device_provider": provider_name,
        "capability_source": "AwsDevice.properties",
    }
    if groups:
        metadata["dynamic_qubit_groups"] = groups
    return CloudBackendProfile(
        provider="amazon-braket",
        name=name,
        n_wires=n_wires,
        basis_gates=tuple(str(gate).lower() for gate in basis),
        coupling_map=_braket_coupling_map(properties, n_wires),
        supports_openqasm=True,
        supports_dynamic_circuits=is_iqm,
        max_classical_bits=None,
        is_simulator="simulator" in f"{name} {arn}".lower(),
        metadata=metadata,
        dynamic_dialect="braket_iqm" if is_iqm else None,
    )


class AmazonBraketProvider(QuantumProvider):
    """Amazon Braket OpenQASM adapter with an explicit no-submit preflight."""

    provider = "amazon-braket"

    def __init__(
        self,
        device: Any,
        *,
        dynamic_qubit_groups: tuple[tuple[int, ...], ...] | None = None,
        program_factory: Any | None = None,
    ) -> None:
        if isinstance(device, str):
            try:
                from braket.aws import AwsDevice
            except ImportError as exc:
                raise ImportError(
                    "AmazonBraketProvider requires amazon-braket-sdk when a device "
                    "ARN is supplied"
                ) from exc
            device = AwsDevice(device)
        self.device = device
        self.backend = braket_backend_profile(
            device,
            dynamic_qubit_groups=dynamic_qubit_groups,
        )
        self._program_factory = program_factory

    @property
    def device_arn(self) -> str:
        return str(self.backend.metadata.get("device_arn", ""))

    def discover_backends(
        self, n_wires: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        if n_wires is not None and self.backend.n_wires < int(n_wires):
            return ()
        return (self.backend,)

    def dry_run(self, package: DeploymentPackage) -> BraketSubmissionPreview:
        """Validate locally and return the exact payload without creating a task."""

        blockers: list[str] = []
        try:
            validate_deployment_package(package)
        except Exception as exc:
            blockers.append(f"invalid_deployment_package:{exc}")
        if package.backend.provider != self.provider:
            blockers.append("backend_provider_is_not_amazon_braket")
        package_arn = str(package.backend.metadata.get("device_arn", ""))
        if self.device_arn and package_arn != self.device_arn:
            blockers.append("deployment_package_targets_a_different_device")
        if package.qasm_version != 3.0:
            blockers.append("amazon_braket_provider_requires_openqasm_3")
        if (
            package.backend.dynamic_dialect == "braket_iqm"
            and not package.backend.metadata.get("dynamic_qubit_groups")
        ):
            blockers.append("braket_iqm_dynamic_qubit_groups_are_required")
        return BraketSubmissionPreview(
            device_arn=self.device_arn,
            backend=package.backend,
            shots=package.shots,
            program=package.qasm,
            compatible=not blockers,
            blockers=tuple(blockers),
        )

    def _program(self, source: str) -> Any:
        if self._program_factory is not None:
            return self._program_factory(source=source)
        try:
            from braket.ir.openqasm import Program
        except ImportError as exc:
            raise ImportError(
                "Amazon Braket submission requires amazon-braket-sdk"
            ) from exc
        return Program(source=source)

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        preview = self.dry_run(package)
        if not preview.compatible:
            raise RuntimeError(
                "Amazon Braket preflight failed: " + ", ".join(preview.blockers)
            )
        task = self.device.run(self._program(preview.program), shots=preview.shots)
        task_id = getattr(task, "id", None)
        if callable(task_id):
            task_id = task_id()
        if task_id is None:
            raise RuntimeError("Amazon Braket task does not expose an id")
        return ProviderTaskHandle(
            provider=self.provider,
            task_id=str(task_id),
            backend_name=package.backend.name,
            payload=build_submission_receipt(package, {"task": task}),
        )

    def query_status(self, handle: ProviderTaskHandle) -> str:
        task = handle.payload.get("task")
        if task is None:
            raise RuntimeError("Amazon Braket task object is missing from the handle")
        state = str(task.state()).upper()
        return {
            "COMPLETED": "Finished",
            "FAILED": "Failed",
            "CANCELLED": "Failed",
            "CANCELLING": "Running",
            "CREATED": "Running",
            "QUEUED": "Running",
            "RUNNING": "Running",
        }.get(state, state.title())

    def fetch_result(self, handle: ProviderTaskHandle) -> DeploymentResult:
        task = handle.payload.get("task")
        if task is None:
            raise RuntimeError("Amazon Braket task object is missing from the handle")
        raw = task.result()
        counts = getattr(raw, "measurement_counts", None)
        if counts is None:
            raise RuntimeError(
                "Amazon Braket result does not contain measurement counts"
            )
        normalized = _normalize_counts(counts)
        return DeploymentResult(
            handle=handle,
            counts=normalized,
            shots=sum(normalized.values()),
            metadata=build_result_metadata(
                handle,
                {
                    "device_arn": self.device_arn,
                    "mid_circuit_measurements_returned": (
                        False if self.backend.dynamic_dialect == "braket_iqm" else None
                    ),
                },
            ),
        )

    def run(self, package: DeploymentPackage) -> DeploymentResult:
        """Submit and block through ``task.result()``, matching Braket semantics."""

        handle = self.submit(package)
        result = self.fetch_result(handle)
        return validate_deployment_result(result)
