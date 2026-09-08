"""In-memory test double for the remote submission contract."""

from __future__ import annotations

from ..circuit import Circuit
from ..deployment.cloud import (
    CloudBackendProfile,
    DeploymentPackage,
    DeploymentResult,
    ProviderTaskHandle,
    QuantumProvider,
    build_result_metadata,
    build_submission_receipt,
    validate_deployment_package,
)


class InMemoryRemoteTarget(QuantumProvider):
    """Exercise the remote task lifecycle locally without network I/O."""

    provider = "local"

    def __init__(self) -> None:
        self._packages: dict[str, DeploymentPackage] = {}
        self._results: dict[str, DeploymentResult] = {}
        self._counter = 0

    def discover_backends(
        self, n_wires: int | None = None
    ) -> tuple[CloudBackendProfile, ...]:
        return (CloudBackendProfile.simulator(n_wires or 32, provider=self.provider),)

    def submit(self, package: DeploymentPackage) -> ProviderTaskHandle:
        validate_deployment_package(package)
        self._counter += 1
        task_id = f"local-{self._counter}"
        handle = ProviderTaskHandle(
            provider=self.provider,
            task_id=task_id,
            backend_name=package.backend.name,
            payload=build_submission_receipt(package, {"name": package.name}),
        )
        circuit = Circuit.from_ir(package.ir)
        counts = circuit.counts(package.shots)[0]
        result = DeploymentResult(
            handle=handle,
            counts=counts,
            shots=package.shots,
            metadata=build_result_metadata(
                handle,
                {
                    "qasm_version": package.qasm_version,
                    "backend": package.backend.name,
                    "simulated": True,
                },
            ),
        )
        self._packages[task_id] = package
        self._results[task_id] = result
        return handle

    def query_status(self, handle: ProviderTaskHandle) -> str:
        return "Finished" if handle.task_id in self._results else "Unknown"

    def fetch_result(self, handle: ProviderTaskHandle) -> DeploymentResult:
        if handle.task_id not in self._results:
            raise KeyError(f"Unknown deployment task {handle.task_id!r}.")
        return self._results[handle.task_id]


__all__ = ["InMemoryRemoteTarget"]
