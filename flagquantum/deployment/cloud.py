"""Historical deployment namespace; authoritative owners are recorded in the migration ledger."""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flagquantum.compute.remote.packages import (
        CloudBackendProfile as CloudBackendProfile,
    )
    from flagquantum.compute.remote.packages import (
        DeploymentPackage as DeploymentPackage,
    )
    from flagquantum.compute.remote.packages import (
        DeploymentPackageIdentityError as DeploymentPackageIdentityError,
    )
    from flagquantum.compute.remote.packages import _native_program as _native_program
    from flagquantum.compute.remote.packages import (
        validate_deployment_package as validate_deployment_package,
    )
    from flagquantum.runtime.planner.deployment import (
        create_deployment_package as create_deployment_package,
    )
    from flagquantum.runtime.planner.deployment import (
        create_pauli_measurement_plan as create_pauli_measurement_plan,
    )
    from flagquantum.runtime.planner.deployment import deploy_circuit as deploy_circuit
    from flagquantum.runtime.results.counts import (
        PauliMeasurementPlan as PauliMeasurementPlan,
    )
    from flagquantum.runtime.results.counts import (
        _grouped_hamiltonian_statistics as _grouped_hamiltonian_statistics,
    )
    from flagquantum.runtime.results.counts import (
        _measured_coefficient as _measured_coefficient,
    )
    from flagquantum.runtime.results.counts import (
        _validate_count_histogram as _validate_count_histogram,
    )
    from flagquantum.runtime.results.counts import (
        expectation_z_from_counts as expectation_z_from_counts,
    )
    from flagquantum.runtime.results.counts import (
        hamiltonian_expectation_from_counts as hamiltonian_expectation_from_counts,
    )
    from flagquantum.runtime.results.counts import (
        hamiltonian_expectation_from_grouped_counts as hamiltonian_expectation_from_grouped_counts,
    )

__all__ = [
    "CloudBackendProfile",
    "DeploymentPackage",
    "DeploymentPackageIdentityError",
    "PauliMeasurementPlan",
    "create_deployment_package",
    "create_pauli_measurement_plan",
    "deploy_circuit",
    "hamiltonian_expectation_from_counts",
    "hamiltonian_expectation_from_grouped_counts",
    "expectation_z_from_counts",
    "validate_deployment_package",
]
_OWNERS = {
    "CloudBackendProfile": "flagquantum.compute.remote.packages",
    "DeploymentPackage": "flagquantum.compute.remote.packages",
    "DeploymentPackageIdentityError": "flagquantum.compute.remote.packages",
    "_native_program": "flagquantum.compute.remote.packages",
    "validate_deployment_package": "flagquantum.compute.remote.packages",
    "create_deployment_package": "flagquantum.runtime.planner.deployment",
    "create_pauli_measurement_plan": "flagquantum.runtime.planner.deployment",
    "deploy_circuit": "flagquantum.runtime.planner.deployment",
    "PauliMeasurementPlan": "flagquantum.runtime.results.counts",
    "_validate_count_histogram": "flagquantum.runtime.results.counts",
    "expectation_z_from_counts": "flagquantum.runtime.results.counts",
    "_measured_coefficient": "flagquantum.runtime.results.counts",
    "hamiltonian_expectation_from_counts": "flagquantum.runtime.results.counts",
    "hamiltonian_expectation_from_grouped_counts": "flagquantum.runtime.results.counts",
    "_grouped_hamiltonian_statistics": "flagquantum.runtime.results.counts",
}


def __getattr__(name: str) -> Any:
    if name in _OWNERS:
        return getattr(_import_module(_OWNERS[name]), name)
    raise AttributeError(name)
