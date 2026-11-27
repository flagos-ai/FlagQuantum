"""Export and package a synthesized Bell circuit using a local target profile."""

from flagquantum.compiler import CouplingMap, translate
from flagquantum.deployment import (
    CloudBackendProfile,
    DeploymentPackage,
    create_deployment_package,
)
from flagquantum.synth import CX, H, Output, QBit, allocate, qfunc, synthesize


@qfunc
def bell(left: Output[QBit], right: Output[QBit]) -> None:
    allocate(left)
    allocate(right)
    H(left)
    CX(left, right)


def build_artifacts() -> tuple[str, DeploymentPackage]:
    """Return standalone OpenQASM 3 and a package for a fixed local target."""
    circuit = synthesize(bell).circuit
    qasm = translate(circuit, format="openqasm-3.0")
    backend = CloudBackendProfile(
        provider="local",
        name="offline-line",
        n_qubits=2,
        basis_gates=("h", "cx"),
        coupling_map=CouplingMap.line(2),
        is_simulator=True,
    )
    package = create_deployment_package(
        circuit, backend=backend, name="bell", shots=256, qasm_version=2.0
    )
    return qasm, package


def main() -> None:
    qasm, package = build_artifacts()
    print(qasm)
    print()
    print(
        f"DeploymentPackage({package.name}, "
        f"target={package.backend.provider}/{package.backend.name}, "
        f"shots={package.shots})"
    )
    print(package.qasm)


if __name__ == "__main__":
    main()
