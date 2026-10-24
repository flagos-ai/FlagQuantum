"""Check a dynamic circuit against a backend before remote submission."""

import flagquantum as fq
import flagquantum.errors as fqe
from flagquantum.deployment import CloudBackendProfile
from flagquantum.dynamic import DynamicCircuit


def main() -> None:
    circuit = DynamicCircuit(2)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("h", 1, classical_bit=0)

    backend = CloudBackendProfile(
        provider="example",
        name="limited-dynamic-qpu",
        n_qubits=2,
        basis_gates=("x",),
        supports_openqasm=True,
        supports_dynamic_circuits=True,
        max_classical_bits=2,
    )
    report = fq.experimental.dynamic.assess_dynamic_backend(circuit, backend)

    print(f"compatible: {report.compatible}")
    print(f"blockers: {report.blockers}")

    try:
        fq.run(circuit, target="quafu:Baihua", shots=128)
    except fqe.CapabilityError as error:
        print(f"stable remote refusal: {error}")


if __name__ == "__main__":
    main()
