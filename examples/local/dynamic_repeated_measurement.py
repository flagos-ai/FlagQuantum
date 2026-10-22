"""Observe repeated-measurement and reset semantics on one qubit."""

import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit


def main() -> None:
    circuit = DynamicCircuit(1)
    circuit.h(0)
    circuit.measure(0, classical_bit=0)
    circuit.measure(0, classical_bit=1)
    circuit.reset(0)
    circuit.measure(0, classical_bit=2)

    result = fq.experimental.dynamic.run_dynamic(circuit, shots=64, seed=19)
    repeated_measurement_preserved_collapse = torch.equal(
        result.classical_bits[:, 0],
        result.classical_bits[:, 1],
    )
    reset_returned_qubit_to_zero = bool(
        torch.count_nonzero(result.classical_bits[:, 2]).item() == 0
        and torch.count_nonzero(result.final_samples).item() == 0
    )

    print(
        "repeated measurement preserved collapse:",
        repeated_measurement_preserved_collapse,
    )
    print("reset returned the qubit to zero:", reset_returned_qubit_to_zero)


if __name__ == "__main__":
    main()
