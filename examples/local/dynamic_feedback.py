"""Run a local dynamic circuit with measurement, feedback, and reset."""

import torch

import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit


def main() -> None:
    circuit = DynamicCircuit(2)
    circuit.h(0)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)
    circuit.reset(0)
    circuit.measure(0, classical_bit=1)

    dynamic = fq.experimental.dynamic.run_dynamic(circuit, shots=128, seed=7)
    result = dynamic.to_execution_result()

    assert torch.equal(dynamic.final_samples[:, 1], dynamic.classical_bits[:, 0])
    assert torch.count_nonzero(dynamic.final_samples[:, 0]) == 0
    assert torch.count_nonzero(dynamic.classical_bits[:, 1]) == 0
    assert result.runtime["mid_circuit_measurements_available"] is True
    print(
        {
            "final_samples": dynamic.final_samples[:8].tolist(),
            "classical_bits": dynamic.classical_bits[:8].tolist(),
            "execution_semantics": dynamic.execution_semantics,
        }
    )


if __name__ == "__main__":
    main()
