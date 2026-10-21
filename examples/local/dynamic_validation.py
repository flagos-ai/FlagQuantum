"""Show the supported local path and fail-closed dynamic-circuit errors."""

import flagquantum as fq
import flagquantum.errors as fqe
from flagquantum.dynamic import DynamicCircuit


def main() -> None:
    circuit = DynamicCircuit(2)
    circuit.h(0)
    circuit.measure(0, classical_bit=0)
    circuit.conditional("x", 1, classical_bit=0)

    result = fq.experimental.dynamic.run_dynamic(circuit, shots=32, seed=7)
    assert result.final_samples.shape == (32, 2)

    try:
        DynamicCircuit(1).conditional("x", 0, classical_bit=0)
    except fqe.ValidationError as error:
        print(f"build-time validation: {error}")

    try:
        circuit.run_distributed(world_size=1)
    except fqe.CapabilityError as error:
        print(f"distributed boundary: {error}")


if __name__ == "__main__":
    main()
