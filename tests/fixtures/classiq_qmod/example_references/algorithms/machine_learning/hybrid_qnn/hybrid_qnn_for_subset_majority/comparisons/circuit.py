"""Check the unchanged source against independent Circuit amplitudes."""

from pathlib import Path

from examples.synthesis.algorithms.machine_learning.hybrid_qnn.hybrid_qnn_for_subset_majority.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize


def run_example() -> None:
    for scale in (1.0, -0.63):
        data = [0.2 * scale, -0.3 * scale, 0.4 * scale, -0.5 * scale]
        weights = [0.1 * scale * (i + 1) for i in range(3)]
        program = synthesize(
            Path(__file__).with_name("source.qmod"),
            classical_inputs={"input_vec": data, "weight": weights},
        )
        compare_states(program, reference_circuit(data, weights), ("result",))


if __name__ == "__main__":
    run_example()
