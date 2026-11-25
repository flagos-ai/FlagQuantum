"""Compare the retained Drude source against a direct independent Circuit."""

from pathlib import Path

from examples.synthesis.applications.chemistry.quantum_drude_oscillator.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize


def run_example() -> None:
    for scale in (0.37, -0.81):
        params = [scale * 0.1 * (index + 1) for index in range(40)]
        compare_states(
            synthesize(
                Path(__file__).with_name("source.qmod"),
                classical_inputs={"params": params},
            ),
            reference_circuit(params),
            ("state",),
        )


if __name__ == "__main__":
    run_example()
