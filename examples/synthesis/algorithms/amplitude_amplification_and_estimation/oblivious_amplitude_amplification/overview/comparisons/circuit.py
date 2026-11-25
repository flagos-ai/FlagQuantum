"""Check both actual OAA experiments against independent complete Circuits."""

from examples.synthesis.algorithms.amplitude_amplification_and_estimation.oblivious_amplitude_amplification.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.oblivious_amplitude_amplification.overview.main import (
    run_workflow,
    source_amplitudes,
)
from examples.synthesis.compare_states import compare_states


def run_example() -> None:
    for repetitions, stage in enumerate(run_workflow()):
        compare_states(
            stage.program,
            reference_circuit(repetitions, source_amplitudes()),
            ("data", "block"),
        )


if __name__ == "__main__":
    run_example()
