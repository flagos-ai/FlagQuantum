"""Test-only comparison of the byte-identical retained variational source."""

from pathlib import Path

from examples.synthesis.applications.chemistry.second_quantized_hamiltonian.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth


def run_example() -> None:
    for params in BINDINGS:
        program = synth.synthesize(
            Path(__file__).with_name("source.qmod"),
            classical_inputs={"params": params},
            max_workspace=0,
        )
        compare_states(program, reference_circuit(params), ("state",))


if __name__ == "__main__":
    run_example()
