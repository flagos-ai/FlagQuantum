"""Original source verification outside the native example library."""

from pathlib import Path

from examples.synthesis.applications.physical_systems.ising_model.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth


def run_example() -> None:
    for params in BINDINGS:
        program = synth.synthesize(
            Path(__file__).with_name("source.qmod"), classical_inputs={"params": params}
        )
        compare_states(program, reference_circuit(params), ("v.z",))


if __name__ == "__main__":
    run_example()
