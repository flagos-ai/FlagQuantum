"""Execute the unchanged source and compare its complete complex state."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.unitary.comparisons.circuit import reference_circuit
from flagquantum.synth import synthesize


def run_example() -> None:
    compare_states(
        synthesize(Path(__file__).with_name("source.qmod")), reference_circuit(), ("x",)
    )


if __name__ == "__main__":
    run_example()
