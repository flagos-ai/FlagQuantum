"""Test-only comparison of the byte-identical retained all-ones control source."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.technology_demonstrations.hardware_aware_mcx.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth


def run_example() -> None:
    program = synth.synthesize(
        Path(__file__).with_name("source.qmod"), memory_limit_bytes=512 * 1024**2
    )
    compare_states(program, reference_circuit(), ("cntrl", "target"))


if __name__ == "__main__":
    run_example()
