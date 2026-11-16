"""Test-only comparison of the pinned original transform."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qct_qst.qct_qst_type2.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth


def run_example() -> None:
    program = synth.synthesize(Path(__file__).with_name("source.qmod"), max_workspace=0)
    compare_states(program, reference_circuit(), ("x", "q"))


if __name__ == "__main__":
    run_example()
