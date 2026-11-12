"""Test-only unchanged source against the independent native Circuit reference."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.workshops.algo_design_qce_tutorial.algo_design_qce_tutorial_part_i.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth


def run_example() -> None:
    program = synth.synthesize(Path(__file__).with_name("source.qmod"), max_workspace=0)
    compare_states(program, reference_circuit(), ("x", "y"))


if __name__ == "__main__":
    run_example()
