"""Retained MaxCut original remains test-only evidence."""

from pathlib import Path

from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.workshops.combinatorial_workshop.combinatorial_qmod_workshop_for_maxcut.comparisons.circuit import (
    BINDINGS,
    reference_circuit,
)
from flagquantum import synth


def run_example() -> None:
    program = synth.synthesize(
        Path(__file__).with_name("source.qmod"),
        classical_inputs={"params": BINDINGS[0]},
    )
    compare_states(program, reference_circuit(), ("v",))


if __name__ == "__main__":
    run_example()
