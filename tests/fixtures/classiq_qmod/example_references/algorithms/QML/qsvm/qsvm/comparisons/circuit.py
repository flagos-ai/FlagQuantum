"""Keep original-source verification outside the native user library."""

from pathlib import Path

from examples.synthesis.algorithms.machine_learning.qsvm.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.compare_states import compare_states
from flagquantum.synth import synthesize

BINDINGS = {"data1": [0.27, -0.43], "data2": [-0.61, 0.18]}


def run_example() -> None:
    compare_states(
        synthesize(Path(__file__).with_name("source.qmod"), classical_inputs=BINDINGS),
        reference_circuit(
            (BINDINGS["data1"][0], BINDINGS["data1"][1]),
            (BINDINGS["data2"][0], BINDINGS["data2"][1]),
        ),
        ("qba",),
    )


if __name__ == "__main__":
    run_example()
