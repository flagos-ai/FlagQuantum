"""Check the untouched Grover-mixer QAOA source against direct controlled RY."""

from pathlib import Path

from examples.synthesis.algorithms.search_and_optimization.grover_mixer_qaoa.gm_qaoa.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.search_and_optimization.grover_mixer_qaoa.gm_qaoa.main import (
    PARAMETERS,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth


def run_example() -> None:
    compare_states(
        synth.synthesize(
            Path(__file__).with_name("source.qmod"),
            classical_inputs={"params": list(PARAMETERS)},
        ),
        reference_circuit(),
        ("x",),
    )


if __name__ == "__main__":
    run_example()
