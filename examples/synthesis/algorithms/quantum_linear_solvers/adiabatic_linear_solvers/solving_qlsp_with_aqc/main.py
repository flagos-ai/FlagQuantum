"""Prepare a source state and compose all 200 native AQC evolutions."""

import flagquantum as fq
from examples.synthesis.algorithms.quantum_linear_solvers.adiabatic_linear_solvers.solving_qlsp_with_aqc.physics import (
    H0,
    H1,
    TIMES,
)
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, prepare_state, qfunc, suzuki_trotter


@qfunc
def main(times: list[float], qba: Output[QArray[QBit]]) -> None:
    prepare_state(
        [
            0.119106699751861,
            0.082712985938792,
            0.23904052936311,
            0.559139784946237,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        0,
        qba,
    )
    for step in range(100):
        suzuki_trotter(H0, times[2 * step], 1, 1, qba)
        suzuki_trotter(H1, times[2 * step + 1], 1, 10, qba)


def build_program() -> synth.Program:
    return synth.synthesize(main, classical_inputs={"times": TIMES}, max_workspace=0)


def run_example() -> None:
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("qba",))))


if __name__ == "__main__":
    run_example()
