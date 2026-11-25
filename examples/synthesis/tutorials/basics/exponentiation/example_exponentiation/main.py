"""Native first-order product-formula evolution of the retained ordered Pauli data."""

from pathlib import Path

import flagquantum as fq
from examples.synthesis.pauli_evolution import load_operator
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, allocate, qfunc, suzuki_trotter

HAMILTONIAN = load_operator(Path(__file__).with_name("hamiltonian.json"))


@qfunc
def main(state: Output[QArray[QBit]]) -> None:
    allocate(12, state)
    suzuki_trotter(HAMILTONIAN, 1.0, 1, 1, state)


def build_program() -> synth.Program:
    return synth.synthesize(main, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("state",))))
