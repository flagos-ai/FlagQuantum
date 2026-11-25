"""Native first-order product-formula evolution of the retained ordered Pauli data."""

from pathlib import Path

import flagquantum as fq
from examples.synthesis.pauli_evolution import load_operator
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, allocate, qfunc, suzuki_trotter

HAMILTONIAN = load_operator(Path(__file__).with_name("hamiltonian.json"))


@qfunc
def main(t: float, qba: Output[QArray[QBit]]) -> None:
    allocate(2, qba)
    suzuki_trotter(HAMILTONIAN, t, 1, 1, qba)


def build_program(t: float = 0.37) -> synth.Program:
    return synth.synthesize(main, classical_inputs={"t": t}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("qba",))))
