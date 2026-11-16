"""Prepare the retained one-qubit molecular variational ansatz natively."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, X, allocate, qfunc, suzuki_trotter

GENERATOR = -0.5 * fq.Y(0)
DEFAULT_PARAMS = (0.37,)


@qfunc
def prepare_reference(state: Output[QArray[QBit]]) -> None:
    allocate(1, state)
    X(state[0])


@qfunc
def main(params: list[float], state: Output[QArray[QBit]]) -> None:
    prepare_reference(state)
    suzuki_trotter(GENERATOR, params[0], 1, 1, state)


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    if len(params) != 1:
        raise ValueError("The retained molecular ansatz requires one parameter")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("state",))))
