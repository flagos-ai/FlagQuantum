"""Compose a parameter-array ansatz with native Pauli-evolution helpers."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, allocate, qfunc, suzuki_trotter

GENERATOR = fq.X(0) @ fq.X(3)


@qfunc
def adapt_layer(theta: float, target: QArray[QBit]) -> None:
    suzuki_trotter(GENERATOR, theta, 1, 1, target)


@qfunc
def adapt_vqe_ansatz(params: list[float], target: QArray[QBit]) -> None:
    adapt_layer(params[0], target)


@qfunc
def main(params: list[float], v: Output[QArray[QBit]]) -> None:
    allocate(4, v)
    adapt_vqe_ansatz(params, v)


def build_program(theta: float = 0.23) -> synth.Program:
    return synth.synthesize(main, classical_inputs={"params": [theta]})


def run_example() -> None:
    """Execute the retained single-layer ansatz and read named probabilities."""
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("v",))))


if __name__ == "__main__":
    run_example()
