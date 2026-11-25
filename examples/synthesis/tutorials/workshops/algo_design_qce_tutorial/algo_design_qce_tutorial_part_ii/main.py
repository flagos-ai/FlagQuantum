"""Compose the native three-layer constrained Knapsack tutorial."""

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.qaoa.main import (
    MIXER_A,
    MIXER_B,
    apply_cost,
)
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QNum,
    allocate,
    hadamard_transform,
    qfunc,
    suzuki_trotter,
)


@qfunc
def init_layer(a: QNum, b: QNum) -> None:
    hadamard_transform(a)
    hadamard_transform(b)


@qfunc
def mixer_layer(beta: float, a: QNum, b: QNum) -> None:
    # All X terms commute: exp(-i*beta*sum(X)/2) applies RX(beta) to each bit.
    suzuki_trotter(MIXER_A, beta / 2, 1, 1, a)
    suzuki_trotter(MIXER_B, beta / 2, 1, 1, b)


@qfunc
def main(params: list[float], a: Output[QNum], b: Output[QNum]) -> None:
    allocate(3, a)
    allocate(2, b)
    init_layer(a, b)
    for layer in range(3):
        apply_cost(params[layer], a, b)
        mixer_layer(params[3 + layer], a, b)


DEFAULT_PARAMS = (
    0.003,
    -0.006,
    0.009000000000000001,
    -0.012,
    0.015,
    -0.018000000000000002,
)


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    if len(params) != 6:
        raise ValueError("Three Knapsack layers require six real parameters")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


def run_example() -> None:
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("a", "b"))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
