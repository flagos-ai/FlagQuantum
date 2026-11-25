"""Native constrained Knapsack quantum program with typed numeric storage."""

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.qaoa.main import ansatz
from flagquantum import synth
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, qfunc


@qfunc
def main(params: list[float], a: Output[QNum], b: Output[QNum]) -> None:
    allocate(3, a)
    allocate(2, b)
    hadamard_transform(a)
    hadamard_transform(b)
    ansatz(params, a, b)


def build_program(
    params: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
) -> synth.Program:
    if len(params) != 6:
        raise ValueError("Three-layer Knapsack requires six real parameters")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("a", "b"))))
