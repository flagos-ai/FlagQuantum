"""Compare width and depth objectives for the same quantum arithmetic program."""

from typing import Literal

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler import estimate_resources
from flagquantum.drawer import draw
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, qfunc


@qfunc
def main(x: Output[QNum], y: Output[QNum]) -> None:
    allocate(3, x)
    hadamard_transform(x)
    y |= x**2 + 1


def build_program(objective: Literal["width", "depth"]) -> synth.Program:
    return synth.synthesize(main, objective=objective, memory_limit_bytes=512 * 1024**2)


def run_example() -> None:
    objectives: tuple[Literal["width", "depth"], ...] = ("depth", "width")
    for objective in objectives:
        program = build_program(objective)
        print(draw(program.circuit))
        resources = estimate_resources(program.circuit)
        print(
            objective,
            "resources:",
            {"width": resources.n_qubits, "depth": resources.depth},
        )
        result = fq.run(program)
        print(dict(program.probabilities(result, registers=("x", "y"))))


if __name__ == "__main__":
    run_example()
