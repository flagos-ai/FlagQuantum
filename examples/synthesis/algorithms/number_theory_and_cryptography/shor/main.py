"""Original ten-bit Shor period finding through native modular exponentiation."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QNum,
    adjoint,
    allocate,
    drop,
    hadamard_transform,
    modular_exp,
    qft,
    qfunc,
)


@qfunc
def main(n: int, a: int, phase_var: Output[Annotated[QNum, (10, False, 10)]]) -> None:
    residue: Annotated[QNum, 5] = QNum()
    allocate(residue)
    allocate(phase_var)
    residue += 1
    hadamard_transform(phase_var)
    modular_exp(n, a, residue, phase_var)
    with adjoint():
        qft(phase_var)
    drop(residue)


def build_program(modulus: int = 21, base: int = 11) -> synth.Program:
    return synth.synthesize(
        main,
        classical_inputs={"n": modulus, "a": base},
        max_qubits=18,
        max_workspace=3,
        memory_limit_bytes=64 * 1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.numeric_probabilities(result, registers=("phase_var",))))
    print(program.report["resources"])
