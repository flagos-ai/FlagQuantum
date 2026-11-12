"""Native molecular walk sequence with a signed phase register."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, QNum, X, allocate, drop, qfunc

from .operations import estimate
from .physics import PROBABILITIES, selection_members


@qfunc
def main(
    probabilities: list[float],
    x_members: list[int],
    y_members: list[int],
    z_members: list[int],
    powers: list[int],
    block: Output[Annotated[QNum, 4]],
    phase_var: Output[Annotated[QNum, (5, True, 5)]],
) -> None:
    data: QArray[QBit] = QArray()
    allocate(4, data)
    X(data[0])
    X(data[2])
    allocate(block)
    allocate(phase_var)
    estimate(
        probabilities, x_members, y_members, z_members, powers, block, data, phase_var
    )
    drop(data)


def build_program() -> synth.Program:
    """Keep every model bit, selector leaf and source phase-estimation step."""
    return synth.synthesize(
        main,
        classical_inputs={
            "probabilities": list(PROBABILITIES),
            "x_members": list(selection_members("X")),
            "y_members": list(selection_members("Y")),
            "z_members": list(selection_members("Z")),
            "powers": [2**index for index in range(4)],
        },
        max_qubits=24,
        max_operations=100_000,
        memory_limit_bytes=1024**3,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.numeric_probabilities(result, registers=("phase_var",))))
    print(program.report["resources"])
