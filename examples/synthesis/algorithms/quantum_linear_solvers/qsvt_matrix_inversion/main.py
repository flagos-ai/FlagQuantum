"""Compose the full 70-phase singular-value inversion with native quantum functions."""

import flagquantum as fq
from examples.synthesis.algorithms.quantum_linear_solvers.qsvt_matrix_inversion.data import (
    AMPLITUDES,
    MATRIX,
    PHASES,
)
from flagquantum import synth
from flagquantum.synth import (
    RZ,
    H,
    Output,
    QArray,
    QBit,
    X,
    adjoint,
    allocate,
    control,
    inplace_prepare_amplitudes,
    qfunc,
    unitary,
)


@qfunc
def projected_rotation(angle: float, block: QBit, aux: QBit) -> None:
    control(block == 0, lambda: X(aux))
    RZ(angle, aux)
    control(block == 0, lambda: X(aux))


@qfunc
def transform(phases: list[float], state: QArray[QBit], block: QBit, aux: QBit) -> None:
    H(aux)
    projected_rotation(phases[0], block, aux)
    for i in range(34):
        with adjoint():
            unitary(MATRIX, (state, block))
        projected_rotation(phases[2 * i + 1], block, aux)
        unitary(MATRIX, (state, block))
        projected_rotation(phases[2 * i + 2], block, aux)
    with adjoint():
        unitary(MATRIX, (state, block))
    projected_rotation(phases[69], block, aux)
    H(aux)


@qfunc
def main(
    phases: list[float],
    amplitudes: list[float],
    state: Output[QArray[QBit]],
    block: Output[QBit],
    aux: Output[QBit],
) -> None:
    allocate(2, state)
    allocate(block)
    allocate(aux)
    inplace_prepare_amplitudes(amplitudes, 0, state)
    transform(phases, state, block, aux)


def build_program(phases: tuple[float, ...] = PHASES) -> synth.Program:
    """Keep all source phases and expose the source struct's two fields explicitly."""
    if len(phases) != 70:
        raise ValueError("this inversion model requires all 70 phases")
    return synth.synthesize(
        main,
        classical_inputs={"phases": list(phases), "amplitudes": list(AMPLITUDES)},
        max_qubits=5,
        max_workspace=1,
        memory_limit_bytes=1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("state", "block", "aux"))))
    print(program.report["resources"])
