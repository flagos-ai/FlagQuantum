"""Dicke-state Grover-mixer QAOA using checked native function composition."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CRX,
    CX,
    Output,
    QArray,
    QBit,
    S,
    X,
    adjoint,
    allocate,
    apply,
    control,
    phase,
    qfunc,
    within,
)

PARAMETERS = (0.13, 0.47, -0.39, 0.11, 0.9, -0.29)


@qfunc
def split_cycle(theta: float, qubits: QArray[QBit]) -> None:
    with within():
        CX(qubits[1], qubits[0])
    with apply():
        # S RX(theta) S† is RY(theta), including its controlled branch phase.
        with adjoint():
            S(qubits[1])
        CRX(theta, qubits[0], qubits[1])
        S(qubits[1])


@qfunc
def initial_state(qubits: QArray[QBit]) -> None:
    X(qubits[0])
    split_cycle(2.0943951023932, qubits)
    split_cycle(1.91063323624902, qubits[1:4])
    split_cycle(1.5707963267949, qubits[2:4])


@qfunc
def mixer(beta: float, qubits: QArray[QBit]) -> None:
    with within():
        with adjoint():
            initial_state(qubits)
        X(qubits[3])
    with apply():
        control(
            qubits[0] + qubits[1] + qubits[2] == 0,
            lambda: phase(qubits[3] - 0.5, -0.318309886183791 * beta),
        )


@qfunc
def main(params: list[float], x: Output[QArray[QBit]]) -> None:
    allocate(4, x)
    initial_state(x)
    for layer in range(3):
        phase(x[0] + 2 * x[1] - x[2] + 12 * x[3], params[layer])
        mixer(params[layer + 3], x)


def build_program(params: tuple[float, ...] = PARAMETERS) -> synth.Program:
    if len(params) != 6:
        raise ValueError("params must contain three cost and three mixer angles")
    return synth.synthesize(
        main, classical_inputs={"params": list(params)}, max_workspace=0
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("x",))))
    print(program.report["resources"])
