"""A concrete square-root QSVT program with reversible comparison XOR."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    H,
    Output,
    QBit,
    QNum,
    allocate,
    apply,
    control,
    hadamard_transform,
    phase,
    qfunc,
    within,
)

PHASES = (
    -2.401688306254694,
    2.669374314067509,
    3.817567704355659,
    3.817567704355659,
    2.669374314067509,
    -14.968058920613865,
)


@qfunc
def project_zero(angle: float, aux: QBit, register: QNum) -> None:
    # RZ(angle) on aux XOR (register == 0), without mutating either borrow.
    phase(aux - 0.5, angle)
    control(register < 1, lambda: phase(1 - 2 * aux, angle))


@qfunc
def main(
    phases: list[float],
    state: Output[QNum],
    ref: Output[QNum],
    ind: Output[QBit],
    qsvt_aux: Output[QBit],
) -> None:
    allocate(4, state)
    allocate(4, ref)
    allocate(ind)
    allocate(qsvt_aux)
    hadamard_transform(state)
    H(qsvt_aux)
    project_zero(phases[0], qsvt_aux, ref)
    for step in range(2):
        with within():
            hadamard_transform(ref)
            ind ^= state <= ref
        with apply():
            project_zero(phases[2 * step + 1], qsvt_aux, ind)
        project_zero(phases[2 * step + 2], qsvt_aux, ref)
    hadamard_transform(ref)
    ind ^= state <= ref
    project_zero(phases[5], qsvt_aux, ind)
    H(qsvt_aux)


def build_program(phases: tuple[float, ...] = PHASES) -> synth.Program:
    """Keep all ten logical outputs; this is the source's concrete polynomial."""
    if len(phases) != 6:
        raise ValueError("this concrete QSVT program requires six phases")
    return synth.synthesize(main, classical_inputs={"phases": list(phases)})


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(
        dict(
            program.probabilities(result, registers=("state", "ref", "ind", "qsvt_aux"))
        )
    )
    print(program.report["resources"])
