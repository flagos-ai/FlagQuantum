"""Native Vlasov-Ampere block encoding with signed velocity and clean work."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    allocate,
    apply,
    control,
    hadamard_transform,
    inplace_prepare_state,
    qfunc,
    within,
)

from .operations import advective, off_diagonal, outer_phase


@qfunc
def main(
    electric: Output[QBit],
    x: Output[Annotated[QNum, 3]],
    velocity: Output[Annotated[QNum, (3, True, 0)]],
    reference_e: Output[QBit],
    reference_x: Output[Annotated[QNum, 3]],
    reference_v: Output[Annotated[QNum, (3, True, 0)]],
    selector: Output[Annotated[QNum, 2]],
    second: Output[QBit],
    first: Output[QBit],
    indicator: Output[QBit],
    block: Output[QBit],
    zeta: Output[QBit],
    flag: Output[QBit],
) -> None:
    allocate(electric)
    allocate(x)
    allocate(velocity)
    allocate(selector)
    allocate(second)
    allocate(first)
    allocate(indicator)
    allocate(block)
    allocate(zeta)
    allocate(flag)
    hadamard_transform(electric)
    hadamard_transform(x)
    hadamard_transform(velocity)
    reference_e |= electric
    reference_x |= x
    reference_v |= velocity
    with within():
        inplace_prepare_state(
            [0.0960237307895, 0.817686639833667, 0.086289629376833, 0.0], 0, selector
        )
    with apply():
        control(
            selector == 0,
            lambda: advective(
                electric, x, velocity, flag, zeta, block, indicator, first, second
            ),
        )
        control(
            selector == 1, lambda: off_diagonal(electric, velocity, flag, indicator)
        )
        outer_phase(selector)


def build_program() -> synth.Program:
    return synth.synthesize(
        main,
        max_qubits=24,
        max_workspace=2,
        max_operations=100_000,
        memory_limit_bytes=256 * 1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(program.report["resources"])
