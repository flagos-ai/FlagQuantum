"""QFold logical data passes through a coin/shift/inverse-coin step."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    allocate,
    apply,
    drop,
    inplace_prepare_amplitudes,
    qfunc,
    within,
)

from .operations import coin, oracle, shift
from .physics import AMPLITUDES, ANGLES


@qfunc
def main(
    a: float, b: float, amplitudes: list[float], phi: Output[QBit], psi: Output[QBit]
) -> None:
    coin_bit: QBit = QBit()
    move_bit: QBit = QBit()
    allocate(coin_bit)
    allocate(move_bit)
    allocate(phi)
    allocate(psi)
    inplace_prepare_amplitudes(amplitudes, 0, (psi, phi))
    with within():
        coin(a, b, coin_bit, move_bit, psi, phi)
    with apply():
        shift(coin_bit, move_bit, psi, phi)
    oracle(coin_bit, move_bit)
    drop(coin_bit)
    drop(move_bit)


def build_program(
    a: float = ANGLES[0],
    b: float = ANGLES[1],
    amplitudes: tuple[float, ...] = AMPLITUDES,
) -> synth.Program:
    if len(amplitudes) != 4:
        raise ValueError("Use four real logical-data amplitudes")
    return synth.synthesize(
        main, classical_inputs={"a": a, "b": b, "amplitudes": list(amplitudes)}
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("phi", "psi"))))
    print(program.report["resources"])
