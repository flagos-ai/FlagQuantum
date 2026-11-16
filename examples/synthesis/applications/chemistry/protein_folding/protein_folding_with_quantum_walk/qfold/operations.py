"""QFold coin, conditional shifts and zero-state phase oracle."""

from math import pi

from flagquantum.synth import H, QBit, R, X, control, phase, qfunc


@qfunc
def coin(
    a: float, b: float, coin_bit: QBit, move_bit: QBit, psi: QBit, phi: QBit
) -> None:
    X(coin_bit)
    H(move_bit)
    control((phi <= 0) & (move_bit <= 0), lambda: R(a, pi / 2, coin_bit))
    control((psi <= 0) & (move_bit >= 1), lambda: R(b, pi / 2, coin_bit))
    X(coin_bit)


@qfunc
def shift(coin_bit: QBit, move_bit: QBit, psi: QBit, phi: QBit) -> None:
    control((coin_bit >= 1) & (move_bit >= 1), lambda: X(psi))
    control((coin_bit >= 1) & (move_bit <= 0), lambda: X(phi))


@qfunc
def oracle(coin_bit: QBit, move_bit: QBit) -> None:
    X(coin_bit)
    X(move_bit)
    control(coin_bit, lambda: phase(move_bit, pi))
    X(coin_bit)
    X(move_bit)
