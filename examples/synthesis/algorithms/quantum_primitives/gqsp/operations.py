"""Single-ancilla generalized signal-processing rotation."""

from math import pi

from flagquantum.synth import QBit, U, X, apply, phase, qfunc, within


@qfunc
def rotate(theta: float, phi: float, lam: float, ind: QBit) -> None:
    with within():
        X(ind)
    with apply():
        U(2 * theta, phi, lam, 0, ind)
    phase(ind, pi)
