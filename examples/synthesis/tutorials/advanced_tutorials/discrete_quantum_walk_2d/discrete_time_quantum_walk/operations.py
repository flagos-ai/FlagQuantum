"""Uniform reflection followed by coherent modular moves on a toroidal grid."""

from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk.quantum_walk_hypercube.operations import (
    diffuse,
)
from flagquantum.synth import QNum, control, qfunc


@qfunc
def increment(position: QNum) -> None:
    position += 1


@qfunc
def decrement(position: QNum) -> None:
    position -= 1


@qfunc
def walk_step(x: QNum, y: QNum, coin: QNum) -> None:
    diffuse(coin)
    control(coin <= 0, lambda: increment(x))
    control((coin >= 1) & (coin <= 1), lambda: decrement(x))
    control((coin >= 2) & (coin <= 2), lambda: increment(y))
    control(coin >= 3, lambda: decrement(y))
