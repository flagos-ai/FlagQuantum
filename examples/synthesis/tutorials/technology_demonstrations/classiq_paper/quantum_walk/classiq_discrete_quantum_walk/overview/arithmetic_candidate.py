"""Second reversible candidate for bounded CX-count comparison."""

from typing import Annotated

from flagquantum import synth
from flagquantum.synth import H, Output, QBit, QNum, allocate, control, qfunc


@qfunc
def increment(x: QNum) -> None:
    x += 1


@qfunc
def decrement(x: QNum) -> None:
    x -= 1


@qfunc
def main(x: Output[Annotated[QNum, 6]], coin: Output[QBit]) -> None:
    allocate(x)
    allocate(coin)
    H(coin)
    control(coin == 0, lambda: increment(x))
    control(coin == 1, lambda: decrement(x))


def build_program() -> synth.Program:
    return synth.synthesize(main, max_qubits=12, max_workspace=5)
