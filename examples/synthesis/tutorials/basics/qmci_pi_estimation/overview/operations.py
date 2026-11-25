"""Native lattice preparation, checked amplification and three-bit phase estimate."""

from math import pi
from typing import Annotated

from flagquantum import synth
from flagquantum.synth import (
    Output,
    QBit,
    QNum,
    adjoint,
    allocate,
    apply,
    control,
    drop,
    hadamard_transform,
    phase,
    qfunc,
    qpe,
    within,
)


@qfunc
def prepare(cutoff: int, x: QNum, y: QNum, indicator: QBit) -> None:
    hadamard_transform(x)
    hadamard_transform(y)
    indicator ^= x * x + y * y < cutoff


@qfunc
def unprepare(cutoff: int, x: QNum, y: QNum, indicator: QBit) -> None:
    with adjoint():
        prepare(cutoff, x, y, indicator)


@qfunc
def reflect(cutoff: int, x: QNum, y: QNum, indicator: QBit) -> None:
    with within():
        unprepare(cutoff, x, y, indicator)
    with apply():
        control((x + y <= 0) & (indicator <= 0), lambda: phase(1.0, pi))
    phase(1.0, pi)


@qfunc
def main(bits: int, k: int, cutoff: int, indicator: Output[QBit]) -> None:
    x: QNum = QNum()
    y: QNum = QNum()
    allocate(bits, x)
    allocate(bits, y)
    allocate(indicator)
    prepare(cutoff, x, y, indicator)
    for i in range(k):
        phase(indicator, pi)
        reflect(cutoff, x, y, indicator)
    drop(x)
    drop(y)


@qfunc
def grover(x: QNum, y: QNum) -> None:
    control(x * x + y * y < 64, lambda: phase(1.0, pi))
    with within():
        hadamard_transform(x)
        hadamard_transform(y)
    with apply():
        control(x + y <= 0, lambda: phase(1.0, pi))
    phase(1.0, pi)


@qfunc
def phase_main(phase_reg: Output[Annotated[QNum, (3, False, 3)]]) -> None:
    x: QNum = QNum()
    y: QNum = QNum()
    allocate(3, x)
    allocate(3, y)
    allocate(phase_reg)
    hadamard_transform(x)
    hadamard_transform(y)
    qpe(lambda: grover(x, y), phase_reg)
    drop(x)
    drop(y)


def build_program(bits: int = 3, k: int = 0) -> synth.Program:
    """Retain 1..4 bits per coordinate and at most 16 Grover iterations."""
    if type(bits) is not int or not 1 <= bits <= 4:
        raise ValueError("use an integer coordinate precision in 1..4")
    if type(k) is not int or not 0 <= k <= 16:
        raise ValueError("use an integer Grover count in 0..16")
    return synth.synthesize(
        main,
        classical_inputs={"bits": bits, "k": k, "cutoff": 1 << (2 * bits)},
        max_qubits=2 * bits + 9,
        max_workspace=8,
        memory_limit_bytes=64 * 1024**2,
    )


def build_qpe() -> synth.Program:
    return synth.synthesize(
        phase_main,
        max_qubits=17,
        max_workspace=8,
        memory_limit_bytes=64 * 1024**2,
    )
