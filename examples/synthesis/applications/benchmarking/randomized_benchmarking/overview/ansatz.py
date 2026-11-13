"""Five fresh seeded native Clifford words and their checked inverses."""

import math
import random
from collections.abc import Callable
from typing import Literal

from flagquantum import synth
from flagquantum.synth import (
    RX,
    RZ,
    H,
    Output,
    QBit,
    X,
    adjoint,
    allocate,
    phase,
    qfunc,
    unitary,
)

Gate = Literal["id", "rz", "sx", "x"]
Sequence = tuple[Gate, ...]
LENGTHS = (5, 10, 15, 20, 25)
GATES: tuple[Gate, ...] = ("id", "rz", "sx", "x")
IDENTITY = ((1, 0), (0, 1))


@qfunc
def identity(target: QBit) -> None:
    unitary(IDENTITY, target)


@qfunc
def rz_pi(target: QBit) -> None:
    RZ(math.pi, target)


@qfunc
def sqrt_x(target: QBit) -> None:
    phase(1, math.pi / 4)
    RX(math.pi / 2, target)


def fresh_sequences(seed: int = 0) -> tuple[Sequence, ...]:
    """Retain source hardware-basis filtering order and Python RNG sequence."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("use a nonnegative 63-bit integer seed")
    rng = random.Random(seed)
    return tuple(tuple(rng.choice(GATES) for _ in range(size)) for size in LENGTHS)


def sequence_function(gates: Sequence) -> Callable[[QBit], None]:
    """Capture immutable native gates without evaluating quantum data in Python."""
    if not 1 <= len(gates) <= 25 or any(gate not in GATES for gate in gates):
        raise ValueError("use one through twenty-five admitted Clifford gates")
    if len(gates) == 1:
        gate = gates[0]
        if gate == "id":
            return identity
        if gate == "rz":
            return rz_pi
        if gate == "sx":
            return sqrt_x
        return X
    middle = len(gates) // 2
    left = sequence_function(gates[:middle])
    right = sequence_function(gates[middle:])

    @qfunc
    def sequence(target: QBit) -> None:
        left(target)
        right(target)

    return sequence


def make_model(
    gates: Sequence, *, with_inverse: bool = True, initial: int = 0
) -> Callable[[Output[QBit]], None]:
    """Capture one original word, optionally retaining its forward prefix."""
    if (
        type(with_inverse) is not bool
        or type(initial) is not int
        or initial not in (0, 1, 2)
    ):
        raise ValueError("use a boolean inverse flag and initial state 0, 1 or 2")
    forward = sequence_function(gates)
    body: Callable[[QBit], None]
    if with_inverse:

        @qfunc
        def body(target: QBit) -> None:
            forward(target)
            with adjoint():
                forward(target)

    else:
        body = forward
    if initial == 0:

        def main(target: Output[QBit]) -> None:
            allocate(target)
            body(target)

    elif initial == 1:

        def main(target: Output[QBit]) -> None:
            allocate(target)
            X(target)
            body(target)

    else:

        def main(target: Output[QBit]) -> None:
            allocate(target)
            H(target)
            RZ(math.pi / 2, target)
            body(target)

    return main


def build_program(
    gates: Sequence, *, with_inverse: bool = True, initial: int = 0
) -> synth.Program:
    return synth.synthesize(
        qfunc(make_model(gates, with_inverse=with_inverse, initial=initial)),
        max_qubits=1,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )
