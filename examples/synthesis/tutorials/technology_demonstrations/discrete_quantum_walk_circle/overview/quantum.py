"""Native controlled walk on circles of synthesis-time size."""

from flagquantum import synth
from flagquantum.synth import (
    H,
    Output,
    QBit,
    QNum,
    X,
    adjoint,
    allocate,
    control,
    qfunc,
)


@qfunc
def increment(width: int, position: QNum) -> None:
    for order in range(width - 1):
        # The compiler expands the callback with the current static index.
        control(
            position[: width - 1 - order],
            lambda: X(position[width - 1 - order]),  # noqa: B023
        )
    X(position[0])


@qfunc
def decrement(width: int, position: QNum) -> None:
    with adjoint():
        increment(width, position)


@qfunc
def main(width: int, position: Output[QNum], coin: Output[QBit]) -> None:
    allocate(width, position)
    allocate(coin)
    H(coin)
    control(coin == 0, lambda: increment(width, position))
    control(coin == 1, lambda: decrement(width, position))


def build(width: int, cap: int) -> synth.Program:
    """Bind the original cascade and a checked physical width ceiling."""
    if type(width) is not int or not 5 <= width <= 11:
        raise ValueError("The source benchmark admits 5 to 11 position bits")
    if type(cap) is not int or not width + 1 <= cap <= 2 * width:
        raise ValueError("The physical width cap is outside the source search range")
    return synth.synthesize(
        main,
        classical_inputs={"width": width},
        max_qubits=cap,
        max_workspace=cap - width - 1,
        objective="width",
        max_operations=250_000,
        memory_limit_bytes=128 * 1024**2,
    )
