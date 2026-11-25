"""Original-size whitebox oracle and Grover program using native quantum logic."""

from typing import Annotated

from flagquantum import synth
from flagquantum.synth import Output, QNum, allocate, hadamard_transform, qfunc

from ..main import main, mark


@qfunc
def oracle(x: Output[Annotated[QNum, 6]], y: Output[Annotated[QNum, 6]]) -> None:
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    mark(x, y)


def build_program(iterations: int | None = 10) -> synth.Program:
    """Retain both six-bit coordinates, ten search rounds and width ceiling 25."""
    if iterations is None:
        return synth.synthesize(
            oracle,
            max_qubits=25,
            max_workspace=8,
            memory_limit_bytes=64 * 1024**2,
        )
    if type(iterations) is not int or not 0 <= iterations <= 16:
        raise ValueError("use a Grover count in 0..16 or None for the bare oracle")
    return synth.synthesize(
        main,
        classical_inputs={"r": iterations},
        max_qubits=25,
        max_workspace=8,
        memory_limit_bytes=64 * 1024**2,
    )
