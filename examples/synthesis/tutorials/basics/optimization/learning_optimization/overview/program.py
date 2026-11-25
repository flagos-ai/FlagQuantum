"""One native QAOA layer for two bounded nonnegative integers."""

import math
from typing import Annotated

from flagquantum import synth
from flagquantum.synth import (
    RX,
    Output,
    QNum,
    allocate,
    control,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def objective_layer(gamma: float, x: QNum, y: QNum) -> None:
    """Apply the objective and the positive-part-squared inequality penalty."""
    phase(3 * x + 2 * y, -gamma)
    control(
        3 * x + y < 2,
        lambda: phase(20 * (2 - 3 * x - y) * (2 - 3 * x - y), -gamma),
    )


@qfunc
def main(
    gamma: float,
    beta: float,
    x: Output[Annotated[QNum, 2]],
    y: Output[Annotated[QNum, 2]],
) -> None:
    """Prepare uniform integers, apply the cost, and mix each bit."""
    allocate(x)
    allocate(y)
    hadamard_transform(x)
    hadamard_transform(y)
    objective_layer(gamma, x, y)
    for bit in range(2):
        RX(2 * beta, x[bit])
        RX(2 * beta, y[bit])


def build_program(gamma: float, beta: float) -> synth.Program:
    """Synthesize finite radian angles within a four-qubit local policy."""
    if not math.isfinite(gamma) or not math.isfinite(beta):
        raise ValueError("gamma and beta must be finite radian angles")
    return synth.synthesize(
        main,
        classical_inputs={"gamma": gamma, "beta": beta},
        max_qubits=4,
        max_workspace=0,
        max_operations=1000,
        memory_limit_bytes=1024**2,
    )
