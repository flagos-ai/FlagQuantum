"""Compose source antenna cost and mixers directly from fresh real coefficients."""

import math

from flagquantum import synth
from flagquantum.synth import (
    RX,
    Output,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)

from .model import validate_coefficients


@qfunc
def main(
    params: list[float],
    coefficients: list[float],
    x: Output[QArray[QBit]],
    slack: Output[QArray[QBit]],
) -> None:
    allocate(13, x)
    allocate(3, slack)
    hadamard_transform(x)
    hadamard_transform(slack)
    for layer in range(4):
        for i in range(13):
            phase(coefficients[i] * x[i], params[layer])
        phase(-3 * (x[1] + x[3] + x[4] - 1) ** 2, params[layer])
        phase(-3 * (x[0] + x[5] - 1) ** 2, params[layer])
        phase(-3 * (x[6] + x[9] - 1) ** 2, params[layer])
        phase(
            -3
            * (
                x[0]
                + x[1]
                + x[2]
                + x[3]
                + x[4]
                + x[5]
                + x[6]
                + x[7]
                + x[8]
                + x[9]
                + x[10]
                + x[11]
                + x[12]
                + slack[0]
                + 2 * slack[1]
                + 4 * slack[2]
                - 7
            )
            ** 2,
            params[layer],
        )
        for i in range(13):
            RX(params[4 + layer], x[i])
        for i in range(3):
            RX(params[4 + layer], slack[i])


def build_program(
    parameters: tuple[float, ...], coefficients: tuple[float, ...]
) -> synth.Program:
    """Bind source four-layer angles and freshly generated classical data."""
    validate_coefficients(coefficients)
    if len(parameters) != 8 or any(
        type(value) not in (int, float) or not math.isfinite(value)
        for value in parameters
    ):
        raise ValueError("four layers require eight finite real angles")
    return synth.synthesize(
        main,
        classical_inputs={"params": parameters, "coefficients": coefficients},
        max_qubits=16,
        max_workspace=0,
        max_operations=10000,
        memory_limit_bytes=32 * 1024**2,
    )
