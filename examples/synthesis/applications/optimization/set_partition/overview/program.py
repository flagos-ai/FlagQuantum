"""Source-data set partition with native integer parameters and quantum functions."""

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

INITIAL_PARAMETERS = (0.13, -0.27, 0.19, 0.22, 0.31, -0.17)


def source_weights() -> tuple[int, ...]:
    """Reproduce the Notebook's seeded data without changing NumPy's global RNG."""
    import numpy as np

    return tuple(int(value) for value in np.random.RandomState(0).randint(1, 12, 10))


def validate_weights(weights: tuple[int, ...]) -> None:
    """Bound the ten original decision variables and their positive integer data."""
    if len(weights) != 10 or any(
        isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 32
        for value in weights
    ):
        raise ValueError("weights require ten integers in [1, 32]")


@qfunc
def cost_layer(gamma: float, weights: list[int], v: QArray[QBit]) -> None:
    phase(
        -(
            (
                weights[0] * (2 * v[0] - 1)
                + weights[1] * (2 * v[1] - 1)
                + weights[2] * (2 * v[2] - 1)
                + weights[3] * (2 * v[3] - 1)
                + weights[4] * (2 * v[4] - 1)
                + weights[5] * (2 * v[5] - 1)
                + weights[6] * (2 * v[6] - 1)
                + weights[7] * (2 * v[7] - 1)
                + weights[8] * (2 * v[8] - 1)
                + weights[9] * (2 * v[9] - 1)
            )
            ** 2
        ),
        gamma,
    )


@qfunc
def main(params: list[float], weights: list[int], v: Output[QArray[QBit]]) -> None:
    allocate(10, v)
    hadamard_transform(v)
    for layer in range(3):
        cost_layer(params[layer], weights, v)
        for index in range(10):
            RX(params[3 + layer], v[index])


def build_program(
    weights: tuple[int, ...], parameters: tuple[float, ...] = INITIAL_PARAMETERS
) -> synth.Program:
    """Bind source data to a native program with explicit local resource limits."""
    validate_weights(weights)
    if len(parameters) != 6 or any(not math.isfinite(value) for value in parameters):
        raise ValueError("three layers require six finite radian parameters")
    return synth.synthesize(
        main,
        classical_inputs={"weights": weights, "params": parameters},
        max_qubits=10,
        max_workspace=0,
        max_operations=5000,
        memory_limit_bytes=4 * 1024**2,
    )
