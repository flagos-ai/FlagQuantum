"""Independent direct-spin state evolution with the source's exact global phase."""

import numpy as np
import numpy.typing as npt

from .model import WIDTH


def direct_costs() -> npt.NDArray[np.float64]:
    """Use direct spin correlations, independently of the Pauli expansion."""
    words = np.arange(1 << WIDTH)
    spins = 2 * ((words[:, None] >> np.arange(WIDTH)) & 1) - 1
    return np.array(
        [
            sum(np.dot(row[: WIDTH - lag], row[lag:]) ** 2 for lag in range(WIDTH - 1))
            for row in spins
        ],
        dtype=np.float64,
    )


def direct_state(parameters: tuple[float, ...]) -> npt.NDArray[np.complex128]:
    """Apply diagonal energies and RX pairs in physical big-endian order."""
    costs = direct_costs()
    physical_words = np.arange(1 << WIDTH)
    state = np.full(1 << WIDTH, 1 / np.sqrt(1 << WIDTH), dtype=np.complex128)
    for layer in range(3):
        state *= np.exp(-1j * parameters[layer] * costs)
        beta = parameters[3 + layer]
        for bit in range(WIDTH):
            flipped = physical_words ^ (1 << (WIDTH - bit - 1))
            state = np.cos(beta / 2) * state - 1j * np.sin(beta / 2) * state[flipped]
    return state
