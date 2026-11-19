"""Generate the original tensor-hypercontraction demonstration inputs."""

import math
from dataclasses import dataclass

import numpy as np

N_EPSILON = 4
N_UD = 3
N_MN = (6 * N_UD - 1).bit_length()
ROWS = (1 << N_MN) - 1
N_D = (N_UD + ROWS * (ROWS // 2) - 1).bit_length()
N_ALEPH = 14 + N_EPSILON
N_ROTATION = N_EPSILON * N_UD
CLOCK_STEPS = (1 << N_EPSILON) - 1
Rows = tuple[tuple[bool, ...], ...]


@dataclass(frozen=True)
class Model:
    rotation: float
    mu_alt: Rows
    nu_alt: Rows
    keep: Rows
    theta: tuple[bool, ...]
    rot_mu: Rows
    rot_nu: Rows
    clock: tuple[float, ...]


def generate() -> Model:
    """Use the original RandomState(0) stream without touching global state."""
    random = np.random.RandomState(0)

    def table(columns: int) -> Rows:
        return tuple(
            tuple(row) for row in random.choice((True, False), (ROWS, columns)).tolist()
        )

    rotation = float(random.rand() * 2 * math.pi)
    mu_alt, nu_alt, keep = table(N_MN), table(N_MN), table(N_ALEPH)
    theta = tuple(random.choice((True, False), ROWS).tolist())
    rot_mu, rot_nu = table(N_ROTATION), table(N_ROTATION)
    size = 1 << N_EPSILON
    amplitudes = tuple(
        math.sqrt(2 / (size + 1)) * math.sin(math.pi * (i + 1) / (size + 1))
        for i in range(size)
    )
    return Model(rotation, mu_alt, nu_alt, keep, theta, rot_mu, rot_nu, amplitudes)
