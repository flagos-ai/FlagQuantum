"""The source eight-spin Hamiltonian and ordered 74-mixer pool."""

from dataclasses import dataclass
from itertools import combinations

import torch

SINGLES = (
    -0.561775,
    -1.238125,
    -1.13145,
    -0.68135,
    -0.713775,
    -0.97945,
    -1.145025,
    -0.77445,
)
PAIRS = (
    (0, 1, 0.5),
    (0, 2, 0.5),
    (0, 4, -0.188125),
    (0, 5, -0.2001),
    (1, 3, 0.5),
    (1, 6, 0.23455),
    (1, 7, 0.053575),
    (2, 3, 0.5),
    (2, 4, -0.0481),
    (2, 5, 0.22955),
    (3, 6, -0.039525),
    (3, 7, -0.229125),
    (4, 5, 0.5),
    (4, 6, 0.5),
    (5, 7, 0.5),
    (6, 7, 0.5),
)
WIDTH = 8


@dataclass(frozen=True)
class Mixer:
    axis: str
    supports: tuple[tuple[int, ...], ...]

    @property
    def kind(self) -> int:
        if len(self.supports) == WIDTH:
            return 0 if self.axis == "X" else 1
        if len(self.supports[0]) == 1:
            return 2 if self.axis == "X" else 3
        return 4 if self.axis == "X" else 5


def pool() -> tuple[Mixer, ...]:
    sites = tuple((i,) for i in range(WIDTH))
    pairs = tuple(combinations(range(WIDTH), 2))
    return (
        *(Mixer(axis, sites) for axis in ("X", "Y")),
        *(Mixer(axis, (site,)) for axis in ("X", "Y") for site in sites),
        *(Mixer(axis, (pair,)) for axis in ("X", "Y") for pair in pairs),
    )


def costs() -> torch.Tensor:
    words = torch.arange(1 << WIDTH)
    z = 1 - 2 * ((words[:, None] >> torch.arange(WIDTH)) & 1).to(torch.float64)
    cost = z @ torch.tensor(SINGLES, dtype=torch.float64)
    for a, b, w in PAIRS:
        cost += w * z[:, a] * z[:, b]
    return cost


def gradients(
    state: torch.Tensor, diagonal: torch.Tensor, mixers: tuple[Mixer, ...]
) -> torch.Tensor:
    """Evaluate each absolute commutator expectation over the full vector."""
    words = torch.arange(1 << WIDTH)
    output = []
    for mixer in mixers:
        action = torch.zeros_like(state)
        for support in mixer.supports:
            source = words ^ sum(1 << bit for bit in support)
            if mixer.axis == "Y":
                sign = torch.ones_like(diagonal)
                for bit in support:
                    sign *= (1 - 2 * ((source >> bit) & 1)).to(torch.float64)
                phase = (1j ** len(support)) * sign
            else:
                phase = torch.ones_like(diagonal)
            action += phase * (diagonal[source] - diagonal) * state[source]
        output.append(torch.vdot(state, action).abs())
    return torch.stack(output)
