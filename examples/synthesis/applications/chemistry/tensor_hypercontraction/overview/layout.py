"""Retain every ELL, PSI, and clock wire and reuse clean circuit scratch."""

from dataclasses import dataclass

from .model import N_ALEPH, N_D, N_EPSILON, N_MN, N_ROTATION, N_UD


@dataclass(frozen=True)
class Layout:
    mu: tuple[int, ...]
    nu: tuple[int, ...]
    m1: int
    succ: int
    r_br: int
    theta: int
    s: tuple[int, ...]
    theta_alt: int
    mu_alt: tuple[int, ...]
    nu_alt: tuple[int, ...]
    keep: tuple[int, ...]
    sigma: tuple[int, ...]
    aux: int
    down: tuple[int, ...]
    up: tuple[int, ...]
    plus1: int
    plus2: int
    clock: tuple[int, ...]
    rotation: tuple[int, ...]
    comparison: tuple[int, ...]
    path: tuple[int, ...]
    ladder: tuple[int, ...]
    width: int

    @property
    def ell(self) -> tuple[int, ...]:
        return tuple(range(self.down[0]))

    @property
    def logical_width(self) -> int:
        return self.rotation[0]


def allocate_layout() -> Layout:
    offset = 0

    def span(size: int) -> tuple[int, ...]:
        nonlocal offset
        wires = tuple(range(offset, offset + size))
        offset += size
        return wires

    def bit() -> int:
        return span(1)[0]

    return Layout(
        span(N_MN),
        span(N_MN),
        bit(),
        bit(),
        bit(),
        bit(),
        span(N_D),
        bit(),
        span(N_MN),
        span(N_MN),
        span(N_ALEPH),
        span(N_ALEPH),
        bit(),
        span(N_UD),
        span(N_UD),
        bit(),
        bit(),
        span(N_EPSILON),
        span(N_ROTATION),
        span(N_ALEPH + 4),
        span(N_MN + N_EPSILON),
        span(8),
        offset,
    )
