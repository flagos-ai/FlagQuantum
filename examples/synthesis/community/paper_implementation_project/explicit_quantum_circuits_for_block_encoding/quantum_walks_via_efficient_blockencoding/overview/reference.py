"""Independent small state transitions for all structured block examples."""

import math

import torch

from .model import COMPLEX


def hadamard(width: int) -> torch.Tensor:
    size = 1 << width
    return torch.tensor(
        [
            [(-1) ** ((out & value).bit_count()) for value in range(size)]
            for out in range(size)
        ],
        dtype=COMPLEX,
    ) / math.sqrt(size)


def tree_action(
    size: int, coefficients: tuple[float, ...], initial: torch.Tensor
) -> torch.Tensor:
    """Expand each selector and carry branch, then Fourier-sum the selector."""
    result = torch.zeros(8, 2, 2, size, dtype=COMPLEX)
    for label in range(8):
        for vertex in range(size):
            target, carry = vertex, 0
            if label in (0, 1):
                target, carry = (2 * vertex + label) % size, (2 * vertex) // size
            if label in (2, 3):
                child = (vertex - (label - 2)) % size
                target, carry = child // 2, child % 2
            if label < 4:
                angle = coefficients[0]
            elif vertex >= size // 2:
                angle = coefficients[2]
            else:
                angle = coefficients[1] + (coefficients[3] if vertex == 0 else 0)
            amplitude = initial[vertex] / math.sqrt(8)
            result[label, 0, carry, target] += amplitude * math.cos(angle / 2)
            result[label, 1, carry, target] += amplitude * math.sin(angle / 2)
    return (hadamard(3) @ result.reshape(8, -1)).flatten()


def rotation(angle: float) -> torch.Tensor:
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    return torch.tensor(((c, -s), (s, c)), dtype=COMPLEX)


def cyclic_operator(theta: tuple[float, float]) -> torch.Tensor:
    label_prep = torch.kron(torch.eye(4, dtype=COMPLEX), rotation(theta[0]))
    second = torch.eye(8, dtype=COMPLEX)
    for selector in (1, 5):
        columns = torch.tensor((selector, selector + 2))
        second[columns[:, None], columns] = rotation(theta[1])
    toffoli = torch.eye(8, dtype=COMPLEX)
    toffoli[[3, 7]] = toffoli[[7, 3]]
    prepare = toffoli @ second @ label_prep
    operator = torch.zeros(64, 64, dtype=COMPLEX)
    for b in range(8):
        for j in range(8):
            operator[((-b) % 8) * 8 + (j + b) % 8, 8 * b + j] = 1
    lift = torch.kron(prepare, torch.eye(8, dtype=COMPLEX))
    return lift.mH @ operator @ lift


def chebyshev_state(
    operator: torch.Tensor, amplitudes: tuple[float, ...]
) -> torch.Tensor:
    """Evaluate the original two i-Z reflections, retaining their global phase."""
    state = torch.zeros((2, 64), dtype=COMPLEX)
    state[:, :8] = torch.tensor(amplitudes, dtype=COMPLEX) / math.sqrt(2)
    selection = torch.ones(64, dtype=COMPLEX)
    selection[:8] = -1
    sign = torch.stack((selection, -selection))
    for _ in range(2):
        state = (state * (1j * sign)) @ operator.T
    return (hadamard(1) @ state).flatten()


def walk_step(state: torch.Tensor, width: int, *, marked: bool) -> torch.Tensor:
    """Apply one discriminant walk to the clean-ancilla data sector."""
    transformed = state.clone()
    transformed[0] *= -1
    if marked:
        rotate = rotation(1.5708)
        controlled = rotate
        for _ in range(1, width):
            controlled = torch.kron(controlled, rotate)
        transformed[:, 0] = controlled @ transformed[:, 0]
    h = hadamard(width)
    advanced = (h @ transformed @ h).T.contiguous()
    if marked:
        advanced[:, 0] = controlled.mH @ advanced[:, 0]
    return advanced
