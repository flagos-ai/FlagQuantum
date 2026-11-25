"""Native Circuit composition for the original seven-bit plasma block encoding."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

import flagquantum as fq

Guard = tuple[int, int]
DTYPE = torch.complex128
X = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
H = torch.tensor([[1, 1], [1, -1]], dtype=DTYPE) / math.sqrt(2)
Z = torch.diag(torch.tensor([1, -1], dtype=DTYPE))
VELOCITY = torch.tensor([0, 1, 2, 3, -4, -3, -2, -1], dtype=torch.float64)
BC = torch.tensor([-3, 3, -1, 0], dtype=torch.float64) / 2
BC_NORM = float(BC.norm())
FORCE = VELOCITY * torch.exp(-VELOCITY.square() / 2) / math.sqrt(2 * math.pi)
FORCE_NORM = float(FORCE.norm())
CURRENT_NORM = float(VELOCITY.norm()) * 8 / 7
ADVECTION_NORM = 2 * (1 + BC_NORM) * 4 / (2 * (100 / 7))
OFF_DIAGONAL_NORM = max(FORCE_NORM, CURRENT_NORM)
NORMALIZATION = ADVECTION_NORM + OFF_DIAGONAL_NORM + 0.8


def prepare(vector: torch.Tensor) -> torch.Tensor:
    """Real orthogonal preparation with an explicit, invertible extension."""
    state = vector.to(DTYPE).reshape(-1)
    if state.numel() & (state.numel() - 1) or not torch.allclose(
        state.norm(), torch.ones((), dtype=torch.float64), atol=1e-12, rtol=0
    ):
        raise ValueError("Preparation requires normalized power-of-two amplitudes")
    if bool(torch.any(state.imag != 0)):
        raise ValueError("This original experiment uses real preparation amplitudes")
    difference = -state.clone()
    difference[0] += 1
    if float(difference.norm()) < 1e-14:
        return torch.eye(state.numel(), dtype=DTYPE)
    difference /= difference.norm()
    return (
        torch.eye(state.numel(), dtype=DTYPE)
        - 2 * difference[:, None] * difference.conj()[None, :]
    )


def ry(angle: float) -> torch.Tensor:
    cosine, sine = math.cos(angle / 2), math.sin(angle / 2)
    return torch.tensor([[cosine, -sine], [sine, cosine]], dtype=DTYPE)


def shift(width: int, amount: int) -> torch.Tensor:
    result = torch.zeros((1 << width, 1 << width), dtype=DTYPE)
    for column in range(1 << width):
        result[(column + amount) % (1 << width), column] = 1
    return result


@dataclass
class Builder:
    circuit: fq.Circuit

    def emit(
        self, matrix: torch.Tensor, wires: Sequence[int], guards: tuple[Guard, ...] = ()
    ) -> None:
        labels = tuple(wires) + tuple(wire for wire, _ in guards)
        if len(set(labels)) != len(labels) or any(
            value not in (0, 1) for _, value in guards
        ):
            raise ValueError(
                "Controlled operations require disjoint wires and binary guards"
            )
        dimension = 1 << len(wires)
        if matrix.shape != (dimension, dimension):
            raise ValueError("Matrix dimension must match its data wires")
        operator = torch.eye(dimension << len(guards), dtype=DTYPE)
        offset = dimension * sum(value << i for i, (_, value) in enumerate(guards))
        operator[offset : offset + dimension, offset : offset + dimension] = matrix
        physical = tuple(self.circuit.n_qubits - 1 - wire for wire in reversed(labels))
        self.circuit.any(*physical, matrix=operator)

    def flip(self, target: int, guards: tuple[Guard, ...] = ()) -> None:
        self.emit(X, (target,), guards)


def derivative(builder: Builder, guards: tuple[Guard, ...]) -> None:
    x, flag, selector, direction = (3, 4, 5), 9, 10, 11
    angle = 2 * math.acos(math.sqrt(BC_NORM / (1 + BC_NORM)))
    builder.emit(ry(angle), (selector,), guards)
    branch = guards + ((selector, 1),)
    builder.emit(H, (direction,), branch)
    builder.emit(Z, (direction,), branch)
    for value, amount in ((0, -1), (1, 1)):
        builder.emit(shift(4, amount), (*x, flag), branch + ((direction, value),))
    builder.emit(H, (direction,), branch)
    for high in (0, 1):
        branch = guards + ((selector, 0), (x[-1], high))
        if high:
            for wire in x[:-1]:
                builder.flip(wire, branch)
            # RY(2 pi) is -I, including its controlled relative phase.
            builder.emit(-torch.eye(2, dtype=DTYPE), (flag,), branch)
        builder.emit(prepare(BC / BC_NORM).mH, x[:-1], branch)
        builder.flip(flag, branch + tuple((wire, 0) for wire in x[:-1]))
        builder.flip(flag, branch)
        if high:
            for wire in x[:-1]:
                builder.flip(wire, branch)
    builder.emit(ry(-angle), (selector,), guards)


def assign(
    builder: Builder,
    amplitudes: torch.Tensor,
    data: Sequence[int],
    target: int,
    guards: tuple[Guard, ...],
) -> None:
    for index, amplitude in enumerate(amplitudes):
        conditions = tuple((wire, (index >> bit) & 1) for bit, wire in enumerate(data))
        builder.emit(
            ry(2 * math.asin(float(amplitude))), (target,), guards + conditions
        )


def block_encoding() -> fq.Circuit:
    """Retain seven data bits and all ten original block/scratch bits."""
    circuit = fq.Circuit(17, dtype=DTYPE)
    builder = Builder(circuit)
    amplitudes = torch.sqrt(
        torch.tensor([ADVECTION_NORM, OFF_DIAGONAL_NORM, 0.8, 0], dtype=torch.float64)
        / NORMALIZATION
    )
    builder.emit(prepare(amplitudes), (7, 8))
    branch = ((7, 0), (8, 0))
    derivative(builder, branch)
    assign(builder, VELOCITY / 4, (0, 1, 2), 12, branch)
    builder.flip(12, branch)
    # Source's two scratch-computed zeta predicates are uncomputed exactly.
    for x_value, negative in ((7, 1), (0, 0)):
        predicate = branch + tuple(
            (wire, (x_value >> i) & 1) for i, wire in enumerate((3, 4, 5))
        )
        builder.flip(15, predicate)
        builder.flip(16, branch + ((2, negative),))
        builder.flip(13, branch + ((15, 1), (16, 1)))
        builder.flip(16, branch + ((2, negative),))
        builder.flip(15, predicate)
    builder.flip(14, branch + ((6, 1),))
    branch = ((7, 1), (8, 0))
    builder.flip(6, branch)
    force = branch + ((6, 0),)
    current = branch + ((6, 1),)
    builder.flip(9, force + ((0, 0), (1, 0), (2, 0)))
    builder.flip(9, force)
    builder.emit(prepare(-FORCE / FORCE_NORM), (0, 1, 2), force)
    builder.emit(prepare(VELOCITY / VELOCITY.norm()).mH, (0, 1, 2), current)
    builder.flip(9, current + ((0, 0), (1, 0), (2, 0)))
    builder.flip(9, current)
    ratio = FORCE_NORM / CURRENT_NORM
    equalization = torch.tensor([ratio, 1.0], dtype=torch.float64)
    equalization /= equalization.max()
    assign(builder, equalization, (6,), 12, branch)
    builder.flip(12, branch)
    # P(pi) RZ(-pi) equals iI exactly, with the selector branch retained.
    builder.emit(1j * torch.eye(2, dtype=DTYPE), (9,), ((7, 0), (8, 1)))
    final = amplitudes.clone()
    final[0] *= -1
    builder.emit(prepare(final).mH, (7, 8))
    return circuit


def input_circuit(vector: torch.Tensor) -> fq.Circuit:
    circuit = fq.Circuit(17, dtype=DTYPE)
    circuit.any(*range(10, 17), matrix=prepare(vector))
    return circuit


def qsvt_step(first_angle: float = 0.1, second_angle: float = 0.2) -> fq.Circuit:
    """Preserve the source's single demonstration pair and both work qubits."""
    if not all(math.isfinite(value) for value in (first_angle, second_angle)):
        raise ValueError("QSVT demonstration angles must be finite")
    circuit = fq.Circuit(18, dtype=DTYPE)
    encoding = block_encoding()
    builder = Builder(circuit)
    for unitary, angle in ((encoding, first_angle), (encoding.adjoint(), second_angle)):
        circuit.compose(unitary, qubits=tuple(range(1, 18)))
        guards = tuple((wire, 0) for wire in range(7, 15))
        builder.flip(17, guards)
        rotation = torch.diag(
            torch.exp(torch.tensor([-0.5j * angle, 0.5j * angle], dtype=DTYPE))
        )
        builder.emit(rotation, (17,))
        builder.flip(17, guards)
    return circuit
