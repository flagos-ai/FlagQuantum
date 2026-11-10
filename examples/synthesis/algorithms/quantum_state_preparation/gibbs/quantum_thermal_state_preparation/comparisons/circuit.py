"""Independent Circuit blocks and Kraus tracing for the complete thermal channel."""

import math

import torch

import flagquantum as fq

from ..physics import ANGLES, GAUSSIAN

DTYPE = torch.complex128
STATE = (0, 1)
BOLTZMANN = 2
BOHR = (3, 4, 5, 6)
JUMP = (7, 8)


def _direct_sum(blocks: list[torch.Tensor]) -> torch.Tensor:
    size = sum(block.shape[0] for block in blocks)
    matrix = torch.zeros((size, size), dtype=DTYPE)
    offset = 0
    for block in blocks:
        width = block.shape[0]
        matrix[offset : offset + width, offset : offset + width] = block
        offset += width
    return matrix


def reference_circuit() -> fq.Circuit:
    gaussian = torch.tensor(GAUSSIAN, dtype=DTYPE)
    assert abs(torch.linalg.vector_norm(gaussian).item() - 1) < 1e-13
    gaussian /= torch.linalg.vector_norm(gaussian)
    identity = torch.eye(16, dtype=DTYPE)
    axis = identity[:, 0] - gaussian
    preparation = identity - 2 * torch.outer(axis, axis.conj()) / axis.vdot(axis)
    block = fq.Circuit(9, dtype=DTYPE).any(*BOHR, matrix=preparation)

    energy = torch.tensor([2, 0, -2, 0], dtype=torch.float64)
    elapsed = 0.2 * math.pi
    evolution = fq.Circuit(9, dtype=DTYPE)
    for bit, control in enumerate(reversed(BOHR)):
        phases = torch.diag(torch.exp(-1j * elapsed * (2**bit) * energy))
        evolution.any(
            control, *STATE, matrix=_direct_sum([torch.eye(4, dtype=DTYPE), phases])
        )
    block.compose(evolution)

    x = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE)
    eye = torch.eye(2, dtype=DTYPE)
    select_jump = _direct_sum(
        [torch.kron(eye, x), torch.kron(eye, y), torch.kron(x, eye), torch.kron(y, eye)]
    )
    block.h(JUMP[0]).h(JUMP[1]).any(*JUMP, *STATE, matrix=select_jump)
    block.compose(evolution.adjoint())

    qft = fq.Circuit(9, dtype=DTYPE)
    bohr_bits = tuple(reversed(BOHR))
    for bit in range(len(bohr_bits) // 2):
        qft.swap(bohr_bits[bit], bohr_bits[-1 - bit])
    for bit, target in enumerate(bohr_bits):
        qft.h(target)
        for other in range(bit + 1, len(bohr_bits)):
            qft.cphase(bohr_bits[other], target, theta=math.pi / 2 ** (other - bit))
    block.compose(qft.adjoint())

    angles = ANGLES
    assert len(angles) == 16
    rotations = []
    for angle in angles:
        c, s = math.cos(angle / 2), math.sin(angle / 2)
        rotations.append(torch.tensor([[c, -s], [s, c]], dtype=DTYPE))
    block.any(*BOHR, BOLTZMANN, matrix=_direct_sum(rotations))
    return block.x(BOLTZMANN)


def kraus_operators(delta: float) -> torch.Tensor:
    block = reference_circuit()
    inverse = block.adjoint()
    theta = math.asin(2 * math.sqrt(delta))
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    # The highest scratch bit in this layout is boltzmann_weight.
    low = ((torch.arange(512) >> 6) & 1) == 0
    columns = []
    for basis in range(4):
        circuit = fq.Circuit(9, dtype=DTYPE)
        for bit, wire in enumerate(reversed(STATE)):
            if basis >> bit & 1:
                circuit.x(wire)
        encoded = fq.run(circuit.compose(block)).statevector().reshape(-1)
        weighted = encoded * torch.where(low, torch.tensor(c, dtype=torch.float64), 1.0)
        recovered = (
            fq.Circuit(9, dtype=DTYPE, inputs=weighted.reshape(1, -1))
            .compose(inverse)
            .state()
            .reshape(-1)
        )
        jump = encoded * low * s
        columns.append(torch.stack((recovered.reshape(4, 128), jump.reshape(4, 128))))
    return torch.stack(columns, dim=-1).permute(0, 2, 1, 3).reshape(256, 4, 4)


def reference_density(
    operators: torch.Tensor, rounds: int, initial: torch.Tensor
) -> torch.Tensor:
    rho = initial[:, None] * initial.conj()[None, :]
    for _ in range(rounds):
        rho = torch.einsum("kij,jl,kml->im", operators, rho, operators.conj())
    return rho


def run_example() -> None:
    """Compare all retained and discarded density entries after fifty cycles."""
    from ..main import build_program

    program = build_program()
    initial = torch.tensor([1, 0, 0, 0], dtype=DTYPE)
    expected = reference_density(kraus_operators(0.1), 50, initial)
    state = fq.run(program).state
    assert isinstance(state, torch.Tensor)
    full = torch.zeros((1024, 1024), dtype=DTYPE)
    full[::256, ::256] = expected
    torch.testing.assert_close(state.reshape(1024, 1024), full, atol=1e-10, rtol=0)
    print(expected)
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
