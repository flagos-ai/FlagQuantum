"""Bounded matrix-to-registered-gate synthesis with absolute phase retained."""

import cmath
import math

import torch

from ...core._composition import controlled_instruction
from ...core.ir import Instruction
from .._hybrid.model import Operation
from ..one_qubit_optimization import _instruction_matrix as _one_qubit_matrix
from ..one_qubit_synthesis import synthesize_one_qubit_matrix
from ..two_qubit_optimization import _block_product
from ..two_qubit_synthesis import synthesize_two_qubit
from .profile import _fail


def unitary_instructions(
    matrix: torch.Tensor, wires: tuple[int, ...], op: Operation
) -> tuple[Instruction, ...]:
    """Synthesize registered gates and restore the decomposition's global phase."""
    if len(wires) == 3:
        return _three_qubit_instructions(matrix, wires, op)
    if len(wires) == 1:
        gates = synthesize_one_qubit_matrix(
            matrix.tolist(), qubit=wires[0], z_rotation="rz"
        )
        assert gates is not None
        product = torch.eye(2, dtype=torch.complex128)
        for gate in gates:
            leaf = _one_qubit_matrix(gate)
            assert leaf is not None
            product = torch.tensor(leaf, dtype=torch.complex128) @ product
    else:
        gates = synthesize_two_qubit(
            matrix.tolist(), qubits=wires, entangler="cx", z_rotation="rz"
        )
        assert gates is not None
        block = _block_product(gates, wires)
        assert block is not None
        product = torch.tensor(block, dtype=torch.complex128)
    angle = cmath.phase(complex(torch.trace(product.mH @ matrix).item()))
    if not torch.allclose(product * cmath.exp(1j * angle), matrix, atol=1e-12, rtol=0):
        _fail(
            "lowering.unitary",
            "registered gate decomposition failed matrix verification",
            op,
        )
    if abs(angle) <= 1e-15:
        return gates
    phase = Instruction("phase", (wires[0],), {"theta": angle})
    flip = Instruction("x", (wires[0],))
    return (*gates, flip, phase, flip, phase)


def _pattern_gate(
    gate: Instruction, wires: tuple[int, ...], pattern: int
) -> tuple[Instruction, ...]:
    controls = tuple(wire for wire in wires if wire not in gate.wires)
    flips = tuple(
        Instruction("x", (wire,))
        for position, wire in enumerate(wires)
        if wire in controls and not pattern & (1 << (len(wires) - position - 1))
    )
    return (*flips, *controlled_instruction(gate, controls), *reversed(flips))


def _two_level_instructions(
    matrix: torch.Tensor,
    first: int,
    second: int,
    wires: tuple[int, ...],
    op: Operation,
) -> tuple[Instruction, ...]:
    # Permute the first basis word along a Gray path, leaving the second fixed.
    path = [first]
    for bit in range(len(wires)):
        if (first ^ second) & (1 << bit):
            path.append(path[-1] ^ (1 << bit))
    permutations: list[tuple[Instruction, ...]] = []
    for left, right in zip(path[:-2], path[1:-1], strict=True):
        bit = (left ^ right).bit_length() - 1
        flip = Instruction("x", (wires[-1 - bit],))
        permutations.append(_pattern_gate(flip, wires, left))
    left = path[-2]
    bit = (left ^ second).bit_length() - 1
    target = wires[-1 - bit]
    if left & (1 << bit):
        matrix = matrix.flip((0, 1))
    gates = [gate for permutation in permutations for gate in permutation]
    for gate in unitary_instructions(matrix, (target,), op):
        gates.extend(_pattern_gate(gate, wires, left))
    # Each transposition is self-inverse, including its negative-control flips.
    for permutation in reversed(permutations):
        gates.extend(permutation)
    return tuple(gates)


def _three_qubit_instructions(
    matrix: torch.Tensor, wires: tuple[int, ...], op: Operation
) -> tuple[Instruction, ...]:
    reduced = matrix.clone()
    rotations: list[tuple[int, int, torch.Tensor]] = []
    for column in range(7):
        for row in range(7, column, -1):
            a, b = reduced[row - 1, column].item(), reduced[row, column].item()
            if abs(b) <= 1e-16:
                continue
            norm = math.hypot(abs(a), abs(b))
            rotation = torch.tensor(
                [[a.conjugate() / norm, b.conjugate() / norm], [-b / norm, a / norm]],
                dtype=torch.complex128,
            )
            reduced[row - 1 : row + 1] = rotation @ reduced[row - 1 : row + 1]
            rotations.append((row - 1, row, rotation))
    angles = torch.angle(reduced.diag())
    reconstructed = torch.diag(torch.exp(1j * angles))
    for first, second, rotation in reversed(rotations):
        reconstructed[first : second + 1] = (
            rotation.mH @ reconstructed[first : second + 1]
        )
    if not torch.allclose(reconstructed, matrix, atol=1e-12, rtol=0):
        _fail(
            "lowering.unitary", "Givens reconstruction failed matrix verification", op
        )
    gates: list[Instruction] = []
    for pattern, angle in enumerate(angles.tolist()):
        if abs(angle) <= 1e-15:
            continue
        flip = () if pattern & 1 else (Instruction("x", (wires[-1],)),)
        phase = Instruction("phase", (wires[-1],), {"theta": angle})
        gates.extend((*flip, *_pattern_gate(phase, wires, pattern), *flip))
    for first, second, rotation in reversed(rotations):
        gates.extend(_two_level_instructions(rotation.mH, first, second, wires, op))
    return tuple(gates)
