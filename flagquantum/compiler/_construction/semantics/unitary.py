"""Bounded matrix-to-registered-gate synthesis with absolute phase retained."""

import cmath
import math

import torch

from flagquantum.compiler.frontends._structured.model import Operation

from ....core._composition import controlled_instruction
from ....core.controlled import _clean_mcx_instructions
from ....core.ir import Instruction
from ...optimization.one_qubit_optimization import (
    _instruction_matrix as _one_qubit_matrix,
)
from ...optimization.one_qubit_synthesis import _zyz_angles, synthesize_one_qubit_matrix
from ...optimization.two_qubit_optimization import _block_product
from ...optimization.two_qubit_synthesis import synthesize_two_qubit
from ..mcx import one_clean_mcx
from ..verification.profile import _fail


def unitary_instructions(
    matrix: torch.Tensor,
    wires: tuple[int, ...],
    op: Operation,
    controls: tuple[int, ...] = (),
    scratch: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Synthesize registered gates and restore the decomposition's global phase."""
    if len(wires) > 2:
        return _givens_instructions(matrix, wires, op, controls, scratch)
    gates, angle = _small_decomposition(matrix, wires, op)
    if abs(angle) > 1e-15:
        phase = Instruction("phase", (wires[0],), {"theta": angle})
        flip = Instruction("x", (wires[0],))
        gates = (*gates, flip, phase, flip, phase)
    return (
        tuple(leaf for gate in gates for leaf in controlled_instruction(gate, controls))
        if controls
        else gates
    )


def _small_decomposition(
    matrix: torch.Tensor, wires: tuple[int, ...], op: Operation, *, zyz: bool = False
) -> tuple[tuple[Instruction, ...], float]:
    gates: tuple[Instruction, ...] | None
    if len(wires) == 1:
        if zyz:
            rows = [[complex(value.item()) for value in row] for row in matrix]
            theta, phi, lam = _zyz_angles(rows)
            gates = tuple(
                Instruction(name, wires, {"theta": angle})
                for name, angle in (("rz", lam), ("ry", theta), ("rz", phi))
                if angle != 0.0
            )
        else:
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
    return gates, angle


def _controlled_block(
    gates: tuple[Instruction, ...],
    controls: tuple[int, ...],
    scratch: tuple[int, ...],
    global_phase: float = 0.0,
) -> tuple[Instruction, ...]:
    if len(controls) < 2 or len(scratch) < min(len(controls) - 1, 2):
        if global_phase:
            target = gates[0].wires[0]
            flip = Instruction("x", (target,))
            phase = Instruction("phase", (target,), {"theta": global_phase})
            gates = (*gates, flip, phase, flip, phase)
        return tuple(
            leaf for gate in gates for leaf in controlled_instruction(gate, controls)
        )
    flag, *workspace = scratch
    if len(scratch) >= len(controls) - 1:
        predicate = _clean_mcx_instructions(
            controls, flag, workspace[: len(controls) - 2]
        )
    else:
        # Compute the predicate once using the existing borrowed-control ladder.
        # Both clean bits and every borrowed control are restored afterward.
        predicate = one_clean_mcx(controls, flag, workspace[0])
    # The clean flag already represents the whole condition, including when the
    # target is zero. Apply the block's global phase before uncomputing it.
    phase_gates = (
        (Instruction("phase", (flag,), {"theta": global_phase}),)
        if global_phase
        else ()
    )

    return (
        *predicate,
        *(leaf for gate in gates for leaf in controlled_instruction(gate, (flag,))),
        *phase_gates,
        *reversed(predicate),
    )


def _pattern_block(
    gates: tuple[Instruction, ...],
    wires: tuple[int, ...],
    pattern: int,
    external_controls: tuple[int, ...] = (),
    scratch: tuple[int, ...] = (),
    global_phase: float = 0.0,
) -> tuple[Instruction, ...]:
    if not gates:
        return ()
    controls = (
        *external_controls,
        *(wire for wire in wires if wire not in gates[0].wires),
    )
    flips = tuple(
        Instruction("x", (wire,))
        for position, wire in enumerate(wires)
        if wire in controls and not pattern & (1 << (len(wires) - position - 1))
    )
    return (
        *flips,
        *_controlled_block(gates, controls, scratch, global_phase),
        *reversed(flips),
    )


def _two_level_instructions(
    matrix: torch.Tensor,
    first: int,
    second: int,
    wires: tuple[int, ...],
    op: Operation,
    controls: tuple[int, ...],
    scratch: tuple[int, ...],
) -> tuple[Instruction, ...]:
    # Permutations commute with external controls and cancel on the inactive branch.
    # Permute the first basis word along a Gray path, leaving the second fixed.
    path = [first]
    for bit in range(len(wires)):
        if (first ^ second) & (1 << bit):
            path.append(path[-1] ^ (1 << bit))
    permutations: list[tuple[Instruction, ...]] = []
    for left, right in zip(path[:-2], path[1:-1], strict=True):
        bit = (left ^ right).bit_length() - 1
        flip = Instruction("x", (wires[-1 - bit],))
        permutations.append(_pattern_block((flip,), wires, left, scratch=scratch))
    left = path[-2]
    bit = (left ^ second).bit_length() - 1
    target = wires[-1 - bit]
    if left & (1 << bit):
        matrix = matrix.flip((0, 1))
    gates = [gate for permutation in permutations for gate in permutation]
    local, angle = _small_decomposition(matrix, (target,), op, zyz=True)
    share_phase = (
        len(wires) >= 4
        and bool(local)
        and len(scratch) >= len(controls) + len(wires) - 2
    )
    if len(wires) >= 4:
        gates.extend(
            _pattern_block(
                local,
                wires,
                left,
                controls,
                scratch,
                angle if share_phase and abs(angle) > 1e-15 else 0.0,
            )
        )
    else:
        for gate in local:
            gates.extend(_pattern_block((gate,), wires, left, controls))
    if abs(angle) > 1e-15 and not share_phase:
        # A two-level global phase acts on both target values. Condition it on
        # the other bits once instead of controlling four phase-repair gates.
        condition = (*controls, *(wire for wire in wires if wire != target))
        flips = tuple(
            Instruction("x", (wire,))
            for position, wire in enumerate(wires)
            if wire != target and not left & (1 << (len(wires) - position - 1))
        )
        phase = Instruction("phase", (condition[-1],), {"theta": angle})
        gates.extend(
            (
                *flips,
                *_controlled_block((phase,), condition[:-1], scratch),
                *reversed(flips),
            )
        )
    # Each transposition is self-inverse, including its negative-control flips.
    for permutation in reversed(permutations):
        gates.extend(permutation)
    return tuple(gates)


def _givens_instructions(
    matrix: torch.Tensor,
    wires: tuple[int, ...],
    op: Operation,
    controls: tuple[int, ...] = (),
    scratch: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    # Elimination in Gray order makes consecutive basis words differ
    # by one bit, avoiding reversible transpositions around each rotation.
    order = list(range(matrix.shape[0]))
    if len(wires) >= 4:
        order = [word ^ (word >> 1) for word in order]
        matrix = matrix[order][:, order]
    reduced = matrix.clone()
    rotations: list[tuple[int, int, torch.Tensor]] = []
    last = matrix.shape[0] - 1
    for column in range(last):
        for row in range(last, column, -1):
            a, b = reduced[row - 1, column].item(), reduced[row, column].item()
            # Match the phase residual floor instead of promoting last-bit
            # eigensolver/BLAS noise into a controlled rotation network. The
            # complete reconstruction below still enforces its 1e-12 bound.
            if abs(b) <= 1e-15:
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
    for pattern, angle in zip(order, angles.tolist(), strict=True):
        if abs(angle) <= 1e-15:
            continue
        flip = () if pattern & 1 else (Instruction("x", (wires[-1],)),)
        phase = Instruction("phase", (wires[-1],), {"theta": angle})
        gates.extend(
            (*flip, *_pattern_block((phase,), wires, pattern, controls, scratch), *flip)
        )
    for first, second, rotation in reversed(rotations):
        gates.extend(
            _two_level_instructions(
                rotation.mH, order[first], order[second], wires, op, controls, scratch
            )
        )
    return tuple(gates)
