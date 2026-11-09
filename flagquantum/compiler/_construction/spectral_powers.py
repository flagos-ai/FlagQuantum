"""Share a verified eigenbasis over consecutive controlled powers of a unitary."""

from collections.abc import Mapping, Sequence

import torch

from ...core.ir import Instruction
from .._hybrid.model import Operation, ValueId
from .unitary import unitary_instructions


def lower_spectral_powers(
    operations: Sequence[Operation],
    start: int,
    bindings: Mapping[ValueId, tuple[int, ...]],
    matrix: torch.Tensor,
    wires: tuple[int, ...],
    scratch: tuple[int, ...],
) -> tuple[tuple[Instruction, ...], int] | None:
    """Factor bounded 5-bit power groups using only registered gates."""
    first = operations[start]
    group: list[tuple[int, int]] = []
    for op in operations[start : start + 8]:
        control = op.attributes.get("control_index")
        if (
            op.name != "qreg.unitary"
            or control is None
            or op.operands[1:-1] != first.operands[1:-1]
            or op.attributes.get("indices") != first.attributes.get("indices")
            or op.attributes["matrix"] != first.attributes["matrix"]
        ):
            break
        group.append(
            (bindings[op.operands[-1].id][-1 - control], op.attributes.get("power", 1))
        )
    if len(group) < 2:
        return None

    _, eigenvectors = torch.linalg.eig(matrix)
    basis, _ = torch.linalg.qr(eigenvectors)
    angles = torch.angle((basis.mH @ matrix @ basis).diag())
    diagonal = torch.diag(torch.exp(1j * angles))
    if not torch.allclose(basis @ diagonal @ basis.mH, matrix, atol=5e-14, rtol=0):
        return None

    powered: dict[int, torch.Tensor] = {}
    for _, power in group:
        if power not in powered:
            diagonal = torch.diag(torch.exp(1j * (angles * power)))
            reference = torch.linalg.matrix_power(matrix, power)
            if not torch.allclose(
                basis @ diagonal @ basis.mH, reference, atol=1e-12, rtol=0
            ):
                return None
            powered[power] = diagonal

    gates = list(unitary_instructions(basis.mH, wires, first, scratch=scratch))
    for control, power in group:
        gates.extend(
            unitary_instructions(powered[power], wires, first, (control,), scratch)
        )
    gates.extend(unitary_instructions(basis, wires, first, scratch=scratch))
    return tuple(gates), start + len(group)
