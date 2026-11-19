"""Copy materialized SDK channels across the native operation boundary."""

from typing import Any

import numpy as np
import torch

from ...core.ir import Instruction
from ...core.ir.program.kraus import kraus_error
from .conversion import _cirq


def _channel_instruction(gate: Any, operands: tuple[int, ...]) -> Instruction | None:
    cirq = _cirq()
    if cirq.has_unitary(gate):
        return None
    operators = cirq.kraus(gate, default=None)
    if operators is None:
        return None
    copied = tuple(
        np.array(operator, dtype=np.complex128, copy=True) for operator in operators
    )
    if any(operator.ndim != 2 for operator in copied):
        raise ValueError("Channel operators require rank-two matrices")
    immutable = tuple(
        tuple(tuple(complex(value) for value in row) for row in operator)
        for operator in copied
    )
    error = kraus_error(immutable, len(operands))
    if error is not None:
        raise ValueError(error)
    return Instruction(
        "kraus",
        operands,
        matrix=tuple(torch.from_numpy(operator) for operator in copied),
        metadata={"is_channel": True},
    )


def _channel_operation(instruction: Instruction, qubits: Any) -> Any:
    """Export copied operators without retaining SDK objects in the native IR."""
    if instruction.matrix is None:
        raise ValueError("Channel export requires materialized Kraus operators")
    operators = tuple(
        matrix.detach().cpu().numpy().copy() for matrix in instruction.matrix
    )
    return (
        _cirq()
        .KrausChannel(operators)
        .on(*(qubits[wire] for wire in instruction.wires))
    )
