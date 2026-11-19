# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _ControlledBase.as_cirq_op.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/controlled.py

"""Place typed quantum controls around an exported native register operation."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...core.ir.program.registers import Register
from ._control_values import export_control_values
from ._register_export import _register_shape
from .conversion import _cirq


def control_register_operation(
    child: Any,
    child_outputs: Mapping[str, NDArray[np.object_]],
    controls: tuple[Register, ...],
    values: tuple[NDArray[Any], ...],
    quregs: Mapping[str, NDArray[np.object_]],
) -> tuple[Any, dict[str, NDArray[np.object_]]]:
    """Control an exported child while retaining its output leases and SDK labels.

    The caller exports the child through its existing owner first. Control ports
    are disjoint quantum registers in declared order, with a trailing bit axis.
    Returned control arrays retain identity. Cirq controls the child operation
    directly and therefore retains its unitary, decomposition and diagram
    protocols, including the child's exponent. This adapter adds no workspace.
    """
    cirq = _cirq()
    if not isinstance(child, cirq.Operation):
        raise ValueError("controlled register export requires a child operation")
    names = tuple(register.name for register in controls)
    if (
        len(set(names)) != len(names)
        or set(quregs) != set(names)
        or set(names) & child_outputs.keys()
    ):
        raise ValueError(
            "control names must be distinct and separate from child outputs"
        )
    operands: list[Any] = []
    for register in controls:
        array = quregs[register.name]
        if array.shape != _register_shape(register):
            raise ValueError("control qubit shape differs from its register")
        operands.extend(array.flat)
    if (
        any(not isinstance(q, cirq.Qid) or q.dimension != 2 for q in operands)
        or len(set(operands)) != len(operands)
        or set(operands) & set(child.qubits)
    ):
        raise ValueError(
            "control operands must be distinct binary qubits outside the child"
        )
    clause = export_control_values(controls, values)
    operation = child.controlled_by(*operands, control_values=clause)
    return operation, dict(child_outputs) | dict(quregs)
