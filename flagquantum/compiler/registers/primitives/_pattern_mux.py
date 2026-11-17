# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix CircuitPattern operand dispatch.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/build/circuit_pattern.cc

"""Resolve typed source patterns before binding physical lane operands."""

from typing import Literal

from ...artifacts.reversible_pattern import LaneColumn, bind_reversible_pattern
from ...artifacts.reversible_phase import ReversiblePhaseOperation as Operation
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ._mixed_gates import _Control

_Z_PHASE_OPCODE = {"cx": "x", "cz": "z", "z": "neg"}
_ENABLE_DOWNGRADE = {"cx": "x", "ccx": "cx", "ccz": "cz", "cswap": "swap", "z": "neg"}
_ENABLE_ARITY = {"cx": 1, "ccx": 2, "ccz": 2, "cswap": 2, "z": 0}


def bind_z_control_pattern(
    opcode: Literal["cx", "cz", "z"],
    controls: tuple[_Control, ...],
    lanes: range,
    *,
    target: LaneColumn | None = None,
) -> tuple[Operation, ...]:
    """Bind the source homogeneous quantum or classical control column."""
    if (target is None) != (opcode == "z"):
        raise ValueError("Pattern target column does not match the opcode")
    if not lanes:
        return ()
    columns: tuple[LaneColumn, ...]
    quantum = tuple(control for control in controls if type(control) is int)
    if len(quantum) == len(controls):
        columns = (quantum,) if target is None else (quantum, target)
        operation = Operation(opcode, tuple(range(len(columns))))
        return bind_reversible_pattern((operation,), lanes, qubits=columns)

    classical = tuple(
        control.slot
        for control in controls
        if isinstance(control, ClassicalFieldCoefficient)
    )
    if len(classical) != len(controls):
        raise ValueError(
            "Pattern controls must be homogeneous quantum or classical lanes"
        )
    columns = () if target is None else (target,)
    operation = Operation(
        _Z_PHASE_OPCODE[opcode], tuple(range(len(columns))), condition_bit=0
    )
    return bind_reversible_pattern(
        (operation,), lanes, qubits=columns, bits=(classical,)
    )


def bind_quantum_enable_pattern(
    opcode: Literal["cx", "ccx", "ccz", "cswap", "z"],
    enable: int | Literal[True],
    operands: tuple[LaneColumn, ...],
    lanes: range,
    *,
    condition: LaneColumn | None = None,
) -> tuple[Operation, ...]:
    """Bind an optional quantum enable, retaining each source operand order."""
    if len(operands) != _ENABLE_ARITY[opcode]:
        raise ValueError("Pattern operand columns do not match the opcode")
    if enable is True:
        operation = Operation(
            _ENABLE_DOWNGRADE[opcode],
            tuple(range(len(operands))),
            condition_bit=0 if condition is not None else None,
        )
        columns = operands
    else:
        if type(enable) is not int:
            raise ValueError("Pattern enable must be a quantum wire or true")
        columns = (enable, *operands)
        order = (1, 0, 2) if opcode in ("ccx", "ccz") else tuple(range(len(columns)))
        operation = Operation(
            opcode, order, condition_bit=0 if condition is not None else None
        )
    return bind_reversible_pattern(
        (operation,),
        lanes,
        qubits=columns,
        bits=() if condition is None else (condition,),
    )
