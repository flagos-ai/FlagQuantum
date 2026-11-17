# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix gen_cmp.cc and gen_cmp_qq.cc.
# https://github.com/quantumlib/Qualtran/tree/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/comparators

"""Export comparator operations with source resets, call marks and feedback."""

from dataclasses import replace
from math import ceil, inf
from typing import Literal

from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ...lowering.physical_trace import PhysicalTrace, lower_physical_trace
from ..finite_field.coefficients import ClassicalFieldCoefficient
from .low_space_comparison import _comparison_passes, low_space_less_than_ir
from .mixed_comparison import (
    mixed_greater_than_ir,
    mixed_less_than_ir,
    quantum_compare_ir,
    quantum_equal_ir,
)

_Coefficient = int | bool | ClassicalFieldCoefficient
_Relation = Literal["lt", "le", "gt", "ge", "eq"]


def physical_comparison(
    left: tuple[int, ...],
    right: tuple[_Coefficient, ...],
    target: int | None,
    workspace: tuple[int, ...],
    *,
    relation: _Relation = "lt",
    or_equal: _Coefficient = False,
    control: int | bool = True,
    btol: float = inf,
    minimize_qubits: bool = False,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Bind big-endian operands and allocate measured bits after classical inputs."""
    if relation not in ("lt", "le", "gt", "ge", "eq"):
        raise ValueError("Unknown comparison relation")
    if relation in ("le", "ge"):
        or_equal = True
    elif relation == "eq":
        or_equal = False
    operands = (*right, or_equal)
    inputs = {q.slot for q in operands if isinstance(q, ClassicalFieldCoefficient)}
    result = max(inputs, default=-1) + 1
    trace = PhysicalTrace(implicit_zero_bits=inputs)
    ports = (*left, *workspace, *operands, target, control)
    last_wire = max((q for q in ports if type(q) is int), default=0)
    if relation == "eq":
        gates = quantum_equal_ir(
            left,
            right,
            target,
            workspace,
            control=control,
            measurement_bit=result,
            emit_resets=True,
        )
    else:
        compare = (
            mixed_greater_than_ir if relation in ("gt", "ge") else mixed_less_than_ir
        )
        depth = len(_comparison_passes(len(left) + int(type(control) is int)))
        low_space_bits = tuple(range(result, result + depth + 1 + int(depth >= 4)))
        gates = compare(
            left,
            right,
            target,
            workspace,
            or_equal=or_equal,
            control=control,
            measurement_bit=result,
            minimize_qubits=minimize_qubits,
            btol=btol,
            low_space_bits=low_space_bits,
            emit_resets=True,
            export=trace,
        )
        if relation in ("lt", "le"):
            call = ComponentCall(
                "flip_if_lt", 0, len(gates), children=tuple(trace.calls)
            )
            trace.calls = [
                _bind_less_than_dispatch(call, right, workspace, btol, minimize_qubits)
            ]
        elif trace.calls:
            (outer,) = trace.calls
            (inner,) = outer.children
            trace.calls = [
                replace(
                    outer,
                    children=(
                        _bind_less_than_dispatch(
                            inner, right, workspace, btol, minimize_qubits
                        ),
                    ),
                )
            ]

    return lower_physical_trace(gates, 0, last_wire, trace, reorder_feedback=False)


def _bind_less_than_dispatch(
    call: ComponentCall,
    right: tuple[_Coefficient, ...],
    workspace: tuple[int, ...],
    btol: float,
    minimize_qubits: bool,
) -> ComponentCall:
    width = len(right)
    if btol < width and btol <= 1:
        return call
    truncated = btol < width
    width = ceil(btol) - 1 if truncated else width
    children = call.children
    if (
        (btol == inf or truncated)
        and len(workspace) < width
        and all(type(q) is int for q in right)
    ):
        children = (replace(call, name="flip_if_lt_qq"),)
    elif (
        width > 1
        and (minimize_qubits or len(workspace) < width - 1)
        and all(type(q) is not int for q in right[:width])
    ):
        children = (replace(call, name="xif_less_than_3anc_2ntof"),)
    if truncated:
        children = (replace(call, children=children),)
    return replace(call, children=children)


def physical_less_than(
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: int,
    workspace: tuple[int, ...],
    *,
    control: int | bool,
    btol: float,
    minimize_qubits: bool,
) -> tuple[ReversiblePhaseProgram, ComponentCall]:
    """Export the quantum less-than command through the shared comparator."""
    program, (call,) = physical_comparison(
        left,
        right,
        target,
        workspace,
        control=control,
        btol=btol,
        minimize_qubits=minimize_qubits,
    )
    return program, call


def physical_low_space_less_than(
    left: tuple[int, ...],
    right: tuple[bool | ClassicalFieldCoefficient, ...],
    target: int | None,
    workspace: tuple[int, ...],
    *,
    or_equal: _Coefficient = False,
    inverted: _Coefficient = False,
    control: int | bool = True,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Export the three-clean-wire source entry and its prefix correction."""
    ports = (*left, *right, target, *workspace, or_equal, inverted, control)
    inputs = {q.slot for q in ports if isinstance(q, ClassicalFieldCoefficient)}
    last_wire = max((q for q in ports if type(q) is int), default=0)
    result = max(inputs, default=-1) + 1
    depth = len(_comparison_passes(len(left) + int(type(control) is int)))
    bits = tuple(range(result, result + depth + 1 + int(depth >= 4)))
    trace = PhysicalTrace(implicit_zero_bits=inputs)
    gates = low_space_less_than_ir(
        left,
        right,
        target,
        workspace,
        measurement_bits=bits,
        or_equal=or_equal,
        inverted=inverted,
        control=control,
        export=trace,
    )
    trace.calls = [
        ComponentCall(
            "xif_less_than_3anc_2ntof", 0, len(gates), children=tuple(trace.calls)
        )
    ]
    return lower_physical_trace(gates, 0, last_wire, trace, reorder_feedback=False)


def physical_quantum_less_than(
    left: tuple[int, ...],
    right: tuple[int, ...],
    target: int | None,
    workspace: tuple[int, ...] = (),
    *,
    or_equal: _Coefficient = False,
    control: int | bool = True,
    minimize_qubits: bool = False,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Export the dedicated quantum schedule even when a full ladder fits."""
    ports = (*left, *right, target, *workspace, or_equal, control)
    inputs = (
        {or_equal.slot} if isinstance(or_equal, ClassicalFieldCoefficient) else set()
    )
    last_wire = max((q for q in ports if type(q) is int), default=0)
    gates = quantum_compare_ir(
        left,
        right,
        target,
        workspace,
        or_equal=or_equal,
        control=control,
        measurement_bit=max(inputs, default=-1) + 1,
        minimize_qubits=minimize_qubits,
        emit_resets=True,
    )
    trace = PhysicalTrace(
        calls=[ComponentCall("flip_if_lt_qq", 0, len(gates))],
        implicit_zero_bits=inputs,
    )
    return lower_physical_trace(gates, 0, last_wire, trace, reorder_feedback=False)
