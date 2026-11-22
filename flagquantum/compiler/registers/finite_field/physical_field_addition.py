# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix gen_gf_iadd and binding control helper.
# https://github.com/quantumlib/Qualtran/tree/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix

"""Physical field addition with the source's call and classical guard layout."""

from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ...lowering.physical_trace import (
    PhysicalTrace,
    lower_physical_trace,
    scope_physical_trace,
)
from .coefficients import ClassicalFieldCoefficient, gf2_add_coefficients_ir


def physical_field_addition(
    source: tuple[int | bool | ClassicalFieldCoefficient, ...],
    target: tuple[int, ...],
    *,
    control: int | bool | ClassicalFieldCoefficient | None = None,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Bind big-endian coefficient views and retain the source's outer scope.

    The operands and classical slots are caller-owned. The generator leases
    no workspace. Explicit views can reverse or stride physical wire IDs.
    """
    guard = control.slot if isinstance(control, ClassicalFieldCoefficient) else None
    gates = gf2_add_coefficients_ir(
        source, target, control=control if guard is None else None
    )
    ports = (*source, *target, control)
    inputs = {
        port.slot for port in ports if isinstance(port, ClassicalFieldCoefficient)
    }
    last_wire = max((port for port in ports if type(port) is int), default=0)
    name = "c_gf_iadd" if type(control) is int else "gf_iadd"
    trace = PhysicalTrace(implicit_zero_bits=inputs)
    if target and control is not False:
        trace.calls.append(ComponentCall(name, 0, len(gates)))
    body, calls = lower_physical_trace(
        gates, 0, last_wire, trace, reorder_feedback=False
    )
    # Preserve a repeated coefficient predicate inside the binding guard.
    return scope_physical_trace(body, calls, () if guard is None else (guard,))
