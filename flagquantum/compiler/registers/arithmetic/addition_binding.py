# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix Python iadd/isub binding.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/circuit/subs/iadd.pybind.cc

"""Dispatch typed in-place arithmetic to the existing physical generators."""

from dataclasses import replace
from math import inf, isfinite, isnan

from ..._component_trace.trace import ComponentCall
from ...artifacts.reversible_phase import ReversiblePhaseProgram
from ...lowering.physical_trace import scope_physical_trace
from ..finite_field.coefficients import ClassicalFieldCoefficient
from ..layout.operand_word import (
    bind_operand_word_or_integer,
    bind_quantum_word,
    bind_z_word,
)
from ..primitives._mixed_gates import _Control
from ..primitives.control_binding import bind_z_enable
from ..primitives.operand_binding import bind_operand
from .physical_addition import physical_mixed_addition
from .physical_inplace_addition import physical_inplace_addition


def physical_addition_binding(
    offset: object,
    *,
    target: object,
    control: _Control = True,
    workspace: tuple[int, ...] = (),
    btol: float = inf,
    subtract: bool = False,
) -> tuple[ReversiblePhaseProgram, tuple[ComponentCall, ...]]:
    """Bind MSB-first target and offset words with caller-owned clean leases.

    Workspace is in allocation order, including the source pool's LIFO order.
    Only the first target-width leases enter the generator. All workspace starts
    and ends at zero. Empty targets return before offset/control conversion.
    Classical enables wrap the complete call and its measurement feedback.
    """
    word = bind_quantum_word(target)
    last_wire = max((*word, *workspace), default=0)
    if not word:
        return ReversiblePhaseProgram(last_wire + 1, 0, ()), ()
    coefficients = bind_operand_word_or_integer(offset, len(word), allow_negative=True)
    enable, guards = bind_z_enable(control)
    slots = {
        item.slot
        for item in (*coefficients, bind_operand(control))
        if isinstance(item, ClassicalFieldCoefficient)
    }
    bits = max(slots, default=-1) + 1
    ports = (*word, *coefficients, *workspace, enable)
    last_wire = max((item for item in ports if type(item) is int), default=0)
    if enable is False:
        return ReversiblePhaseProgram(last_wire + 1, bits, (), tuple(sorted(slots))), ()
    clean = workspace[: len(word)]
    if all(type(item) is not int for item in coefficients):
        program, calls = physical_mixed_addition(
            word,
            bind_z_word(coefficients),
            clean=clean,
            dirty=(),
            control=enable,
            classical_width=bits,
            # Source comparisons with NaN select the full exact schedule.
            btol=inf if isnan(btol) else btol,
            subtract=subtract,
        )
    else:
        if isfinite(btol):
            raise ValueError("btol is only supported for classical addition")
        quantum = bind_quantum_word(coefficients)
        program, calls = physical_inplace_addition(
            word,
            quantum,
            clean=clean,
            control=enable,
            measurement_bit=bits,
            subtract=subtract,
        )
    program = replace(
        program,
        n_qubits=max(last_wire + 1, program.n_qubits),
        n_bits=max(bits, program.n_bits),
        input_bits=tuple(sorted(set(program.input_bits).union(slots))),
    )
    return scope_physical_trace(program, calls, guards)
