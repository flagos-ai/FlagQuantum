# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# ModAddK value contract from Qualtran v0.7.0 mod_addition.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_addition.py

"""Bounded modular constant addition across the full binary word domain."""

from ....core.ir import Instruction
from ..arithmetic.addition import add_constant_ir
from ..arithmetic.controlled_addition import controlled_add_constant_ir
from ..bits.bitwise import _validate_wires
from ..comparison.less_than import less_than_constant_ir
from ..control.control_predicate import ControlRegister
from ..control.controlled_component import controlled_component_ir


def bounded_modular_add_requirements(
    width: int, modulus: int, control_count: int = 0
) -> tuple[int, int]:
    """Return clean comparator, predicate and shared arithmetic storage."""
    if (
        type(width) is not int
        or width < 0
        or type(modulus) is not int
        or not 1 <= modulus <= 1 << width
        or type(control_count) is not int
        or control_count < 0
    ):
        raise ValueError("Bounded modular addition needs a fitting positive modulus")
    if modulus == 1:
        return 0, 0
    if modulus == 1 << width:
        prefix = max(0, control_count - 1)
        return prefix + 2 * width - 1, max(prefix, width - 1)
    return 2 + control_count + max(width + 2, 2 * width - 1), max(control_count, width)


def bounded_modular_add_ir(
    target: tuple[int, ...],
    value: int,
    modulus: int,
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    controls: tuple[int, ...] = (),
    control_values: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Add a constant on admitted residues and preserve every padding state.

    Control and target words are big endian. The native realization composes
    reversible membership and wrap flags with the existing constant adder.
    ModAddK provides a permutation and a resource declaration upstream.
    """
    width = len(target)
    work, slots = bounded_modular_add_requirements(width, modulus, len(controls))
    if (
        type(value) is not int
        or len(workspace) != work
        or len(classical_bits) != slots
        or len(controls) != len(control_values)
        or any(type(bit) is not int or bit not in (0, 1) for bit in control_values)
    ):
        raise ValueError(
            "Bounded modular addition needs matching work, slots and controls"
        )
    if target or controls or workspace:
        _validate_wires((*target, *controls, *workspace))
    if classical_bits:
        _validate_wires(classical_bits)
    value %= modulus
    if value == 0:
        return ()
    full_ring = modulus == 1 << width
    available = workspace if full_ring else workspace[1:]
    membership: tuple[Instruction, ...] = ()
    if not full_ring:
        valid = workspace[0]
        membership = less_than_constant_ir(
            target, modulus, valid, available[: width + 1], classical_bits[:width]
        )
        controls = (*controls, valid)
        control_values = (*control_values, 1)
    prefix = max(0, len(controls) - 1)
    child = available[prefix:]
    if not controls:
        return add_constant_ir(
            target, value, child[:width], child[width:], classical_bits[: width - 1]
        )

    def emit(enable: int) -> tuple[Instruction, ...]:
        scratch = child if full_ring else child[1:]
        constant = scratch[:width]
        carry = scratch[width : 2 * width - 1]
        slots = classical_bits[: width - 1]
        addition = controlled_add_constant_ir(
            enable, target, value, constant, carry, slots
        )
        if full_ring:
            return addition
        wrap, indicator = child[0], scratch[0]

        def toggle_wrap(threshold: int, *, complement: bool) -> tuple[Instruction, ...]:
            compare = less_than_constant_ir(
                target,
                threshold,
                indicator,
                scratch[1 : width + 2],
                classical_bits[:width],
            )
            flip = (Instruction("x", (indicator,)),) if complement else ()
            return (
                *compare,
                *flip,
                Instruction("ccx", (enable, indicator, wrap)),
                *flip,
                *compare,
            )

        subtract = controlled_add_constant_ir(
            wrap, target, -modulus, constant, carry, slots
        )
        return (
            *toggle_wrap(modulus - value, complement=True),
            *addition,
            *subtract,
            *toggle_wrap(value, complement=False),
        )

    update = controlled_component_ir(
        (controls,),
        (ControlRegister(1, control_values),),
        emit,
        available[:prefix],
        classical_bits[:prefix],
    )
    return (*membership, *update, *membership)
