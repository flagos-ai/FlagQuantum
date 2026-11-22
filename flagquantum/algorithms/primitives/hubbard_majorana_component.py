# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 select_hubbard.py.

"""Retain Hubbard Majorana's inlined shared-register decomposition."""

from ...compiler.registers.component_builder import ComponentProgramBuilder
from ...compiler.registers.component_expansion import expand_component_calls_once
from ...compiler.registers.control.unary_wire_calls import adjoint_unary_wire_program
from ...compiler.registers.lookup.majorana_component import majorana_program
from ...core.ir.program import ProgramRecord
from ...core.ir.program.register_types import (
    bounded_unsigned_register_type,
    quantum_bit_type,
)
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from .hubbard_majorana import HubbardMajorana


def hubbard_majorana_program(
    operation: HubbardMajorana, *, adjoint: bool = False
) -> ProgramRecord:
    """Store successful source bodies with MSB-first x, y, spin selection.

    The source inlines its shared Majorana child in connection visitation order.
    Private parity and predicate work remain owned by that child. Original
    singleton and zero-control stored decomposition failures stay explicit.
    """
    if type(adjoint) is not bool:
        raise ValueError("Stored Hubbard direction must be boolean")
    if operation.dimension == 1:
        raise ValueError(
            "Original Hubbard Majorana stored decomposition refuses singleton coordinates"
        )
    if operation.controlled and operation.control_value == 0:
        raise ValueError(
            "Original Hubbard Majorana stored decomposition cannot bind zero control"
        )
    width = (operation.dimension - 1).bit_length()
    selection = tuple(
        Register(name, bounded_unsigned_register_type(bits, bound))
        for name, bits, bound in (
            ("x", width, operation.dimension),
            ("y", width, operation.dimension),
            ("spin", 1, 2),
        )
    )
    child = majorana_program(
        selection,
        operation.pauli,
        control=(
            Register("control", quantum_bit_type()) if operation.controlled else None
        ),
        adjoint=False,
    )
    builder = ComponentProgramBuilder(Signature(child.functions[0].registers or ()))
    root = builder.finalize(builder.call_program(child, builder.inputs))
    root = expand_component_calls_once(root)[0]
    return adjoint_unary_wire_program(root) if adjoint else root
