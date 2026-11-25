# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 select_hubbard.py.

"""Retain Hubbard Majorana's inlined shared-register decomposition."""

from typing import Literal

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import (
    bounded_unsigned_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_expansion import expand_component_calls_once
from ..control.unary_wire_calls import adjoint_unary_wire_program
from ..lookup.majorana_component import majorana_program


def hubbard_majorana_program(
    dimension: int,
    pauli: Literal["X", "Y"] = "Y",
    *,
    controlled: bool = False,
    adjoint: bool = False,
) -> ProgramRecord:
    """Store successful source bodies with MSB-first x, y, spin selection.

    The source inlines its shared Majorana child in connection visitation order.
    Private parity and predicate work remain owned by that child. Original
    singleton and zero-control stored decomposition failures stay explicit.
    """
    if type(adjoint) is not bool:
        raise ValueError("Stored Hubbard direction must be boolean")
    if dimension == 1:
        raise ValueError(
            "Original Hubbard Majorana stored decomposition refuses singleton coordinates"
        )
    width = (dimension - 1).bit_length()
    selection = tuple(
        Register(name, bounded_unsigned_register_type(bits, bound))
        for name, bits, bound in (
            ("x", width, dimension),
            ("y", width, dimension),
            ("spin", 1, 2),
        )
    )
    child = majorana_program(
        selection,
        pauli,
        control=(Register("control", quantum_bit_type()) if controlled else None),
        adjoint=False,
    )
    builder = ComponentProgramBuilder(Signature(child.functions[0].registers or ()))
    root = builder.finalize(builder.call_program(child, builder.inputs))
    root = expand_component_calls_once(root)[0]
    return adjoint_unary_wire_program(root) if adjoint else root
