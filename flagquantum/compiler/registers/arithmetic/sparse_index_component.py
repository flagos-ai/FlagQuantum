# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Calls follow locked Qualtran v0.7.0 block_encoding/sparse_matrix.py.

"""Retain sided sparse row/column index maps and exact arithmetic inverses."""

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import (
    bounded_unsigned_register_type,
    signed_register_type,
    unsigned_register_type,
)
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from .addition_component import addition_component_program
from .constant_addition_component import constant_addition_program


def row_column_signature(width: SymbolicInt, sparsity: SymbolicInt) -> Signature:
    """Keep source LEFT bounded ordinal, RIGHT location and THRU index words."""
    return Signature(
        (
            Register(
                "l",
                bounded_unsigned_register_type(width, sparsity),
                side=RegisterSide.LEFT,
            ),
            Register("l", unsigned_register_type(width), side=RegisterSide.RIGHT),
            Register("i", unsigned_register_type(width)),
        )
    )


def top_left_index_program(
    width: SymbolicInt, sparsity: SymbolicInt, *, adjoint: bool = False
) -> ProgramRecord:
    """Return the ordinal unchanged, preserving source sided type metadata."""
    if type(adjoint) is not bool:
        raise ValueError("Sparse index direction must be boolean")
    signature = row_column_signature(width, sparsity)
    builder = ComponentProgramBuilder(signature.adjoint() if adjoint else signature)
    return builder.finalize(builder.inputs)


def banded_index_program(
    width: SymbolicInt, bandsize: SymbolicInt, *, adjoint: bool = False
) -> ProgramRecord:
    """Call source unsigned Add then signed negative AddK, with explicit inverses."""
    if type(adjoint) is not bool:
        raise ValueError("Sparse index direction must be boolean")
    signature = row_column_signature(width, 2 * bandsize + 1)
    if type(width) is not int or type(bandsize) is not int or width < 1:
        return ProgramRecord(
            (
                FunctionRecord(
                    "main",
                    (),
                    registers=tuple(signature.adjoint() if adjoint else signature),
                ),
            )
        )
    dtype = unsigned_register_type(width)
    signed = signed_register_type(width)
    add = addition_component_program(width, left_dtype=dtype, right_dtype=dtype)
    shift = constant_addition_program(signed, (-bandsize) % (1 << width))
    builder = ComponentProgramBuilder(signature)
    values = builder.call_program(
        add, {"a": builder.inputs["i"], "b": builder.inputs["l"]}
    )
    location = builder.call_program(shift, {"x": values["b"]})["x"]
    record = builder.finalize({"l": location, "i": values["a"]})
    return (
        adjoint_component_calls(
            record,
            (
                (
                    add,
                    addition_component_program(
                        width, left_dtype=dtype, right_dtype=dtype, adjoint=True
                    ),
                ),
                (
                    shift,
                    constant_addition_program(
                        signed, (-bandsize) % (1 << width), adjoint=True
                    ),
                ),
            ),
        )
        if adjoint
        else record
    )
