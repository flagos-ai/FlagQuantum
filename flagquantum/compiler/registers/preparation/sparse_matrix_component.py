# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Calls follow locked Qualtran v0.7.0 block_encoding/sparse_matrix.py.

"""Compose literal sparse-access matrix calls and exact diffusion inverses."""

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import opaque_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ..component_adjoint import adjoint_component_calls
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..layout.auto_partition import AutoPartition
from ..layout.component_auto_partition import auto_partition_component_pair
from ..layout.packed_block_component import packed_block_signature
from ..primitives.swap_component import swap_word_program
from .uniform_preparation_component import uniform_preparation_program


def sparse_matrix_signature(width: SymbolicInt) -> Signature:
    """Keep the opaque system then MSB flag/location ancilla word."""
    return packed_block_signature(width, width + 1, 0)


def sparse_diffusion_pair(
    width: int, sparsity: int
) -> tuple[ProgramRecord, ProgramRecord]:
    """Keep the leading Unused slice, including its original zero-width case."""
    forward = uniform_preparation_program(sparsity)
    inverse = uniform_preparation_program(sparsity, adjoint=True)
    used = (sparsity - 1).bit_length()
    result, reverse = auto_partition_component_pair(
        forward,
        (Register("target", opaque_register_type(width)),),
        AutoPartition((("target", ((None, width - used), ("target", used))),)),
        inverse,
    )
    assert reverse is not None
    return result, reverse


def sparse_matrix_component_program(
    width: SymbolicInt,
    sparsity: SymbolicInt,
    row_pair: tuple[ProgramRecord, ProgramRecord],
    column_pair: tuple[ProgramRecord, ProgramRecord],
    entry_pair: tuple[ProgramRecord, ProgramRecord],
    *,
    adjoint: bool = False,
) -> ProgramRecord:
    """Call diffusion/column/entry/Swap/row-inverse/diffusion-inverse in order.

    System and packed flag/location inputs are caller-owned. Supplied children
    own complete helper closures, exact inverse semantics and private cleanup.
    Symbolic roots remain declarations, preserving the source physical refusal.
    """
    if type(adjoint) is not bool:
        raise ValueError("Sparse matrix direction must be boolean")
    signature = sparse_matrix_signature(width)
    if type(width) is not int or type(sparsity) is not int:
        return ProgramRecord((FunctionRecord("main", (), registers=tuple(signature)),))
    diffusion, undiffusion = sparse_diffusion_pair(width, sparsity)
    swap = swap_word_program(width)
    row, unrow = row_pair
    column, uncolumn = column_pair
    entry, unentry = entry_pair
    builder = ComponentProgramBuilder(signature)
    ancilla = builder.inputs["ancilla"]
    assert isinstance(ancilla, RegisterBinding)
    bits = builder.split(ancilla)
    flag = bits[0]
    location: BindingCollection = builder.join(bits[1:])
    location = builder.call_program(diffusion, {"target": location})["target"]
    values = builder.call_program(
        column,
        {"l": location, "i": builder.inputs["system"]},
        entry=column.functions[0].name,
    )
    values = builder.call_program(
        entry,
        {"q": flag, "i": values["l"], "j": values["i"]},
        entry=entry.functions[0].name,
    )
    flag = values["q"]
    values = builder.call_program(swap, {"x": values["i"], "y": values["j"]})
    values = builder.call_program(
        unrow, {"l": values["x"], "i": values["y"]}, entry=unrow.functions[0].name
    )
    location = builder.call_program(undiffusion, {"target": values["l"]})["target"]
    assert isinstance(location, RegisterBinding)
    record = builder.finalize(
        {
            "system": values["i"],
            "ancilla": builder.join((flag, *builder.split(location))),
        }
    )
    return (
        adjoint_component_calls(
            record,
            (
                (diffusion, undiffusion),
                (undiffusion, diffusion),
                (column, uncolumn),
                (entry, unentry),
                (swap, swap),
                (unrow, row),
            ),
        )
        if adjoint
        else record
    )
