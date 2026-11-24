# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 block_encoding/sparse_matrix_hermitian.py.

"""Compose Hermitian sparse access with unconditional preparation and cleanup."""

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ..component_adjoint import adjoint_component_calls
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..layout.component_partition import partition_component_program
from ..primitives.always_component import always_component_program
from ..primitives.controlled_swap_component import controlled_swap_program
from ..primitives.swap_component import swap_word_program
from .sparse_matrix_component import sparse_diffusion_pair


def hermitian_sparse_matrix_signature(
    width: SymbolicInt, *, controlled: bool = False
) -> Signature:
    """Declare caller-owned system and MSB a/location/b ancilla words."""
    ports = {"ctrl": quantum_bit_type()} if controlled else {}
    return Signature.build_from_dtypes(
        **ports,
        system=opaque_register_type(width),
        ancilla=opaque_register_type(width + 2),
    )


def hermitian_sparse_matrix_component_program(
    width: SymbolicInt,
    sparsity: SymbolicInt,
    column_pair: tuple[ProgramRecord, ProgramRecord],
    entry_pair: tuple[ProgramRecord, ProgramRecord],
    *,
    controlled: bool = False,
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain the source Partition, six Always calls and two intervening swaps.

    Children supply complete closures and exact inverse Programs. Their private
    work remains their responsibility. The source adjoint method returns this
    same construction; a generic reversed graph is a separate operation.
    """
    if type(controlled) is not bool or type(adjoint) is not bool:
        raise ValueError("Hermitian sparse control and adjoint flags must be boolean")
    signature = hermitian_sparse_matrix_signature(width, controlled=controlled)
    if type(width) is not int or type(sparsity) is not int:
        return ProgramRecord((FunctionRecord("main", (), registers=tuple(signature)),))
    parts = (
        Register("a", quantum_bit_type()),
        Register("l", opaque_register_type(width)),
        Register("b", quantum_bit_type()),
    )
    partition = partition_component_program(parts, "partition_ancilla", False)
    unpartition = partition_component_program(parts, "join_ancilla", True)
    diffusion_pair = sparse_diffusion_pair(width, sparsity)

    def wrap(pair: tuple[ProgramRecord, ProgramRecord]) -> tuple[ProgramRecord, ...]:
        return tuple(
            always_component_program(child, child.functions[0].name) for child in pair
        )

    diffusion, undiffusion = wrap(diffusion_pair)
    column, uncolumn = wrap(column_pair)
    entry, unentry = wrap(entry_pair)
    builder = ComponentProgramBuilder(signature)

    def call(
        child: ProgramRecord, **values: BindingCollection
    ) -> dict[str, BindingCollection]:
        return builder.call_program(child, values, entry=child.functions[0].name)

    values = call(partition, x=builder.inputs["ancilla"])
    a, location, b = values["a"], values["l"], values["b"]
    location = call(diffusion, target=location)["target"]
    values = call(column, l=location, i=builder.inputs["system"])
    values = call(entry, q=b, i=values["l"], j=values["i"])
    b = values["q"]
    control = {"ctrl": builder.inputs["ctrl"]} if controlled else {}
    swap = controlled_swap_program if controlled else swap_word_program
    word_swap, flag_swap = swap(width), swap(1)
    values = call(word_swap, **control, x=values["i"], y=values["j"])
    if controlled:
        control = {"ctrl": values["ctrl"]}
    location, system = values["x"], values["y"]
    flags = call(flag_swap, **control, x=a, y=b)
    a, b = flags["x"], flags["y"]
    values = call(unentry, q=b, i=location, j=system)
    b = values["q"]
    values = call(uncolumn, l=values["i"], i=values["j"])
    location = call(undiffusion, target=values["l"])["target"]
    ancilla = call(unpartition, a=a, l=location, b=b)["x"]
    outputs = {"system": values["i"], "ancilla": ancilla}
    if controlled:
        outputs["ctrl"] = flags["ctrl"]
    record = builder.finalize(outputs)
    return (
        adjoint_component_calls(
            record,
            (
                (partition, unpartition),
                (unpartition, partition),
                (diffusion, undiffusion),
                (undiffusion, diffusion),
                (column, uncolumn),
                (uncolumn, column),
                (entry, unentry),
                (unentry, entry),
                (word_swap, word_swap),
                (flag_swap, flag_swap),
            ),
        )
        if adjoint
        else record
    )
