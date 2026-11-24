# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Call order follows Qualtran v0.7.0 SparseStatePreparationViaRotations.

"""Compose source partition, normalized dense preparation and ordered cycles."""

import numpy as np

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_types import (
    opaque_register_type,
    unsigned_register_type,
)
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ..component_adjoint import adjoint_component_calls
from ..component_builder import ComponentProgramBuilder
from ..layout.component_partition import partition_component_program
from ..phase.rotation_table_words import rotation_table_words
from ..sorting.permutation_component import permutation_program
from ..sorting.permutation_declaration import PermutationDeclaration
from .rotation_preparation_component import rotation_preparation_program


def sparse_rotation_preparation_program(
    indices: tuple[int, ...],
    coefficients: tuple[complex, ...],
    size: int,
    phase_width: SymbolicInt,
    target_width: int,
    *,
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain source padding, normalization and partial-map cycle discovery.

    The unsigned target and opaque gradient remain caller-owned and MSB-first.
    Exact inverses use existing component pairs, not an optimized alternative
    permutation or a reversal of measured gates. Symbolic phase width retains
    the original direct graph with an unexecutable dense child declaration.
    """
    if type(adjoint) is not bool:
        raise ValueError("Sparse rotation direction must be boolean")
    if not indices or len(indices) != len(coefficients):
        raise ValueError(
            "Sparse rotation requires matching nonempty support and coefficients"
        )
    low_width = (len(indices) - 1).bit_length()
    signature = Signature.build_from_dtypes(
        target_state=unsigned_register_type(target_width),
        phase_gradient=opaque_register_type(phase_width),
    )
    if low_width == 0:
        return ProgramRecord((FunctionRecord("main", (), registers=tuple(signature)),))
    parts = (
        Register("high_bits", opaque_register_type(target_width - low_width)),
        Register("low_bits", unsigned_register_type(low_width)),
    )
    partition = partition_component_program(parts, "main", False)
    unpartition = partition_component_program(parts, "main", True)
    # This is the original compile-time normalization, not the wider stable
    # extreme-value fallback of the existing Algorithms flat emitter.
    padded = np.pad(coefficients, (0, (1 << low_width) - len(coefficients)))
    normalized = tuple(complex(value) for value in (padded / np.linalg.norm(padded)))
    if type(phase_width) is int:
        rows, phases = rotation_table_words(normalized, phase_width)
        dense = rotation_preparation_program(rows, phases, phase_width)
        dense_inverse = (
            rotation_preparation_program(rows, phases, phase_width, adjoint=True)
            if adjoint
            else dense
        )
    else:
        dense = ProgramRecord(
            (
                FunctionRecord(
                    "main",
                    (),
                    registers=tuple(
                        Signature.build(
                            target_state=low_width, phase_gradient=phase_width
                        )
                    ),
                ),
            )
        )
        dense_inverse = dense
    declaration = PermutationDeclaration.from_partial(size, dict(enumerate(indices)))
    assert isinstance(declaration.cycles, tuple)
    cycles = tuple(cycle for cycle in declaration.cycles if isinstance(cycle, tuple))
    permutation = permutation_program(size, cycles, target_width)
    builder = ComponentProgramBuilder(signature)
    split = builder.call_program(
        partition,
        {"x": builder.inputs["target_state"]},
        entry=partition.functions[0].name,
    )
    prepared = builder.call_program(
        dense,
        {
            "target_state": split["low_bits"],
            "phase_gradient": builder.inputs["phase_gradient"],
        },
    )
    joined = builder.call_program(
        unpartition,
        {"high_bits": split["high_bits"], "low_bits": prepared["target_state"]},
        entry=unpartition.functions[0].name,
    )
    result = builder.call_program(permutation, {"x": joined["x"]})
    record = builder.finalize(
        {"target_state": result["x"], "phase_gradient": prepared["phase_gradient"]}
    )
    if not adjoint:
        return record
    return adjoint_component_calls(
        record,
        (
            (partition, unpartition),
            (unpartition, partition),
            (dense, dense_inverse),
            (
                permutation,
                permutation_program(size, cycles, target_width, adjoint=True),
            ),
        ),
    )
