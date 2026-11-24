# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 sparse_state_preparation_via_rotations.py.

"""Resolve sparse rotation support and precision through shared typed owners."""

from collections.abc import Mapping

import numpy as np

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import (
    opaque_register_type,
    unsigned_register_type,
)
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt, bit_length
from ...errors import SerializationError
from ..registers.preparation.sparse_rotation_preparation_component import (
    sparse_rotation_preparation_program,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_rotation_preparation_factories import rotation_preparation_factories
from .legacy_word_factories import _rename_program

_Recipe = tuple[
    tuple[int, ...], tuple[complex, ...], SymbolicInt, SymbolicInt, SymbolicInt, bool
]


def sparse_rotation_preparation_factories(
    *, adjoint_factory: ComponentFactory | None = None
) -> dict[str, ComponentFactory]:
    """Own source sparse recipes and delegate dense/PRGA inverse dependencies.

    No upstream constructor is imported. Exact support ordering, explicit target
    width and generic source cycle discovery are retained. Shaped/HasLength
    failures in the original serializer remain separate from logical declarations.
    """
    recipes: dict[FunctionRecord, _Recipe] = {}
    dependencies = rotation_preparation_factories(adjoint_factory=adjoint_factory)

    def make(
        name: str, registers: tuple[Register, ...] | None, recipe: _Recipe
    ) -> ProgramRecord:
        indices, coefficients, size, phase, width, inverse = recipe
        ports = tuple(
            Signature.build_from_dtypes(
                target_state=unsigned_register_type(width),
                phase_gradient=opaque_register_type(phase),
            )
        )
        if registers is not None and registers != ports:
            raise SerializationError("Invalid sparse rotation register signature")
        if type(size) is not int or type(width) is not int:
            record = ProgramRecord((FunctionRecord("main", (), registers=ports),))
        else:
            record = sparse_rotation_preparation_program(
                indices, coefficients, size, phase, width, adjoint=inverse
            )
        record = _rename_program(record, name)
        recipes[record.functions[0]] = recipe
        return record

    def forward(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if not {
            "sparse_indices",
            "nonzero_coeffs",
            "N",
            "phase_bitsize",
        } <= arguments.keys() or arguments.keys() - {
            "sparse_indices",
            "nonzero_coeffs",
            "N",
            "phase_bitsize",
            "target_bitsize",
        }:
            raise SerializationError("Invalid sparse rotation constructor arguments")
        raw_indices, raw_coefficients = (
            arguments["sparse_indices"],
            arguments["nonzero_coeffs"],
        )
        if not isinstance(raw_indices, (tuple, list, np.ndarray)) or not isinstance(
            raw_coefficients, (tuple, list, np.ndarray)
        ):
            raise SerializationError(
                "Sparse rotation requires concrete support vectors"
            )
        indices_array, coefficient_array = np.asarray(raw_indices), np.asarray(
            raw_coefficients
        )
        if (
            indices_array.ndim != 1
            or indices_array.dtype.kind not in "iu"
            or coefficient_array.ndim != 1
            or coefficient_array.dtype.kind not in "buifc"
        ):
            raise SerializationError(
                "Sparse rotation requires integer support and coefficient vectors"
            )
        size = _integer(arguments, "N")
        phase = _integer(arguments, "phase_bitsize")
        width = _integer(
            {"target_bitsize": arguments.get("target_bitsize", bit_length(size - 1))},
            "target_bitsize",
        )
        recipe = (
            tuple(int(v) for v in indices_array),
            # Retain NumPy scalar precision through source normalization.
            tuple(coefficient_array),
            size,
            phase,
            width,
            False,
        )
        return make(name, registers, recipe)

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            set(arguments) != {"subbloq"}
            or not isinstance(child, FunctionRecord)
            or child not in recipes
        ):
            return dependencies["qualtran._infra.adjoint.Adjoint"](
                name, registers, arguments
            )
        indices, coefficients, size, phase, width, inverse = recipes[child]
        return make(
            name, registers, (indices, coefficients, size, phase, width, not inverse)
        )

    return {
        **dependencies,
        "qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations.SparseStatePreparationViaRotations": forward,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
