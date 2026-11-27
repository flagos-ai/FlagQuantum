# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow Qualtran v0.7.0 state_preparation_alias_sampling.py.

"""Resolve concrete alias metadata and own its exact inverse dependencies."""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial

import numpy as np

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_compatibility import check_register_types_consistent
from ...core.ir.program.register_types import unsigned_register_type
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt, bit_length
from ...errors import SerializationError
from ..registers.lookup.qrom_table import QromTable
from ..registers.preparation.alias_preparation_component import (
    alias_preparation_component_program,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_reflection_factories import reflection_factories
from .legacy_rotation_factories import _real_argument
from .legacy_word_factories import _rename_program

_Recipe = tuple[Signature, QromTable, SymbolicInt, str, bool, bool]


def alias_preparation_factories(
    *, adjoint_factory: ComponentFactory | None = None
) -> dict[str, ComponentFactory]:
    """Supply dense/sparse constructors, exact Adjoint and reflection dependencies.

    Selection metadata is explicit; it is never inferred from maximum table
    values. Recipes are scoped to one compilation. Reflection receives the
    complete exact inverse and selection names from the same owned recipes.
    Source symbolic Shaped serialization remains outside this concrete codec.
    """
    recipes: dict[FunctionRecord, _Recipe] = {}

    def make(
        name: str,
        registers: tuple[Register, ...] | None,
        signature: Signature,
        lookup: QromTable,
        rows: SymbolicInt,
        selection_name: str,
        sparse: bool,
        inverse: bool,
    ) -> ProgramRecord:
        if registers is not None and registers != tuple(signature):
            raise SerializationError("Invalid alias preparation register signature")
        record = _rename_program(
            alias_preparation_component_program(
                signature,
                lookup,
                rows=rows,
                selection_name=selection_name,
                sparse=sparse,
                adjoint=inverse,
            ),
            name,
        )
        recipes[record.functions[0]] = (
            signature,
            lookup,
            rows,
            selection_name,
            sparse,
            inverse,
        )
        return record

    def forward(
        sparse: bool,
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        required = {
            "selection_registers",
            "alt",
            "keep",
            "mu",
            "sum_of_unnormalized_probabilities",
        }
        if sparse:
            required.add("index")
        if arguments.keys() != required:
            raise SerializationError("Invalid alias preparation constructor arguments")
        raw = arguments["selection_registers"]
        selections = (raw,) if isinstance(raw, Register) else raw
        if (
            not isinstance(selections, (tuple, list))
            or len(selections) != 1
            or not isinstance(selections[0], Register)
        ):
            raise SerializationError(
                "Alias preparation requires one typed selection register"
            )
        selection = selections[0]
        if (
            selection.shape_symbolic
            or selection.side != RegisterSide.THRU
            or selection.dtype.num_cbits
        ):
            raise SerializationError(
                "Alias selection requires a scalar through quantum word"
            )
        bound = dict(selection.dtype.parameters).get("bound")
        if not sparse and (selection.dtype.encoding != "uint" or bound is None):
            raise SerializationError(
                "Dense alias selection requires an explicit unsigned iteration bound"
            )
        if sparse and (
            selection.name != "selection"
            or not check_register_types_consistent(
                selection.dtype, unsigned_register_type(selection.bitsize)
            )
        ):
            raise SerializationError(
                "Sparse alias requires the source selection name and supported word storage"
            )
        # The sparse body uses support length and physical width even when
        # source n_coeff is zero for QUInt/QAny. Do not invent an iteration bound.
        count = _integer({"bound": bound}, "bound") if bound is not None else 0
        mu = _integer(arguments, "mu")
        if type(mu) is int and mu <= 0:
            raise SerializationError("Alias probability width must be positive")
        _real_argument(arguments, "sum_of_unnormalized_probabilities")
        keys = ("index", "alt", "keep") if sparse else ("alt", "keep")
        data = []
        for key in keys:
            value = arguments[key]
            if (
                not isinstance(value, (tuple, list, np.ndarray))
                or np.asarray(value).ndim != 1
            ):
                raise SerializationError(
                    "Alias lookup data requires concrete row vectors"
                )
            data.append(np.asarray(value))
        rows = len(data[0]) if sparse else count
        index_width = bit_length(len(data[0]) - 1) if sparse else selection.bitsize
        width = selection.bitsize
        targets = (width, width, mu) if sparse else (width, mu)
        lookup = QromTable.from_data(*data, target_bitsizes=targets)
        lookup = replace(lookup, selection_bitsizes=(index_width,))
        junk = tuple(
            Signature.build(
                sigma_mu=mu,
                sparse_index=index_width if sparse else 0,
                alt=width,
                keep=mu,
                less_than_equal=1,
            )
        )
        return make(
            name,
            registers,
            Signature((selection, *junk)),
            lookup,
            rows,
            selection.name,
            sparse,
            False,
        )

    def exact_inverse(child: FunctionRecord) -> tuple[ProgramRecord, tuple[str, ...]]:
        if child not in recipes:
            raise SerializationError(
                "Missing exact alias PREPARE inverse constructor context"
            )
        signature, lookup, rows, selection, sparse, inverse = recipes[child]
        if any(reg.name != selection for reg in signature):
            raise SerializationError(
                "Alias reflection requires caller-owned signal junk; "
                "source reflection ports cannot retain that environment"
            )
        return make(
            child.name + "_inverse",
            None,
            signature,
            lookup,
            rows,
            selection,
            sparse,
            not inverse,
        ), (selection,)

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
            if adjoint_factory is not None:
                return adjoint_factory(name, registers, arguments)
            raise SerializationError("Missing exact alias adjoint constructor context")
        signature, lookup, rows, selection, sparse, inverse = recipes[child]
        return make(
            name, registers, signature, lookup, rows, selection, sparse, not inverse
        )

    prefix = "qualtran.bloqs.state_preparation.state_preparation_alias_sampling."
    return {
        **reflection_factories(preparation_inverse=exact_inverse),
        prefix + "StatePreparationAliasSampling": partial(forward, False),
        prefix + "SparseStatePreparationAliasSampling": partial(forward, True),
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
