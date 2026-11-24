# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Metadata follows Qualtran v0.7.0 block_encoding/linear_combination.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/linear_combination.py

"""Capture omitted LCU constructor roles at the external object boundary."""

from collections.abc import Callable
from typing import Any, cast

from ..compiler.formats.legacy_component_arguments import (
    ComponentArgument,
    deserialize_component_argument,
    serialize_component_argument,
)
from ..compiler.formats.legacy_component_library import deserialize_component_library
from ..compiler.formats.legacy_linear_combination_factories import (
    ExactPair,
    LinearCombinationContext,
)
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..core.ir.program import ProgramRecord
from ..core.ir.program.signatures import Signature
from ..core.value_math import SymbolicInt
from ..errors import SerializationError

ExternalOracleOwner = Callable[[object], ExactPair]


def capture_linear_combination_context(
    source: object,
    *,
    prepare_owner: ExternalOracleOwner,
    select_owner: ExternalOracleOwner,
) -> LinearCombinationContext:
    """Immediately replace source roles by complete native directional Programs.

    The caller explicitly binds the result to an assigned resolver definition.
    Source objects are passed only to Ecosystem-owned conversion callbacks;
    neither the resulting context nor Compiler retains an external object.
    No missing coefficient or block is inferred from a shallow declaration.
    """
    from qualtran import CtrlSpec
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination
    from qualtran.serialization.bloq import bloqs_to_proto

    if not isinstance(source, LinearCombination):
        raise SerializationError("Expected an explicit source LinearCombination object")
    foreign = cast(Any, source)
    flag = foreign.is_controlled
    if type(flag) not in (bool, int) or flag not in (0, 1):
        raise SerializationError(
            "Source linear-combination control flag must be binary"
        )
    controlled = bool(flag)
    preparation, selection = foreign.prepare, foreign.select
    if controlled:
        selection, _ = selection.get_ctrl_system(CtrlSpec())
    declaration = (
        deserialize_component_library(
            bloqs_to_proto(foreign, max_depth=0).SerializeToString()
        )
        .entries[0]
        .declaration
    )
    if declaration.registers is None:
        raise SerializationError("Source linear combination requires typed ports")
    arguments = {argument.name: argument.value for argument in declaration.arguments}

    def integer(value: Any) -> SymbolicInt:
        # Match the existing wire's projection of symbol assumptions. Widths
        # are resolved by the source before projection, not recomputed later.
        raw = _integer({"value": value}, "value")
        normalized = deserialize_component_argument(
            serialize_component_argument(ComponentArgument("value", raw))
        ).value
        return _integer({"value": normalized}, "value")

    def native_pair(owner: ExternalOracleOwner, child: object) -> ExactPair:
        pair = owner(child)
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(
                not isinstance(program, ProgramRecord)
                or program.physical_context is not None
                for program in pair
            )
        ):
            raise SerializationError(
                "Oracle owner must return two logical native Programs"
            )
        return pair

    return LinearCombinationContext(
        signature=Signature(declaration.registers),
        constituent_ancilla=integer(foreign.be_ancilla_bitsize),
        constituent_resource=integer(foreign.be_resource_bitsize),
        selection=integer(preparation.selection_bitsize),
        junk=integer(preparation.junk_bitsize),
        precision=_integer(arguments, "lambd_bits"),
        controlled=controlled,
        prepare_pair=native_pair(prepare_owner, preparation),
        select_pair=native_pair(select_owner, selection),
    )
