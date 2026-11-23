# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 conversions/sign_extension.py.

"""Resolve sign-prefix lifetime conversions through the existing component owner."""

from collections.abc import Mapping
from functools import partial

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...errors import SerializationError
from ..registers.bits.sign_extension_component import sign_extension_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_word_factories import _rename_program


def _sign_factory(
    truncate: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if set(arguments) != {"inp_dtype", "out_dtype"}:
        raise SerializationError("Sign conversion requires input and output dtypes")
    before, after = arguments["inp_dtype"], arguments["out_dtype"]
    if (
        not isinstance(before, RegisterType)
        or not isinstance(after, RegisterType)
        or before.encoding not in {"signed", "ones_complement"}
        or after.encoding != before.encoding
    ):
        raise SerializationError("Sign conversion requires matching signed encodings")
    signature = (
        Register("x", before, side=RegisterSide.LEFT),
        Register("y", after, side=RegisterSide.RIGHT),
    )
    if registers is not None and signature != registers:
        raise SerializationError("Invalid register signature for sign conversion")
    if before.is_symbolic() or after.is_symbolic():
        return FunctionRecord(name, (), registers=signature)
    narrow, wide = (after, before) if truncate else (before, after)
    return _rename_program(sign_extension_program(narrow, wide, adjoint=truncate), name)


def sign_extension_factories() -> dict[str, ComponentFactory]:
    """Bind extension and truncation to the shared redundant-prefix owner."""
    prefix = "qualtran.bloqs.arithmetic.conversions.sign_extension."
    return {
        prefix + "SignExtend": partial(_sign_factory, False),
        prefix + "SignTruncate": partial(_sign_factory, True),
    }
