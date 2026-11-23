# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 bookkeeping sources.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/serialization/resolver_dict.py

"""Map lifetime and storage-view library constructors to their native owners."""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Literal

import numpy as np

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register, RegisterType
from ...errors import SerializationError
from ..registers.component_rebindings import join_word_function, split_word_function
from ..registers.layout.component_cast import cast_word_function
from ..registers.layout.component_workspace import (
    allocate_word_function,
    free_word_function,
)
from ..registers.primitives.always_component import always_component_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument

_Kind = Literal["Allocate", "Free", "Split", "Join", "Cast"]


def _always_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    if set(arguments) != {"subbloq"}:
        raise SerializationError("Always requires only subbloq")
    child = arguments["subbloq"]
    if not isinstance(child, FunctionRecord) or child.registers is None:
        raise SerializationError("Always requires a typed child function")
    if registers is not None and registers != child.registers:
        raise SerializationError("Invalid register signature for Always")
    # Resolution already retains the child's real body and all helper definitions.
    # The declaration view lets the existing wrapper owner construct the root call
    # without duplicating or renaming that independently resolved closure.
    declaration = replace(child, blocks=())
    return always_component_program(
        ProgramRecord((declaration,)), child.name, name
    ).functions[0]


def _dtype(arguments: Mapping[str, ResolvedArgument], key: str) -> RegisterType:
    value = arguments.get(key)
    if not isinstance(value, RegisterType):
        raise SerializationError(f"Bookkeeping constructor requires a dtype at {key!r}")
    return value


def _bookkeeping_factory(
    kind: _Kind,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    allowed = (
        {"inp_dtype", "out_dtype", "shape", "allow_quantum_to_classical"}
        if kind == "Cast"
        else {"dtype", "dirty"} if kind in ("Allocate", "Free") else {"dtype"}
    )
    if arguments.keys() - allowed:
        raise SerializationError(f"Unexpected constructor arguments for {kind}")
    if kind == "Cast":
        allow = arguments.get("allow_quantum_to_classical", False)
        if type(allow) not in (int, bool) or allow not in (0, 1):
            raise SerializationError("Cast permission flag must be binary")
        shape = arguments.get("shape", ())
        if isinstance(shape, np.ndarray):
            shape = shape.tolist()
        if type(shape) is int:
            shape = (shape,)
        if not isinstance(shape, (tuple, list)):
            raise SerializationError("Cast shape must contain nonnegative integers")
        dimensions: list[int] = []
        for extent in shape:
            if not isinstance(extent, int) or isinstance(extent, bool) or extent < 0:
                raise SerializationError("Cast shape must contain nonnegative integers")
            dimensions.append(extent)
        # Existing native Cast owns storage admission, including its protected
        # quantum/classical boundary even when the foreign permission is true.
        function = cast_word_function(
            name,
            _dtype(arguments, "inp_dtype"),
            _dtype(arguments, "out_dtype"),
            tuple(dimensions),
        )
    else:
        dtype = _dtype(arguments, "dtype")
        if kind in ("Allocate", "Free"):
            dirty = arguments.get("dirty", False)
            if type(dirty) not in (int, bool) or dirty not in (0, 1):
                raise SerializationError("Workspace dirty flag must be binary")
            constructor = (
                allocate_word_function if kind == "Allocate" else free_word_function
            )
            function = constructor(name, dtype, bool(dirty))
        else:
            rebind = split_word_function if kind == "Split" else join_word_function
            function = rebind(name, dtype)
    if registers is not None and function.registers != registers:
        raise SerializationError(f"Invalid register signature for {kind}")
    return function


def bookkeeping_factories() -> dict[str, ComponentFactory]:
    """Supply existing lifetime/view constructors to explicit library resolution.

    Logical metadata and executable storage retain their existing owners and
    admission rules. The caller can combine this mapping with other native
    factory groups without sharing mutable registry state.
    """
    kinds: tuple[_Kind, ...] = ("Allocate", "Free", "Split", "Join", "Cast")
    return {
        **{
            f"qualtran.bloqs.bookkeeping.{kind.lower()}.{kind}": partial(
                _bookkeeping_factory, kind
            )
            for kind in kinds
        },
        "qualtran.bloqs.bookkeeping.always.Always": _always_factory,
    }
