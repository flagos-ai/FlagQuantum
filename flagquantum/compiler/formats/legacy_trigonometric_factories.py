# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from Qualtran v0.7.0 arithmetic/trigonometric.

"""Resolve complete trigonometric constructors through their declaration owners."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

from ...core.ir.program import FunctionRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.fixed_point.arcsine_declaration import ArcsineDeclaration
from ..registers.lookup.arctangent_declaration import ArctangentDeclaration
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer


def _trigonometric_factory(
    kind: Literal["arcsin", "arctan"],
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    required = (
        {"bitsize", "num_frac"}
        if kind == "arcsin"
        else {"selection_bitsize", "target_bitsize"}
    )
    allowed = required | {"num_iters", "degree"} if kind == "arcsin" else required
    if not required <= arguments.keys() or arguments.keys() - allowed:
        raise SerializationError(f"Invalid {kind} constructor arguments")
    if kind == "arcsin":
        fields = {key: _integer(arguments, key) for key in arguments}
        signature = ArcsineDeclaration(**fields).signature
    else:
        signature = ArctangentDeclaration(
            _integer(arguments, "selection_bitsize"),
            _integer(arguments, "target_bitsize"),
        ).signature
    ports = tuple(signature)
    if registers is not None and registers != ports:
        raise SerializationError(f"Invalid {kind} register signature")
    return FunctionRecord(name, (), registers=ports)


def trigonometric_factories() -> dict[str, ComponentFactory]:
    """Supply both original constructors without inventing a primitive body."""
    return {
        "qualtran.bloqs.arithmetic.trigonometric.arcsin.ArcSin": partial(
            _trigonometric_factory, "arcsin"
        ),
        "qualtran.bloqs.arithmetic.trigonometric.arctan.ArcTan": partial(
            _trigonometric_factory, "arctan"
        ),
    }
