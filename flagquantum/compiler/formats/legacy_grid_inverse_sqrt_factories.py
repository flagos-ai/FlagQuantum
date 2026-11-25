# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 chemistry/trotter/grid_ham/inverse_sqrt.py.

"""Bind both original grid inverse-root declarations without invented gates."""

from collections.abc import Mapping
from functools import partial

from ...core.ir.program import FunctionRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.fixed_point.grid_inverse_sqrt_signature import (
    grid_inverse_sqrt_signature,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer


def _inverse_sqrt_factory(
    newton: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    output_key = "target_bitsize" if newton else "out_bitsize"
    if arguments.keys() != {"x_sq_bitsize", "poly_bitsize", output_key}:
        raise SerializationError(
            "Invalid grid inverse-square-root constructor arguments"
        )
    ports = tuple(
        grid_inverse_sqrt_signature(
            _integer(arguments, "x_sq_bitsize"),
            _integer(arguments, "poly_bitsize"),
            _integer(arguments, output_key),
            newton=newton,
        )
    )
    if registers is not None and registers != ports:
        raise SerializationError("Invalid grid inverse-square-root register signature")
    return FunctionRecord(name, (), registers=ports)


def grid_inverse_sqrt_factories() -> dict[str, ComponentFactory]:
    """Keep formula-only roots separate from independently specified execution."""
    prefix = "qualtran.bloqs.chemistry.trotter.grid_ham.inverse_sqrt."
    return {
        prefix
        + "NewtonRaphsonApproxInverseSquareRoot": partial(_inverse_sqrt_factory, True),
        prefix
        + "PolynmomialEvaluationInverseSquareRoot": partial(
            _inverse_sqrt_factory, False
        ),
    }
