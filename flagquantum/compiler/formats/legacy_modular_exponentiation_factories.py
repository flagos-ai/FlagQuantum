# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 cryptography/rsa/rsa_mod_exp.py.

"""Restore the original modular exponentiation constructor and typed body."""

from collections.abc import Mapping
from math import gcd

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import unsigned_register_type
from ...core.ir.program.registers import Register, RegisterSide
from ...errors import SerializationError
from ..registers.modular.modular_exponentiation_component import (
    modular_exponentiation_program,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program


def _modexp_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
    *,
    adjoint: bool = False,
) -> FunctionRecord | ProgramRecord:
    if arguments.keys() != {"base", "mod", "exp_bitsize", "x_bitsize"}:
        raise SerializationError("Invalid modular exponentiation constructor arguments")
    base, modulus, exponent, output = (
        _integer(arguments, key) for key in ("base", "mod", "exp_bitsize", "x_bitsize")
    )
    signature = (
        Register("exponent", unsigned_register_type(exponent)),
        Register(
            "x",
            unsigned_register_type(output),
            side=RegisterSide.LEFT if adjoint else RegisterSide.RIGHT,
        ),
    )
    if registers is not None and registers != signature:
        raise SerializationError("Invalid modular exponentiation register signature")
    if isinstance(base, int) and isinstance(modulus, int) and gcd(base, modulus) != 1:
        raise SerializationError(
            "Modular exponentiation base and modulus must be coprime"
        )
    if any(isinstance(v, sympy.Expr) for v in (base, modulus, exponent, output)):
        return FunctionRecord(name, (), registers=signature)
    assert (
        isinstance(base, int)
        and isinstance(modulus, int)
        and isinstance(exponent, int)
        and isinstance(output, int)
    )
    return _rename_program(
        modular_exponentiation_program(
            base, modulus, exponent, output, adjoint=adjoint
        ),
        name,
    )


def modular_exponentiation_factories(
    *,
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Scope exact source-adjoint recipes to this resolver compilation."""
    recipes: dict[FunctionRecord, tuple[dict[str, ResolvedArgument], bool]] = {}

    def make(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
        inverse: bool,
    ) -> FunctionRecord | ProgramRecord:
        record = _modexp_factory(name, registers, arguments, adjoint=inverse)
        root = record.functions[0] if isinstance(record, ProgramRecord) else record
        recipes[root] = (dict(arguments), inverse)
        return record

    def forward(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        return make(name, registers, arguments, False)

    def inverse(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            arguments.keys() == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in recipes
        ):
            values, direction = recipes[child]
            return make(name, registers, values, not direction)
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing exact modular exponentiation adjoint context")

    return {
        "qualtran.bloqs.cryptography.rsa.rsa_mod_exp.ModExp": forward,
        "qualtran._infra.adjoint.Adjoint": inverse,
    }
