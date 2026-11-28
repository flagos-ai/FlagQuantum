# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 first_quantization/prepare_uv.py.

"""Translate potential preparation while retaining the source's opaque zeta leaf."""

from collections.abc import Mapping
from functools import partial

from ..algorithms.first_quantization_nu import _nu_declarations
from ..algorithms.first_quantization_potential import _zeta_declaration
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.registers.preparation.potential_preparation_component import (
    potential_preparation_program,
)
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..errors import SerializationError
from ._nu_preparation_library import _finite, nu_preparation_factories

_BASE = "qualtran.bloqs.chemistry.pbc.first_quantization."


def _nuclear(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    required = {"num_atoms", "lambda_zeta", "num_bits_nuc_pos"}
    if not required <= arguments.keys() or arguments.keys() - required - {"is_adjoint"}:
        raise SerializationError("Invalid nuclear preparation constructor fields")
    # Charge and position precision do not participate in this declaration.
    # Their source cost-query failures must not prevent reading a signature.
    atoms = _finite(arguments, "num_atoms")
    direction = arguments.get("is_adjoint", False)
    if type(direction) not in (bool, int):
        raise SerializationError("Nuclear direction requires a boolean or integer")
    ports = tuple(_zeta_declaration(atoms))
    if registers is not None and registers != ports:
        raise SerializationError("Invalid nuclear preparation register signature")
    return FunctionRecord(name, (), registers=ports)


def _potential(
    projectile: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
    *,
    adjoint: bool = False,
) -> ProgramRecord:
    required = {
        "num_bits_p",
        "eta",
        "num_atoms",
        "m_param",
        "lambda_zeta",
        "num_bits_nuc_pos",
    }
    if projectile:
        required.add("num_bits_n")
    if arguments.keys() != required:
        raise SerializationError("Invalid potential preparation constructor fields")
    sizes = {
        field: _integer(arguments, field)
        for field in ("num_bits_p", "num_bits_n")
        if field in required
    }
    sample = _finite(arguments, "m_param")
    atoms = _finite(arguments, "num_atoms")
    width = sizes["num_bits_n"] if projectile else sizes["num_bits_p"]
    mu, nu, m, flag = _nu_declarations(width, (sample - 1).bit_length())[
        "PrepareNuState"
    ]
    (index,) = _zeta_declaration(atoms)
    signature = Signature((mu, nu, m, index, flag))
    if registers is not None and registers != tuple(signature):
        raise SerializationError("Invalid potential preparation register signature")
    recipe: dict[str, ResolvedArgument] = {
        "num_bits_p": sizes["num_bits_p"],
        "m_param": sample,
    }
    if projectile:
        recipe["num_bits_n"] = sizes["num_bits_n"]
    kind = (
        "projectile.prepare_nu.PrepareNuStateWithProj"
        if projectile
        else "prepare_nu.PrepareNuState"
    )
    momentum_factories = nu_preparation_factories()
    momentum = momentum_factories[_BASE + kind]("momentum", None, recipe)
    if adjoint:
        child = (
            momentum.functions[0] if isinstance(momentum, ProgramRecord) else momentum
        )
        momentum = momentum_factories["qualtran._infra.adjoint.Adjoint"](
            "momentum", None, {"subbloq": child}
        )
    momentum = (
        momentum if isinstance(momentum, ProgramRecord) else ProgramRecord((momentum,))
    )
    nuclear = ProgramRecord(
        (
            _nuclear(
                "nuclear_adjoint" if adjoint else "nuclear",
                None,
                {
                    **{
                        field: arguments[field]
                        for field in ("num_atoms", "lambda_zeta", "num_bits_nuc_pos")
                    },
                    **({"is_adjoint": True} if adjoint else {}),
                },
            ),
        )
    )
    return _rename_program(
        potential_preparation_program(signature, momentum, nuclear, adjoint=adjoint),
        name,
    )


def potential_preparation_factories(
    *,
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Retain shallow root inverses without inventing opaque child bodies."""
    recipes: dict[FunctionRecord, tuple[str, Mapping[str, ResolvedArgument], bool]] = {}

    def make(
        kind: str,
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
        *,
        inverse: bool = False,
    ) -> FunctionRecord | ProgramRecord:
        if kind == "nuclear":
            metadata = dict(arguments)
            if inverse:
                metadata["is_adjoint"] = not bool(arguments.get("is_adjoint", False))
            record: FunctionRecord | ProgramRecord = _nuclear(name, registers, metadata)
        else:
            record = _potential(
                kind == "projectile", name, registers, arguments, adjoint=inverse
            )
        root = record.functions[0] if isinstance(record, ProgramRecord) else record
        recipes[root] = kind, dict(arguments), inverse
        return record

    def inverse(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            set(arguments) == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in recipes
        ):
            kind, metadata, direction = recipes[child]
            return make(kind, name, registers, metadata, inverse=not direction)
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing exact potential inverse constructor context")

    return {
        _BASE + "prepare_zeta.PrepareZetaState": partial(make, "nuclear"),
        _BASE + "prepare_uv.PrepareUVFirstQuantization": partial(make, "ordinary"),
        _BASE
        + "projectile.prepare_uv.PrepareUVFirstQuantizationWithProj": partial(
            make, "projectile"
        ),
        "qualtran._infra.adjoint.Adjoint": inverse,
    }
