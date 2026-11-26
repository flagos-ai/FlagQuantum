# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 first_quantization/prepare_uv.py.

"""Translate potential preparation while retaining the source's opaque zeta leaf."""

from collections.abc import Mapping
from functools import partial

from ..algorithms.first_quantization_potential import (
    prepare_uv_signature,
    prepare_zeta_signature,
)
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.registers.preparation.potential_preparation_component import (
    potential_preparation_program,
)
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.registers import Register
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
    sizes = {field: _finite(arguments, field) for field in required}
    direction = arguments.get("is_adjoint", False)
    if type(direction) not in (bool, int):
        raise SerializationError("Nuclear direction requires a boolean or integer")
    ports = tuple(prepare_zeta_signature(sizes["num_atoms"]))
    if registers is not None and registers != ports:
        raise SerializationError("Invalid nuclear preparation register signature")
    return FunctionRecord(name, (), registers=ports)


def _potential(
    projectile: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
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
    sizes = {field: _finite(arguments, field) for field in required}
    width = sizes["num_bits_n"] if projectile else sizes["num_bits_p"]
    signature = prepare_uv_signature(width, sizes["num_atoms"], sizes["m_param"])
    if registers is not None and registers != tuple(signature):
        raise SerializationError("Invalid potential preparation register signature")
    recipe: dict[str, ResolvedArgument] = {
        "num_bits_p": sizes["num_bits_p"],
        "m_param": sizes["m_param"],
    }
    if projectile:
        recipe["num_bits_n"] = sizes["num_bits_n"]
    kind = (
        "projectile.prepare_nu.PrepareNuStateWithProj"
        if projectile
        else "prepare_nu.PrepareNuState"
    )
    momentum = nu_preparation_factories()[_BASE + kind]("momentum", None, recipe)
    assert isinstance(momentum, ProgramRecord)
    nuclear = ProgramRecord(
        (
            _nuclear(
                "nuclear",
                None,
                {
                    field: sizes[field]
                    for field in ("num_atoms", "lambda_zeta", "num_bits_nuc_pos")
                },
            ),
        )
    )
    return _rename_program(
        potential_preparation_program(signature, momentum, nuclear), name
    )


def potential_preparation_factories() -> dict[str, ComponentFactory]:
    """Bind all three original potential/nuclear constructors without new bodies."""
    return {
        _BASE + "prepare_zeta.PrepareZetaState": _nuclear,
        _BASE + "prepare_uv.PrepareUVFirstQuantization": partial(_potential, False),
        _BASE
        + "projectile.prepare_uv.PrepareUVFirstQuantizationWithProj": partial(
            _potential, True
        ),
    }
