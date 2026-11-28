# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 first_quantization/prepare_t.py.

"""Translate original kinetic constructors without inventing opaque leaf bodies."""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial

from ..algorithms.first_quantization_kinetic import _kinetic_declaration
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_uniform_preparation_factories import (
    uniform_preparation_factories,
)
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.registers.component_adjoint import adjoint_component_calls
from ..compiler.registers.preparation.kinetic_preparation_component import (
    kinetic_preparation_program,
)
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..errors import SerializationError
from ._source_constructor_adjoints import constructor_adjoint_factories

_BASE = "qualtran.bloqs.chemistry.pbc.first_quantization."


def _kinetic(
    kind: str,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
    *,
    adjoint: bool = False,
) -> FunctionRecord | ProgramRecord:
    projectile = kind.endswith("WithProj")
    power = kind.startswith("PreparePowerTwoState")
    required = (
        ({"bitsize_n", "bitsize_p"} if projectile else {"bitsize"})
        if power
        else (
            {"num_bits_p", "num_bits_n", "eta"} if projectile else {"num_bits_p", "eta"}
        )
    )
    optional = ({"is_adjoint"} if projectile else set()) | (
        set() if power else {"num_bits_rot_aa"}
    )
    if not required <= arguments.keys() or arguments.keys() - required - optional:
        raise SerializationError("Invalid kinetic preparation constructor fields")
    sizes = {field: _integer(arguments, field) for field in required}
    if "num_bits_rot_aa" in arguments:
        _integer(arguments, "num_bits_rot_aa")
    direction = arguments.get("is_adjoint", False)
    if type(direction) not in (bool, int):
        raise SerializationError("Kinetic direction requires a boolean or integer")
    width = (
        sizes["bitsize_n" if projectile else "bitsize"]
        if power
        else sizes["num_bits_n" if projectile else "num_bits_p"]
    )
    if power and projectile and width - sizes["bitsize_p"] < 0:
        raise ValueError("Projectile momentum width is less than electron width")
    signature = _kinetic_declaration(width, projectile=projectile, power=power)
    ports = tuple(signature)
    if registers is not None and ports != registers:
        raise SerializationError("Invalid kinetic preparation register signature")
    if power or projectile or width == 0:
        return FunctionRecord(
            name, (), registers=tuple(signature.adjoint()) if adjoint else ports
        )
    unary = FunctionRecord(
        "power_two",
        (),
        registers=tuple(_kinetic_declaration(width, power=True)),
    )
    uniform = uniform_preparation_factories()[
        "qualtran.bloqs.state_preparation.prepare_uniform_superposition.PrepareUniformSuperposition"
    ]("coordinate", None, {"n": 3})
    assert isinstance(uniform, ProgramRecord)
    record = kinetic_preparation_program(signature, uniform, ProgramRecord((unary,)))
    if adjoint:
        owners = uniform_preparation_factories()
        forward = owners[
            "qualtran.bloqs.state_preparation.prepare_uniform_superposition.PrepareUniformSuperposition"
        ]("coordinate", None, {"n": 3})
        child = forward.functions[0] if isinstance(forward, ProgramRecord) else forward
        inverse = owners["qualtran._infra.adjoint.Adjoint"](
            "coordinate_adjoint", None, {"subbloq": child}
        )
        assert isinstance(inverse, ProgramRecord)
        inverse_power = replace(
            unary,
            name="power_two_adjoint",
            registers=tuple(Signature(unary.registers or ()).adjoint()),
        )
        record = adjoint_component_calls(
            record,
            (
                (uniform, inverse),
                (ProgramRecord((unary,)), ProgramRecord((inverse_power,))),
            ),
        )
    return _rename_program(record, name)


def kinetic_preparation_factories() -> dict[str, ComponentFactory]:
    """Preserve all four source identifiers at their original physical boundary."""
    kinds = (
        "PreparePowerTwoState",
        "PrepareTFirstQuantization",
        "PreparePowerTwoStateWithProj",
        "PrepareTFirstQuantizationWithProj",
    )
    owners = {
        _BASE
        + ("projectile.prepare_t." if kind.endswith("WithProj") else "prepare_t.")
        + kind: partial(_kinetic, kind)
        for kind in kinds
    }
    inverses = {
        _BASE
        + ("projectile.prepare_t." if kind.endswith("WithProj") else "prepare_t.")
        + kind: partial(_kinetic, kind, adjoint=True)
        for kind in kinds
    }
    return constructor_adjoint_factories(owners, inverses)
