# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 first_quantization/prepare_t.py.

"""Translate original kinetic constructors without inventing opaque leaf bodies."""

from collections.abc import Mapping
from functools import partial

from ..algorithms.first_quantization_kinetic import (
    prepare_power_two_signature,
    prepare_t_signature,
    projectile_prepare_power_two_signature,
    projectile_prepare_t_signature,
)
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_uniform_preparation_factories import (
    uniform_preparation_factories,
)
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.registers.preparation.kinetic_preparation_component import (
    kinetic_preparation_program,
)
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.registers import Register
from ..errors import SerializationError
from ._nu_preparation_library import _finite

_BASE = "qualtran.bloqs.chemistry.pbc.first_quantization."


def _kinetic(
    kind: str,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
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
    sizes = {field: _finite(arguments, field) for field in required}
    if "num_bits_rot_aa" in arguments:
        _finite(arguments, "num_bits_rot_aa")
    direction = arguments.get("is_adjoint", False)
    if type(direction) not in (bool, int):
        raise SerializationError("Kinetic direction requires a boolean or integer")
    if power:
        signature = (
            projectile_prepare_power_two_signature(
                sizes["bitsize_n"], sizes["bitsize_p"]
            )
            if projectile
            else prepare_power_two_signature(sizes["bitsize"])
        )
    else:
        signature = (
            projectile_prepare_t_signature(sizes["num_bits_p"], sizes["num_bits_n"])
            if projectile
            else prepare_t_signature(sizes["num_bits_p"])
        )
    ports = tuple(signature)
    if registers is not None and ports != registers:
        raise SerializationError("Invalid kinetic preparation register signature")
    if power or projectile:
        return FunctionRecord(name, (), registers=ports)
    unary = FunctionRecord(
        "power_two",
        (),
        registers=tuple(prepare_power_two_signature(sizes["num_bits_p"])),
    )
    uniform = uniform_preparation_factories()[
        "qualtran.bloqs.state_preparation.prepare_uniform_superposition.PrepareUniformSuperposition"
    ]("coordinate", None, {"n": 3})
    assert isinstance(uniform, ProgramRecord)
    return _rename_program(
        kinetic_preparation_program(signature, uniform, ProgramRecord((unary,))), name
    )


def kinetic_preparation_factories() -> dict[str, ComponentFactory]:
    """Preserve all four source identifiers at their original physical boundary."""
    return {
        _BASE
        + ("projectile.prepare_t." if kind.endswith("WithProj") else "prepare_t.")
        + kind: partial(_kinetic, kind)
        for kind in (
            "PreparePowerTwoState",
            "PrepareTFirstQuantization",
            "PreparePowerTwoStateWithProj",
            "PrepareTFirstQuantizationWithProj",
        )
    }
