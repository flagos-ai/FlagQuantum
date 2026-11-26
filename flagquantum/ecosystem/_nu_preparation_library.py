# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 first_quantization/prepare_nu.py.

"""Translate finite momentum-preparation constructors at their source boundary."""

from collections.abc import Mapping
from functools import partial

from ..algorithms.first_quantization_nu import (
    prepare_nu_leaf_signatures,
    prepare_nu_signature,
    projectile_prepare_mu_signature,
)
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_uniform_preparation_factories import (
    uniform_preparation_factories,
)
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.registers.preparation.nu_preparation_component import (
    nu_preparation_program,
)
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.registers import Register
from ..errors import SerializationError

_ORDINARY = "qualtran.bloqs.chemistry.pbc.first_quantization.prepare_nu."
_PROJECTILE = "qualtran.bloqs.chemistry.pbc.first_quantization.projectile.prepare_nu."
_LEAVES = (
    "PrepareMuUnaryEncodedOneHot",
    "PrepareNuSuperPositionState",
    "FlagZeroAsFailure",
    "TestNuLessThanMu",
    "TestNuInequality",
)


def _finite(arguments: Mapping[str, ResolvedArgument], field: str) -> int:
    value = _integer(arguments, field)
    if type(value) is not int:
        raise SerializationError("Momentum constructor requires concrete integer sizes")
    return value


def _leaf(
    kind: str,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    projectile = kind == "PrepareMuUnaryEncodedOneHotWithProj"
    required = {"bitsize_n", "bitsize_p"} if projectile else {"num_bits_p"}
    if kind == "TestNuInequality":
        required.add("num_bits_m")
    optional = set() if kind == "PrepareMuUnaryEncodedOneHot" else {"is_adjoint"}
    if not required <= arguments.keys() or arguments.keys() - required - optional:
        raise SerializationError("Invalid momentum leaf constructor arguments")
    direction = arguments.get("is_adjoint", False)
    if type(direction) not in (bool, int):
        raise SerializationError("Momentum direction requires a boolean or integer")
    if projectile:
        # The unchanged source keeps RIGHT even for its declared adjoint.
        signature = projectile_prepare_mu_signature(
            _finite(arguments, "bitsize_p"), _finite(arguments, "bitsize_n")
        )
    else:
        width = _finite(arguments, "num_bits_p")
        m_width = _finite(arguments, "num_bits_m") if kind == "TestNuInequality" else 0
        if m_width < 0:
            raise SerializationError("Momentum sample width must be nonnegative")
        signature = prepare_nu_leaf_signatures(
            width, 1 << m_width, adjoint=bool(direction)
        )[kind]
    ports = tuple(signature)
    if registers is not None and ports != registers:
        raise SerializationError("Invalid momentum leaf register signature")
    return FunctionRecord(name, (), registers=ports)


def _root(
    projectile: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> ProgramRecord:
    required = {"num_bits_p", "m_param"} | ({"num_bits_n"} if projectile else set())
    if arguments.keys() != required:
        raise SerializationError("Invalid momentum root constructor arguments")
    electrons, sample = _finite(arguments, "num_bits_p"), _finite(arguments, "m_param")
    width = _finite(arguments, "num_bits_n") if projectile else electrons
    signature = prepare_nu_signature(width, sample)
    if registers is not None and tuple(signature) != registers:
        raise SerializationError("Invalid momentum root register signature")
    roles = ("mu", "nu", "zero", "less", "inequality")
    leaves = {}
    for role, kind in zip(roles, _LEAVES, strict=True):
        recipe: dict[str, ResolvedArgument] = {"num_bits_p": width}
        if role == "mu" and projectile:
            kind = "PrepareMuUnaryEncodedOneHotWithProj"
            recipe = {"bitsize_n": width, "bitsize_p": electrons}
        elif role == "inequality":
            recipe["num_bits_m"] = (sample - 1).bit_length()
        leaves[role] = ProgramRecord((_leaf(kind, kind, None, recipe),))
    uniform = uniform_preparation_factories()[
        "qualtran.bloqs.state_preparation.prepare_uniform_superposition.PrepareUniformSuperposition"
    ]("PrepareUniformSuperposition", None, {"n": sample})
    record = (
        uniform if isinstance(uniform, ProgramRecord) else ProgramRecord((uniform,))
    )
    return _rename_program(nu_preparation_program(signature, leaves, record), name)


def nu_preparation_factories() -> dict[str, ComponentFactory]:
    """Keep opaque leaves and invalid source inverse ownership observable."""
    return {
        **{_ORDINARY + kind: partial(_leaf, kind) for kind in _LEAVES},
        _ORDINARY + "PrepareNuState": partial(_root, False),
        _PROJECTILE
        + "PrepareMuUnaryEncodedOneHotWithProj": partial(
            _leaf, "PrepareMuUnaryEncodedOneHotWithProj"
        ),
        _PROJECTILE + "PrepareNuStateWithProj": partial(_root, True),
    }
