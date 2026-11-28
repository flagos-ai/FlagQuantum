# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 first_quantization/prepare_nu.py.

"""Translate finite momentum-preparation constructors at their source boundary."""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial

from ..algorithms.first_quantization_nu import _nu_declarations
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
from ..core.ir.program.signatures import Signature
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
        electron = _integer(arguments, "bitsize_p")
        width = _integer(arguments, "bitsize_n")
        try:
            invalid_width = bool(width - electron < 0)
        except TypeError:
            if registers is None:
                raise
            # The original codec drops symbol assumptions. An existing stored
            # declaration still owns its signature, but supplies no gate body.
            invalid_width = False
        if invalid_width:
            raise SerializationError("Projectile width is less than electron width")
        signature = _nu_declarations(width, 0)["PrepareMuUnaryEncodedOneHot"]
    else:
        width = _integer(arguments, "num_bits_p")
        m_width = _integer(arguments, "num_bits_m") if kind == "TestNuInequality" else 0
        signature = _nu_declarations(width, m_width, adjoint=bool(direction))[kind]
    ports = tuple(signature)
    if registers is not None and ports != registers:
        raise SerializationError("Invalid momentum leaf register signature")
    return FunctionRecord(name, (), registers=ports)


def _root(
    projectile: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
    *,
    adjoint: bool = False,
) -> FunctionRecord | ProgramRecord:
    required = {"num_bits_p", "m_param"} | ({"num_bits_n"} if projectile else set())
    if arguments.keys() != required:
        raise SerializationError("Invalid momentum root constructor arguments")
    electrons = _integer(arguments, "num_bits_p")
    sample = _finite(arguments, "m_param")
    width = _integer(arguments, "num_bits_n") if projectile else electrons
    signature = _nu_declarations(width, (sample - 1).bit_length())["PrepareNuState"]
    if registers is not None and tuple(signature) != registers:
        raise SerializationError("Invalid momentum root register signature")
    if projectile and adjoint:
        # The original inverse attempts to consume a flag still declared RIGHT.
        return FunctionRecord(name, (), registers=tuple(signature))
    if sample <= 1:
        # The source singleton loses its target; nonpositive counts fail log2.
        return FunctionRecord(name, (), registers=tuple(signature))
    if projectile:
        try:
            invalid_widths = bool(width - electrons < 0)
        except TypeError:
            # A shallow source root exists even when constructing its mu child
            # fails the unresolved relational test. Preserve only that header.
            return FunctionRecord(name, (), registers=tuple(signature))
        if invalid_widths:
            return FunctionRecord(name, (), registers=tuple(signature))
    roles = ("mu", "nu", "zero", "less", "inequality")
    leaves = {}
    for role, kind in zip(roles, _LEAVES, strict=True):
        recipe: dict[str, ResolvedArgument] = {"num_bits_p": width}
        if role == "mu" and projectile:
            kind = "PrepareMuUnaryEncodedOneHotWithProj"
            recipe = {"bitsize_n": width, "bitsize_p": electrons}
        elif role == "inequality":
            recipe["num_bits_m"] = (sample - 1).bit_length()
        if adjoint and role != "mu":
            recipe["is_adjoint"] = True
        leaf = _leaf(kind, f"{kind}_adjoint" if adjoint else kind, None, recipe)
        if adjoint and role == "mu":
            assert leaf.registers is not None
            leaf = replace(leaf, registers=tuple(Signature(leaf.registers).adjoint()))
        leaves[role] = ProgramRecord((leaf,))
    uniform_factories = uniform_preparation_factories()
    uniform = uniform_factories[
        "qualtran.bloqs.state_preparation.prepare_uniform_superposition.PrepareUniformSuperposition"
    ]("PrepareUniformSuperposition", None, {"n": sample})
    if adjoint:
        child = uniform.functions[0] if isinstance(uniform, ProgramRecord) else uniform
        uniform = uniform_factories["qualtran._infra.adjoint.Adjoint"](
            "PrepareUniformSuperposition", None, {"subbloq": child}
        )
    record = (
        uniform if isinstance(uniform, ProgramRecord) else ProgramRecord((uniform,))
    )
    return _rename_program(
        nu_preparation_program(signature, leaves, record, adjoint=adjoint), name
    )


def nu_preparation_factories(
    *,
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Keep literal logical recipes and exact inverse context in one resolution."""
    recipes: dict[FunctionRecord, tuple[str, Mapping[str, ResolvedArgument], bool]] = {}

    def make(
        kind: str,
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
        *,
        inverse: bool = False,
    ) -> FunctionRecord | ProgramRecord:
        if kind in ("ordinary", "projectile"):
            record = _root(
                kind == "projectile", name, registers, arguments, adjoint=inverse
            )
        else:
            record = _leaf(kind, name, registers, arguments)
            if inverse:
                assert record.registers is not None
                record = replace(
                    record, registers=tuple(Signature(record.registers).adjoint())
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
            # Inverse signatures are checked after reversing an opaque leaf.
            record = make(kind, name, None, metadata, inverse=not direction)
            root = record.functions[0] if isinstance(record, ProgramRecord) else record
            if registers is not None and root.registers != registers:
                raise SerializationError("Invalid momentum inverse register signature")
            return record
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing exact momentum inverse constructor context")

    return {
        **{_ORDINARY + kind: partial(make, kind) for kind in _LEAVES},
        _ORDINARY + "PrepareNuState": partial(make, "ordinary"),
        _PROJECTILE
        + "PrepareMuUnaryEncodedOneHotWithProj": partial(
            make, "PrepareMuUnaryEncodedOneHotWithProj"
        ),
        _PROJECTILE + "PrepareNuStateWithProj": partial(make, "projectile"),
        "qualtran._infra.adjoint.Adjoint": inverse,
    }
