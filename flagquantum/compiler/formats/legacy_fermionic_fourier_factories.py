# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 qft/two_bit_ffft.py.

"""Bind literal fermionic Fourier declarations and explicit generic adjoints."""

from collections.abc import Mapping
from dataclasses import replace
from math import pi

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.phase.fermionic_fourier import two_bit_fermionic_fourier_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_rotation_factories import _real_argument, _rotation_factory
from .legacy_word_factories import _rename_program


def _fourier_program(
    arguments: Mapping[str, ResolvedArgument], *, inverse: bool
) -> ProgramRecord:
    if not {"k", "n"} <= arguments.keys() or arguments.keys() - {
        "k",
        "n",
        "eps",
        "is_adjoint",
    }:
        raise SerializationError("Fermionic Fourier requires k and n metadata")
    k, n = _integer(arguments, "k"), _integer(arguments, "n")
    if n == 0:
        raise SerializationError("Fermionic Fourier requires nonzero n")
    precision = _real_argument(arguments, "eps", 1e-10)
    adjoint = arguments.get("is_adjoint", False)
    if type(adjoint) not in (bool, int) or adjoint not in (0, 1):
        raise SerializationError("Fermionic Fourier direction requires a boolean")
    # The original class flag does not reverse its decomposition. Only the
    # explicit generic Adjoint constructor selects this exact inverse body.
    program = two_bit_fermionic_fourier_program(0, 1, adjoint=inverse)
    angle = 2 * pi * k / n
    rotation = _rotation_factory(
        "Rz",
        "fourier_rz",
        None,
        {"angle": -angle if inverse else angle, "eps": precision},
    )
    assert isinstance(rotation, FunctionRecord)
    return replace(
        program,
        functions=tuple(
            rotation if function.name == rotation.name else function
            for function in program.functions
        ),
    )


def fermionic_fourier_factories(
    *, adjoint_factory: ComponentFactory | None = None
) -> dict[str, ComponentFactory]:
    """Keep exact inverse recipes local to one explicit library resolution."""
    recipes: dict[FunctionRecord, tuple[Mapping[str, ResolvedArgument], bool]] = {}

    def make(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
        *,
        inverse: bool = False,
    ) -> ProgramRecord:
        program = _fourier_program(arguments, inverse=inverse)
        if registers is not None and registers != program.functions[0].registers:
            raise SerializationError("Invalid fermionic Fourier register signature")
        record = _rename_program(program, name)
        recipes[record.functions[0]] = dict(arguments), inverse
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
            metadata, direction = recipes[child]
            return make(name, registers, metadata, inverse=not direction)
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing exact fermionic Fourier adjoint context")

    return {
        "qualtran.bloqs.qft.two_bit_ffft.TwoBitFFFT": make,
        "qualtran._infra.adjoint.Adjoint": inverse,
    }
