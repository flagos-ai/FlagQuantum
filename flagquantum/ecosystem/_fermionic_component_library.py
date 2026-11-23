# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adapted from locked Qualtran v0.7.0 qft/two_bit_ffft.py.

"""Convert stored fermionic Fourier metadata into the existing owned Program."""

from collections.abc import Mapping
from dataclasses import replace
from math import pi

from ..algorithms.primitives.fermionic_fourier import (
    source_two_bit_fermionic_fourier_signature,
    two_bit_fermionic_fourier_program,
)
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_rotation_factories import (
    _real_argument,
    _rotation_factory,
)
from ..compiler.formats.legacy_word_factories import _rename_program
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.registers import Register
from ..errors import SerializationError


def _fermionic_fourier_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
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
    ports = tuple(source_two_bit_fermionic_fourier_signature())
    if registers is not None and registers != ports:
        raise SerializationError("Invalid fermionic Fourier register signature")
    program = two_bit_fermionic_fourier_program(0, 1)
    rotation = _rotation_factory(
        "Rz", "fourier_rz", None, {"angle": 2 * pi * k / n, "eps": precision}
    )
    assert isinstance(rotation, FunctionRecord)
    program = replace(
        program,
        functions=tuple(
            rotation if function.name == rotation.name else function
            for function in program.functions
        ),
    )
    # The frozen decomposition ignores is_adjoint. Its analytical tensor honors it.
    return _rename_program(program, name)


def fermionic_library_factories() -> dict[str, ComponentFactory]:
    """Supply literal source-decomposition semantics using the owned algorithm."""
    return {"qualtran.bloqs.qft.two_bit_ffft.TwoBitFFFT": _fermionic_fourier_factory}
