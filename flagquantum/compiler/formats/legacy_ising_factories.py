# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 chemistry/trotter/ising/unitaries.py.

"""Resolve periodic Ising constructors through shared typed rotation owners."""

from collections.abc import Mapping
from functools import partial

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ..registers.primitives.ising_component import ising_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_fixed_gate_factories import fixed_gate_factories
from .legacy_modular_addition_factories import _integer
from .legacy_rotation_factories import _real_argument, rotation_factories
from .legacy_word_factories import _rename_program


def _ising_factory(
    periodic: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if not {"nsites", "angle"} <= arguments.keys() or arguments.keys() - {
        "nsites",
        "angle",
        "eps",
    }:
        raise SerializationError("Ising requires sites, angle and optional precision")
    sites = _integer(arguments, "nsites")
    angle = _real_argument(arguments, "angle")
    epsilon = _real_argument(arguments, "eps", 1e-10)
    signature = tuple(Signature.build(system=sites))
    if registers is not None and registers != signature:
        raise SerializationError("Invalid Ising register signature")
    if type(sites) is not int or sites < (2 if periodic else 1):
        return FunctionRecord(name, (), registers=signature)
    assert isinstance(sites, int)
    kind = "Rz" if periodic else "Rx"
    rotation = rotation_factories()[f"qualtran.bloqs.basic_gates.rotation.{kind}"](
        "ising_rotation", None, {"angle": angle, "eps": epsilon if periodic else 1e-11}
    )
    assert isinstance(rotation, FunctionRecord)
    cnot: FunctionRecord | None = None
    if periodic:
        dependency = fixed_gate_factories()["qualtran.bloqs.basic_gates.cnot.CNOT"](
            "ising_cnot", None, {}
        )
        assert isinstance(dependency, FunctionRecord)
        cnot = dependency
    return _rename_program(ising_program(sites, rotation, cnot=cnot), name)


def ising_factories() -> dict[str, ComponentFactory]:
    """Bind both source constructors without importing their SDK classes."""
    prefix = "qualtran.bloqs.chemistry.trotter.ising.unitaries."
    return {
        prefix + "IsingXUnitary": partial(_ising_factory, False),
        prefix + "IsingZZUnitary": partial(_ising_factory, True),
    }
