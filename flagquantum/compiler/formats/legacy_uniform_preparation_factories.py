# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 prepare_uniform_superposition.py.

"""Resolve uniform preparation and scoped exact inverse dependency recipes."""

from collections.abc import Mapping

import numpy as np
import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...errors import SerializationError
from ..registers.preparation.uniform_preparation_component import (
    uniform_preparation_program,
)
from ..registers.preparation.uniform_preparation_resource_model import (
    source_uniform_preparation_signature,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_conjunction_factories import _control_value
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program


def uniform_preparation_factories(
    *, adjoint_factory: ComponentFactory | None = None
) -> dict[str, ComponentFactory]:
    """Reuse the owned uniform graph, retaining symbolic and singleton declarations.

    Recipes are local to one compilation. Unknown inverse children delegate to
    an explicit owner or refuse. No external constructor is imported at runtime.
    """
    recipes: dict[
        FunctionRecord, tuple[SymbolicInt, tuple[int | sympy.Expr, ...], bool]
    ] = {}

    def make(
        name: str,
        registers: tuple[Register, ...] | None,
        count: SymbolicInt,
        values: tuple[int | sympy.Expr, ...],
        inverse: bool,
    ) -> ProgramRecord:
        ports = tuple(source_uniform_preparation_signature(count, len(values)))
        if registers is not None:
            # The source codec drops integer assumptions from dimension symbols.
            width = (
                source_uniform_preparation_signature(count).get_left("target").bitsize
                if count != 1
                else 0
            )
            normalized = (
                width.xreplace({sympy.floor(count): count})
                if isinstance(width, sympy.Expr)
                else width
            )
            integer_ports = tuple(Signature.build(ctrl=len(values), target=normalized))
            if registers not in (ports, integer_ports):
                raise SerializationError(
                    "Invalid uniform preparation register signature"
                )
            ports = registers
        if (
            type(count) is not int
            or count == 1
            or any(not isinstance(value, int) for value in values)
        ):
            record = ProgramRecord((FunctionRecord(name, (), registers=ports),))
        else:
            record = _rename_program(
                uniform_preparation_program(
                    count, tuple(int(value) for value in values), adjoint=inverse
                ),
                name,
            )
        recipes[record.functions[0]] = count, values, inverse
        return record

    def forward(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if "n" not in arguments or arguments.keys() - {"n", "cvs"}:
            raise SerializationError("Uniform preparation requires n and optional cvs")
        count = _integer(arguments, "n")
        raw = arguments.get("cvs", ())
        if isinstance(raw, np.ndarray):
            if raw.ndim != 1:
                raise SerializationError("Uniform controls require a vector")
            raw = tuple(raw)
        if not isinstance(raw, (tuple, list)):
            raise SerializationError("Uniform controls require a value sequence")
        values = tuple(_control_value(value) for value in raw)
        return make(name, registers, count, values, False)

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            set(arguments) != {"subbloq"}
            or not isinstance(child, FunctionRecord)
            or child not in recipes
        ):
            if adjoint_factory is not None:
                return adjoint_factory(name, registers, arguments)
            raise SerializationError(
                "Missing exact uniform adjoint constructor context"
            )
        count, values, inverse = recipes[child]
        return make(name, registers, count, values, not inverse)

    return {
        "qualtran.bloqs.state_preparation.prepare_uniform_superposition.PrepareUniformSuperposition": forward,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
