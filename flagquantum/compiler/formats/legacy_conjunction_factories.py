# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 mcmt/and_bloq.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/and_bloq.py

"""Resolve conjunction constructors and their specialized adjoint dependency."""

from collections.abc import Mapping

import numpy as np
import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.component_graph import _namespace_helpers
from ..registers.control.conjunction_program import (
    and_component_program,
    conjunction_declaration,
    conjunction_program,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument


def _control_value(value: object) -> int | sympy.Expr:
    if isinstance(value, sympy.Expr) and value.free_symbols:
        return value
    if isinstance(value, (int, np.integer, np.bool_, sympy.Integer)) and value in (
        0,
        1,
    ):
        return int(value)
    raise SerializationError("Conjunction physical controls require binary integers")


def _checked_program(
    program: ProgramRecord, registers: tuple[Register, ...] | None
) -> ProgramRecord:
    if registers is not None and program.functions[0].registers != registers:
        raise SerializationError("Invalid register signature for conjunction")
    return _namespace_helpers(program, program.functions[0].name)


def conjunction_factories() -> dict[str, ComponentFactory]:
    """Supply And, MultiAnd and the latter's serialized Adjoint dependency.

    The catalog retains MultiAnd constructor recipes keyed by complete resolved
    functions, since referenced arguments supply a function without its helper
    definitions. It recreates specialized reverse ladders through their existing
    owner. Names alone cannot select a recipe. Unknown adjoint children refuse.
    Create a catalog per compilation to bound retained constructor context.
    """
    recipes: dict[FunctionRecord, tuple[tuple[int | sympy.Expr, ...], bool]] = {}

    def multi_program(
        name: str,
        registers: tuple[Register, ...] | None,
        values: tuple[int | sympy.Expr, ...],
        uncompute: bool,
    ) -> ProgramRecord:
        if len(values) < 3:
            raise SerializationError("MultiAnd requires at least three controls")
        if all(isinstance(value, int) for value in values):
            program = conjunction_program(
                tuple(int(value) for value in values), name=name, uncompute=uncompute
            )
        else:
            program = conjunction_declaration(
                len(values), name=name, uncompute=uncompute
            )
        program = _checked_program(program, registers)
        root = program.functions[0]
        recipe = (values, uncompute)
        if root in recipes and recipes[root] != recipe:
            raise SerializationError("Ambiguous conjunction constructor context")
        recipes[root] = recipe
        return program

    def and_factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if arguments.keys() - {"cv1", "cv2", "uncompute"}:
            raise SerializationError("Unexpected constructor arguments for And")
        values = tuple(_control_value(arguments.get(key, 1)) for key in ("cv1", "cv2"))
        direction = arguments.get("uncompute", 0)
        if not isinstance(direction, (int, bool)) or direction not in (0, 1):
            raise SerializationError("And uncompute requires a boolean direction")
        uncompute = bool(direction)
        if all(isinstance(value, int) for value in values):
            left, right = values
            program = and_component_program(
                (int(left), int(right)), name=name, uncompute=uncompute
            )
        else:
            program = conjunction_declaration(2, name=name, uncompute=uncompute)
        return _checked_program(program, registers)

    def multi_factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if set(arguments) != {"cvs"}:
            raise SerializationError("MultiAnd requires only cvs")
        controls = arguments["cvs"]
        if isinstance(controls, np.ndarray):
            if controls.ndim != 1:
                raise SerializationError(
                    "MultiAnd cvs requires a one-dimensional array"
                )
            controls = tuple(controls)
        if not isinstance(controls, (tuple, list)):
            raise SerializationError("MultiAnd cvs requires a control sequence")
        values = tuple(_control_value(value) for value in controls)
        return multi_program(name, registers, values, False)

    def adjoint_factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if set(arguments) != {"subbloq"}:
            raise SerializationError("Adjoint requires only subbloq")
        child = arguments["subbloq"]
        if not isinstance(child, FunctionRecord) or child not in recipes:
            raise SerializationError(
                "Missing exact conjunction adjoint constructor context"
            )
        values, uncompute = recipes[child]
        return multi_program(name, registers, values, not uncompute)

    return {
        "qualtran.bloqs.mcmt.and_bloq.And": and_factory,
        "qualtran.bloqs.mcmt.and_bloq.MultiAnd": multi_factory,
        "qualtran._infra.adjoint.Adjoint": adjoint_factory,
    }
