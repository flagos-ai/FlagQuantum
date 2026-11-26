# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation follows locked Qualtran v0.7.0 prepare_hubbard.py.

"""Resolve Hubbard PREPARE using injected coefficient conversion and typed calls."""

from collections.abc import Callable, Mapping

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...core.parameters import Parameter, ParameterExpression
from ...core.value_math import SymbolicReal
from ...errors import SerializationError
from ..registers.layout.hubbard_signatures import hubbard_preparation_signature
from ..registers.preparation.hubbard_prepare_component import hubbard_prepare_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_rotation_factories import _real_argument
from .legacy_word_factories import _rename_program

_Angle = float | Parameter | ParameterExpression
_Angles = tuple[_Angle, _Angle]
PreparationAngles = Callable[[int, SymbolicReal, SymbolicReal], _Angles | None]
_Recipe = tuple[int, _Angles | None, bool]
PreparationPairObserver = Callable[
    [FunctionRecord, tuple[ProgramRecord, ProgramRecord]], None
]


def hubbard_prepare_factories(
    *,
    angles: PreparationAngles,
    adjoint_factory: ComponentFactory | None = None,
    on_pair: PreparationPairObserver | None = None,
) -> dict[str, ComponentFactory]:
    """Keep coefficient conversion outside Compiler and inverse recipes scoped."""
    recipes: dict[FunctionRecord, _Recipe] = {}

    def body(recipe: _Recipe) -> ProgramRecord:
        dimension, values, inverse = recipe
        if dimension == 1 or values is None:
            return ProgramRecord(
                (
                    FunctionRecord(
                        "main",
                        (),
                        registers=tuple(hubbard_preparation_signature(dimension)),
                    ),
                )
            )
        return hubbard_prepare_program(dimension, *values, adjoint=inverse)

    def make(
        name: str, registers: tuple[Register, ...] | None, recipe: _Recipe
    ) -> ProgramRecord:
        record = _rename_program(body(recipe), name)
        root = record.functions[0]
        if registers is not None and registers != root.registers:
            raise SerializationError("Invalid Hubbard PREPARE register signature")
        recipes[root] = recipe
        if on_pair is not None:
            dimension, values, inverse = recipe
            on_pair(root, (record, body((dimension, values, not inverse))))
        return record

    def prepare(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if arguments.keys() != {"x_dim", "y_dim", "t", "u"}:
            raise SerializationError(
                "Hubbard preparation requires dimensions and weights"
            )
        x, y = (_integer(arguments, key) for key in ("x_dim", "y_dim"))
        if type(x) is not int or type(y) is not int or x < 1 or x != y:
            raise SerializationError(
                "Hubbard example requires a positive concrete square"
            )
        assert isinstance(x, int)

        def coefficient(key: str) -> SymbolicReal:
            checked = _real_argument(arguments, key)
            raw = arguments[key]
            # Validation must not round exact integer coefficients before the
            # source's multiplication and floor normalization.
            return raw if isinstance(raw, int) else checked

        t, u = (coefficient(key) for key in ("t", "u"))
        values = None if x == 1 else angles(x, t, u)
        return make(name, registers, (x, values, False))

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            arguments.keys() == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in recipes
        ):
            dimension, values, direction = recipes[child]
            return make(name, registers, (dimension, values, not direction))
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing exact Hubbard PREPARE adjoint context")

    return {
        "qualtran.bloqs.chemistry.hubbard_model.qubitization.prepare_hubbard.PrepareHubbard": prepare,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
