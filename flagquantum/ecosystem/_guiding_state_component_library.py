# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor follows locked Qualtran v0.7.0 kikuchi_guiding_state.py.

"""Translate the serializable uncompute leaf without inventing an instance codec."""

from collections.abc import Mapping

import sympy

from ..algorithms.kikuchi_guiding_state import ProbabilisticUncompute
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_word_factories import _rename_program
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.registers import Register
from ..errors import SerializationError


def _uncompute(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    width = arguments.get("bitsize")
    if arguments.keys() != {"bitsize"} or (
        type(width) is not int and not isinstance(width, sympy.Expr)
    ):
        raise SerializationError(
            "Probabilistic uncompute requires integer or symbolic bitsize"
        )
    assert isinstance(width, (int, sympy.Expr))
    if isinstance(width, sympy.Expr) and not width.free_symbols and width.is_Integer:
        width = int(width)
    oracle = ProbabilisticUncompute(width)
    ports = tuple(oracle.signature)
    if registers is not None and registers != ports:
        raise SerializationError("Invalid probabilistic uncompute signature")
    if type(width) is int and width > 0:
        return _rename_program(oracle.program(), name)
    if type(width) is int and width < 0:
        raise SerializationError("Probabilistic uncompute width must be nonnegative")
    # Zero has source metadata but its decomposition lacks the removed q port.
    # Symbolic dimensions retain declarations, not a made-up finite body.
    return FunctionRecord(name, (), registers=ports)


def guiding_state_library_factories() -> dict[str, ComponentFactory]:
    """Resolve the original serializable leaf through its shared Algorithms owner."""
    return {
        "qualtran.bloqs.optimization.k_xor_sat.kikuchi_guiding_state.ProbabilisticUncompute": _uncompute
    }
