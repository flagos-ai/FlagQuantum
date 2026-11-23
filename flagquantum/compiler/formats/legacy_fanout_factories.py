# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adapted from locked Qualtran v0.7.0 mcmt/multi_target_cnot.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/multi_target_cnot.py

"""Resolve the shared logarithmic-depth fanout constructor."""

from collections.abc import Mapping

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ..registers.bits.fanout_component import fanout_component_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_word_factories import _rename_program


def _fanout_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if set(arguments) != {"bitsize"}:
        raise SerializationError("MultiTargetCNOT requires only bitsize")
    width = arguments["bitsize"]
    if type(width) is not int and not isinstance(width, sympy.Expr):
        raise SerializationError("Fanout requires an integer or symbolic width")
    signature = tuple(Signature.build(control=1, targets=width))
    if registers is not None and registers != signature:
        raise SerializationError("Invalid register signature for fanout")
    if isinstance(width, sympy.Expr):
        return FunctionRecord(name, (), registers=signature)
    return _rename_program(fanout_component_program(width), name)


def fanout_factories() -> dict[str, ComponentFactory]:
    """Supply MultiTargetCNOT without adding scratch or a numerical owner."""
    return {"qualtran.bloqs.mcmt.multi_target_cnot.MultiTargetCNOT": _fanout_factory}
