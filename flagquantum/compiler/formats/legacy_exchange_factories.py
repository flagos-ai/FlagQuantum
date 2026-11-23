# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 identity.py and swap.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/swap.py

"""Resolve identity and word-exchange libraries through existing native owners."""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Literal

import sympy

from ...core.ir import CircuitIR
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ..registers.component_gate_function import bit_gate_function
from ..registers.component_graph import _namespace_helpers
from ..registers.primitives.controlled_swap import controlled_swap_bit_ir
from ..registers.primitives.identity import (
    register_identity_ir,
    register_identity_program,
)
from ..registers.primitives.swap import swap_bit_ir
from ..registers.primitives.swap_component import SwapDeclaration
from .legacy_component_resolution import ComponentFactory, ResolvedArgument

_Kind = Literal["Identity", "TwoBitSwap", "TwoBitCSwap", "Swap", "CSwap"]


def _exchange_factory(
    kind: _Kind,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    atomic_swap = kind in ("TwoBitSwap", "TwoBitCSwap")
    allowed = set() if atomic_swap else {"bitsize"}
    if arguments.keys() - allowed:
        raise SerializationError(f"Unexpected constructor arguments for {kind}")
    if not atomic_swap and kind != "Identity" and "bitsize" not in arguments:
        raise SerializationError(f"{kind} requires bitsize")
    width = arguments.get("bitsize", 1)
    if type(width) is not int and not isinstance(width, sympy.Expr):
        raise SerializationError(f"{kind} requires an integer or symbolic bitsize")
    controlled = kind in ("TwoBitCSwap", "CSwap")
    signature = tuple(
        Signature.build(q=width)
        if kind == "Identity"
        else SwapDeclaration(width, controlled).signature
    )
    if registers is not None and registers != signature:
        raise SerializationError(f"Invalid register signature for {kind}")
    if isinstance(width, sympy.Expr):
        return FunctionRecord(name, (), registers=signature)
    if kind == "Identity":
        if width == 1:
            return bit_gate_function(
                name, signature, CircuitIR(1, register_identity_ir((0,)))
            )
        program = register_identity_program(width)
    elif atomic_swap:
        gates = controlled_swap_bit_ir(0, 1, 2) if controlled else swap_bit_ir(0, 1)
        return bit_gate_function(name, signature, CircuitIR(len(signature), gates))
    else:
        program = SwapDeclaration(width, controlled).program()
    # The resolver assigns root names. All nested helper definitions and calls
    # must share that namespace when several libraries reuse word primitives.
    program = replace(
        program,
        functions=(replace(program.functions[0], name=name), *program.functions[1:]),
    )
    return _namespace_helpers(program, name)


def exchange_factories() -> dict[str, ComponentFactory]:
    """Supply independent explicit identity/exchange constructor mappings.

    Serialized names identify interchange declarations. Native owners retain
    physical width admission, MSB ordering and exact controlled-swap synthesis.
    """
    kinds: tuple[_Kind, ...] = (
        "Identity",
        "TwoBitSwap",
        "TwoBitCSwap",
        "Swap",
        "CSwap",
    )
    return {
        f"qualtran.bloqs.basic_gates.{'identity' if kind == 'Identity' else 'swap'}.{kind}": partial(
            _exchange_factory, kind
        )
        for kind in kinds
    }
