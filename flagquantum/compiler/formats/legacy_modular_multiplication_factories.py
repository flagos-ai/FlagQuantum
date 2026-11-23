# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 mod_arithmetic/mod_multiplication.py.

"""Resolve modular product constructors through their existing graph owners."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import montgomery_register_type, quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...errors import SerializationError
from ..registers.modular.modular_doubling_component import modular_doubling_program
from ..registers.modular.modular_multiplication_component import (
    modular_multiplication_program,
)
from ..registers.modular.montgomery_multiplication_component import (
    montgomery_multiplication_program,
)
from ..registers.modular.single_window_component import single_window_program
from ..registers.modular.windowed_multiplication_component import (
    windowed_multiplication_program,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program

_Kind = Literal[
    "ModDbl",
    "CModMulK",
    "SingleWindowModMul",
    "_DirtyOutOfPlaceMontgomeryModMulImpl",
    "DirtyOutOfPlaceMontgomeryModMul",
]


def _modular_factory(
    kind: _Kind,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    word = kind in {"ModDbl", "CModMulK"}
    required = {"dtype", "mod"} if word else {"bitsize", "window_size", "mod"}
    if kind == "CModMulK":
        required = required | {"k"}
    optional = {"uncompute"} if kind == "DirtyOutOfPlaceMontgomeryModMul" else set()
    if not required <= arguments.keys() or arguments.keys() - required - optional:
        raise SerializationError(f"Invalid constructor arguments for {kind}")
    modulus = _integer(arguments, "mod")
    direction = arguments.get("uncompute", False)
    if type(direction) not in (bool, int) or direction not in (0, 1):
        raise SerializationError("Montgomery direction requires a boolean")
    inverse = bool(direction)
    if word:
        dtype = arguments["dtype"]
        if not isinstance(dtype, RegisterType) or dtype.encoding not in {
            "uint",
            "montgomery",
        }:
            raise SerializationError(
                "Modular products require unsigned or Montgomery words"
            )
        width = dtype.num_qubits
        ports = [Register("ctrl", quantum_bit_type())] if kind == "CModMulK" else []
        ports.append(Register("x", dtype))
        value = _integer(arguments, "k") if kind == "CModMulK" else 0
        window = 1
    else:
        width = _integer(arguments, "bitsize")
        window = _integer(arguments, "window_size")
        dtype = montgomery_register_type(width, modulus)
        value = 0
        if kind == "SingleWindowModMul":
            ports = [
                Register("x", quantum_bit_type(), (window,)),
                Register("y", dtype),
                Register("target", quantum_bit_type(), (width + window,)),
                Register("qrom_index", montgomery_register_type(window, modulus)),
            ]
        else:
            side = RegisterSide.LEFT if inverse else RegisterSide.RIGHT
            if kind == "_DirtyOutOfPlaceMontgomeryModMulImpl":
                side = RegisterSide.THRU
            history_width = ((width + window - 1) // window) * window
            ports = [
                Register("x", dtype),
                Register("y", dtype),
                Register("target", dtype, side=side),
                Register(
                    "qrom_indices",
                    montgomery_register_type(history_width, modulus),
                    side=side,
                ),
                Register("reduced", quantum_bit_type(), side=side),
            ]
    signature = tuple(ports)
    if registers is not None and registers != signature:
        # The original wire codec omits Montgomery modulus metadata. Preserve
        # the source constructor signature instead of discarding this mismatch.
        raise SerializationError(f"Invalid register signature for {kind}")
    if any(isinstance(item, sympy.Expr) for item in (width, window, modulus, value)):
        return FunctionRecord(name, (), registers=signature)
    assert isinstance(modulus, int) and isinstance(width, int)
    assert isinstance(window, int) and isinstance(value, int)
    if kind == "ModDbl":
        program = modular_doubling_program(dtype, modulus)
    elif kind == "CModMulK":
        program = modular_multiplication_program(dtype, modulus, value)
    elif kind == "SingleWindowModMul":
        program = single_window_program(width, window, modulus)
    elif kind == "_DirtyOutOfPlaceMontgomeryModMulImpl":
        program = windowed_multiplication_program(width, window, modulus)
    else:
        program = montgomery_multiplication_program(
            width, window, modulus, adjoint=inverse
        )
    return _rename_program(program, name)


def modular_multiplication_factories() -> dict[str, ComponentFactory]:
    """Supply the complete modular product source family and its private core."""
    kinds: tuple[_Kind, ...] = (
        "ModDbl",
        "CModMulK",
        "SingleWindowModMul",
        "_DirtyOutOfPlaceMontgomeryModMulImpl",
        "DirtyOutOfPlaceMontgomeryModMul",
    )
    return {
        f"qualtran.bloqs.mod_arithmetic.mod_multiplication.{kind}": partial(
            _modular_factory, kind
        )
        for kind in kinds
    }
