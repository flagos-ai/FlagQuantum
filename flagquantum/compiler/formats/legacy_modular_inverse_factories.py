# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 mod_arithmetic/mod_division.py.

"""Resolve the complete Kaliski constructor chain through shared graph owners."""

from collections.abc import Callable, Mapping
from functools import partial

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import montgomery_register_type, quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...errors import SerializationError
from ..registers.modular.inversion.inverse_algorithm_component import (
    inverse_algorithm_program,
)
from ..registers.modular.inversion.inverse_arithmetic_component import (
    inverse_arithmetic_program,
)
from ..registers.modular.inversion.inverse_exchange_component import (
    inverse_exchange_program,
)
from ..registers.modular.inversion.inverse_iteration_component import (
    inverse_iteration_program,
)
from ..registers.modular.inversion.inverse_ordering_component import (
    inverse_ordering_program,
)
from ..registers.modular.inversion.inverse_parity_component import (
    inverse_parity_program,
)
from ..registers.modular.inversion.inverse_scaling_component import (
    inverse_scaling_program,
)
from ..registers.modular.inversion.inverse_termination_component import (
    inverse_termination_program,
)
from ..registers.modular.modular_inverse_component import modular_inverse_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program

# Ordered original signatures are independent of finite decomposition.
_STEP_PORTS = {
    "_KaliskiIterationStep1": ("v", "m", "f", "is_terminal"),
    "_KaliskiIterationStep2": ("u", "v", "b", "a", "m", "f"),
    "_KaliskiIterationStep3": ("u", "v", "b", "a", "m", "f"),
    "_KaliskiIterationStep4": ("u", "v", "r", "s", "a"),
    "_KaliskiIterationStep5": ("u", "v", "r", "s", "b", "f"),
    "_KaliskiIterationStep6": ("u", "v", "r", "s", "b", "a", "m", "f"),
    "_KaliskiIteration": ("u", "v", "r", "s", "m", "f", "is_terminal"),
    "_KaliskiModInverseImpl": ("u", "v", "r", "s", "m", "f", "terminal_condition"),
}
_SIMPLE_BUILDERS: dict[str, Callable[[int], ProgramRecord]] = {
    "_KaliskiIterationStep1": inverse_termination_program,
    "_KaliskiIterationStep2": inverse_parity_program,
    "_KaliskiIterationStep3": inverse_ordering_program,
    "_KaliskiIterationStep4": inverse_exchange_program,
    "_KaliskiIterationStep5": inverse_arithmetic_program,
}


def _inverse_factory(
    kind: str,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    required = {"bitsize"} if kind in _SIMPLE_BUILDERS else {"bitsize", "mod"}
    optional = {"uncompute"} if kind == "KaliskiModInverse" else set()
    if not required <= arguments.keys() or arguments.keys() - required - optional:
        raise SerializationError(f"Invalid constructor arguments for {kind}")
    width = _integer(arguments, "bitsize")
    modulus = _integer(arguments, "mod") if "mod" in required else None
    direction = arguments.get("uncompute", False)
    if type(direction) not in (bool, int) or direction not in (0, 1):
        raise SerializationError("Kaliski direction requires a boolean")
    inverse = bool(direction)
    word = montgomery_register_type(width, modulus)
    ports: tuple[Register, ...]
    if kind == "KaliskiModInverse":
        side = RegisterSide.LEFT if inverse else RegisterSide.RIGHT
        ports = (
            Register("x", word),
            Register("junk", RegisterType("opaque", 4 * width), side=side),
        )
    else:
        histories = (
            {"m", "terminal_condition"} if kind == "_KaliskiModInverseImpl" else set()
        )
        ports = tuple(
            Register(
                port,
                (
                    word
                    if port in {"u", "v", "r", "s"}
                    else (
                        RegisterType("opaque", 2 * width)
                        if port in histories
                        else quantum_bit_type()
                    )
                ),
            )
            for port in _STEP_PORTS[kind]
        )
    if registers is not None and registers != ports:
        raise SerializationError(f"Invalid register signature for {kind}")
    if isinstance(width, sympy.Expr) or isinstance(modulus, sympy.Expr):
        return FunctionRecord(name, (), registers=ports)
    assert isinstance(width, int)
    if kind in _SIMPLE_BUILDERS:
        program = _SIMPLE_BUILDERS[kind](width)
    else:
        assert isinstance(modulus, int)
        if kind == "_KaliskiIterationStep6":
            program = inverse_scaling_program(width, modulus)
        elif kind == "_KaliskiIteration":
            program = inverse_iteration_program(width, modulus)
        elif kind == "_KaliskiModInverseImpl":
            program = inverse_algorithm_program(width, modulus)
        else:
            program = modular_inverse_program(width, modulus, adjoint=inverse)
    return _rename_program(program, name)


def modular_inverse_factories() -> dict[str, ComponentFactory]:
    """Supply all six steps, iteration, algorithm and directional wrapper."""
    return {
        f"qualtran.bloqs.mod_arithmetic.mod_division.{kind}": partial(
            _inverse_factory, kind
        )
        for kind in (*_STEP_PORTS, "KaliskiModInverse")
    }
