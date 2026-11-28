# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 mod_arithmetic/mod_division.py.

"""Resolve the complete Kaliski constructor chain through shared graph owners."""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Protocol

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import montgomery_register_type, quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...errors import SerializationError
from ..registers.bits.bitwise_component import bitwise_word_program
from ..registers.comparison._component_gates import comparison_bit_functions
from ..registers.component_adjoint import adjoint_component_calls
from ..registers.modular.inversion.inverse_algorithm_component import (
    inverse_algorithm_component_program,
    inverse_algorithm_program,
)
from ..registers.modular.inversion.inverse_arithmetic_component import (
    inverse_arithmetic_program,
)
from ..registers.modular.inversion.inverse_exchange_component import (
    inverse_exchange_program,
)
from ..registers.modular.inversion.inverse_iteration_component import (
    inverse_iteration_component_program,
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
from ..registers.modular.modular_inverse_component import (
    modular_inverse_component_program,
    modular_inverse_program,
)
from ._inverse_iteration_library import inverse_ports
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_constant_control_factories import (
    unresolved_constant_addition_program,
    unresolved_xor_constant_program,
)
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program
from .montgomery_port_views import restore_montgomery_ports, source_modulus_omission

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


class _SimpleBuilder(Protocol):
    def __call__(self, width: int, *, adjoint: bool = False) -> ProgramRecord: ...


_SIMPLE_BUILDERS: dict[str, _SimpleBuilder] = {
    "_KaliskiIterationStep1": inverse_termination_program,
    "_KaliskiIterationStep2": inverse_parity_program,
    "_KaliskiIterationStep3": inverse_ordering_program,
    "_KaliskiIterationStep4": inverse_exchange_program,
    "_KaliskiIterationStep5": inverse_arithmetic_program,
}


_SIMPLE_ADJOINT_BUILDERS: dict[str, Callable[[int], ProgramRecord]] = {
    kind: partial(builder, adjoint=True) for kind, builder in _SIMPLE_BUILDERS.items()
}


def _symbolic_inverse_program(
    kind: str, width: int, modulus: sympy.Expr, *, adjoint: bool
) -> ProgramRecord:
    steps = tuple(builder(width) for builder in _SIMPLE_BUILDERS.values())
    # Only the actual sixth step rejects a symbolic modulus. Retain every
    # available parent and the complete accepted first-five-step owners.
    scaling = ProgramRecord(
        (
            FunctionRecord(
                "inverse_scaling",
                (),
                registers=inverse_ports("_KaliskiIterationStep6", width, modulus),
            ),
        )
    )
    iteration = inverse_iteration_component_program(
        montgomery_register_type(width, modulus), (*steps, scaling)
    )
    forward_iteration = iteration
    if adjoint:
        inverse_steps = tuple(
            builder(width) for builder in _SIMPLE_ADJOINT_BUILDERS.values()
        )
        iteration = adjoint_component_calls(
            iteration,
            (
                *zip(steps, inverse_steps, strict=True),
                (scaling, _rename_program(scaling, "inverse_scaling_adjoint")),
            ),
        )
    if kind == "_KaliskiIteration":
        return iteration
    word = montgomery_register_type(width)
    modulus_load = unresolved_xor_constant_program(word, modulus, name="load_modulus")
    unit_load = bitwise_word_program(word, "xor_constant", pattern=1)
    complement = bitwise_word_program(word, "bitwise_not")
    correction = unresolved_constant_addition_program(
        word, modulus + 1, name="correct_residue"
    )
    controlled_modulus = unresolved_xor_constant_program(
        word, modulus, name="controlled_modulus", controlled=True
    )
    controlled_predecessor = unresolved_xor_constant_program(
        word, modulus - 1, name="controlled_predecessor", controlled=True
    )
    algorithm = inverse_algorithm_component_program(
        width,
        modulus,
        forward_iteration,
        modulus_load,
        unit_load,
        complement,
        correction,
        controlled_modulus,
        controlled_predecessor,
    )
    if adjoint:
        x_gate = ProgramRecord((comparison_bit_functions()[0],))
        algorithm = adjoint_component_calls(
            algorithm,
            (
                (forward_iteration, iteration),
                (modulus_load, modulus_load),
                (unit_load, unit_load),
                (complement, complement),
                (
                    correction,
                    unresolved_constant_addition_program(
                        word, modulus + 1, name="correct_residue_inverse", adjoint=True
                    ),
                ),
                (controlled_modulus, controlled_modulus),
                (controlled_predecessor, controlled_predecessor),
                (x_gate, x_gate),
            ),
        )
    if kind == "_KaliskiModInverseImpl":
        return algorithm
    return modular_inverse_component_program(width, modulus, algorithm, adjoint=adjoint)


def _inverse_factory(
    kind: str,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
    *,
    adjoint: bool = False,
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
    inverse = bool(direction) ^ adjoint
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
        if len(registers) != len(ports) or not all(
            outer == inner or source_modulus_omission(outer, inner)
            for outer, inner in zip(registers, ports, strict=True)
        ):
            raise SerializationError(f"Invalid register signature for {kind}")
    if (
        type(width) is int
        and isinstance(modulus, sympy.Expr)
        and modulus.free_symbols
        and kind in {"_KaliskiIteration", "_KaliskiModInverseImpl", "KaliskiModInverse"}
    ):
        program = _symbolic_inverse_program(kind, width, modulus, adjoint=inverse)
        if registers is not None and registers != ports:
            program = restore_montgomery_ports(program, registers)
        return _rename_program(program, name)
    if isinstance(width, sympy.Expr) or isinstance(modulus, sympy.Expr):
        return FunctionRecord(
            name, (), registers=ports if registers is None else registers
        )
    assert isinstance(width, int)
    if kind in _SIMPLE_BUILDERS:
        program = (_SIMPLE_ADJOINT_BUILDERS if inverse else _SIMPLE_BUILDERS)[kind](
            width
        )
    else:
        assert isinstance(modulus, int)
        if kind == "_KaliskiIterationStep6":
            program = inverse_scaling_program(width, modulus, adjoint=inverse)
        elif kind == "_KaliskiIteration":
            program = inverse_iteration_program(width, modulus, adjoint=inverse)
        elif kind == "_KaliskiModInverseImpl":
            program = inverse_algorithm_program(width, modulus, adjoint=inverse)
        else:
            program = modular_inverse_program(width, modulus, adjoint=inverse)
    if registers is not None and registers != ports:
        program = restore_montgomery_ports(program, registers)
    return _rename_program(program, name)


def modular_inverse_factories(
    *, adjoint_factory: ComponentFactory | None = None
) -> dict[str, ComponentFactory]:
    """Keep original inverse constructors and exact adjoints in one compilation."""
    recipes: dict[FunctionRecord, tuple[str, dict[str, ResolvedArgument], bool]] = {}

    def make(
        kind: str,
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
        inverse: bool = False,
    ) -> FunctionRecord | ProgramRecord:
        result = _inverse_factory(kind, name, registers, arguments, adjoint=inverse)
        root = result.functions[0] if isinstance(result, ProgramRecord) else result
        recipes[root] = kind, dict(arguments), inverse
        return result

    def inverse(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            set(arguments) == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in recipes
        ):
            kind, values, direction = recipes[child]
            return make(kind, name, registers, values, not direction)
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing exact Kaliski adjoint constructor context")

    return {
        **{
            f"qualtran.bloqs.mod_arithmetic.mod_division.{kind}": partial(make, kind)
            for kind in (*_STEP_PORTS, "KaliskiModInverse")
        },
        "qualtran._infra.adjoint.Adjoint": inverse,
    }
