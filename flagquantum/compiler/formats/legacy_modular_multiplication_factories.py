# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 mod_arithmetic/mod_multiplication.py.

"""Resolve modular product constructors through their existing graph owners."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
    signed_register_type,
    unsigned_register_type,
)
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...errors import SerializationError
from ..registers.comparison._component_gates import comparison_bit_functions
from ..registers.component_adjoint import adjoint_component_calls
from ..registers.component_control import ControlAdder, _stored_control_adder
from ..registers.modular.controlled_doubling_component import (
    controlled_modular_doubling_program,
)
from ..registers.modular.modular_doubling_component import (
    modular_doubling_component_program,
    modular_doubling_program,
)
from ..registers.modular.modular_multiplication_component import (
    modular_multiplication_program,
)
from ..registers.modular.montgomery_multiplication_component import (
    montgomery_multiplication_program,
)
from ..registers.modular.montgomery_signature import montgomery_product_signature
from ..registers.modular.single_window_component import single_window_program
from ..registers.modular.windowed_multiplication_component import (
    windowed_multiplication_program,
)
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_constant_control_factories import unresolved_constant_addition_program
from .legacy_controlled_factories import controlled_via_and_factories
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program
from .montgomery_port_views import restore_montgomery_ports, source_modulus_omission

_Kind = Literal[
    "ModDbl",
    "CModMulK",
    "SingleWindowModMul",
    "_DirtyOutOfPlaceMontgomeryModMulImpl",
    "DirtyOutOfPlaceMontgomeryModMul",
]


def symbolic_modular_doubling_program(
    dtype: RegisterType, modulus: sympy.Expr, *, adjoint: bool = False
) -> ProgramRecord:
    """Preserve the original available source root and genuine constant refusals."""
    width = dtype.num_qubits
    assert type(width) is int
    signed, unsigned = signed_register_type(width + 2), unsigned_register_type(
        width + 1
    )
    subtraction = unresolved_constant_addition_program(
        signed, -modulus, name="subtract_modulus"
    )
    correction = unresolved_constant_addition_program(
        unsigned, modulus, name="restore_modulus", controlled=True
    )
    record = modular_doubling_component_program(dtype, subtraction, correction)
    if not adjoint:
        return record
    gates = tuple(ProgramRecord((f,)) for f in comparison_bit_functions())
    return adjoint_component_calls(
        record,
        (
            (
                subtraction,
                unresolved_constant_addition_program(
                    signed, -modulus, name="subtract_modulus_inverse", adjoint=True
                ),
            ),
            (
                correction,
                unresolved_constant_addition_program(
                    unsigned,
                    modulus,
                    name="restore_modulus_inverse",
                    adjoint=True,
                    controlled=True,
                ),
            ),
            *((gate, gate) for gate in gates),
        ),
    )


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
            ports = list(
                montgomery_product_signature(width, window, modulus, adjoint=inverse)
            )
            if kind == "_DirtyOutOfPlaceMontgomeryModMulImpl":
                ports = [
                    Register(
                        register.name,
                        register.dtype,
                        register.shape_symbolic,
                        side=RegisterSide.THRU,
                    )
                    for register in ports
                ]
    signature = tuple(ports)
    if registers is not None and registers != signature:
        if (
            word
            or len(registers) != len(signature)
            or not all(
                outer == inner or source_modulus_omission(outer, inner)
                for outer, inner in zip(registers, signature, strict=True)
            )
        ):
            raise SerializationError(f"Invalid register signature for {kind}")
    if (
        kind == "ModDbl"
        and type(width) is int
        and isinstance(modulus, sympy.Expr)
        and modulus.free_symbols
    ):
        return _rename_program(symbolic_modular_doubling_program(dtype, modulus), name)
    if any(isinstance(item, sympy.Expr) for item in (width, window, modulus, value)):
        return FunctionRecord(
            name, (), registers=signature if registers is None else registers
        )
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
    if registers is not None and registers != signature:
        program = restore_montgomery_ports(program, registers)
    return _rename_program(program, name)


def modular_multiplication_factories(
    *,
    adjoint_factory: ComponentFactory | None = None,
    controlled_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Keep exact doubling adjoints in a compilation-local constructor context."""
    kinds: tuple[_Kind, ...] = (
        "ModDbl",
        "CModMulK",
        "SingleWindowModMul",
        "_DirtyOutOfPlaceMontgomeryModMulImpl",
        "DirtyOutOfPlaceMontgomeryModMul",
    )
    recipes: dict[FunctionRecord, tuple[dict[str, ResolvedArgument], bool]] = {}

    def make(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
        inverse: bool,
    ) -> FunctionRecord | ProgramRecord:
        result = _modular_factory("ModDbl", name, registers, arguments)
        if inverse and isinstance(result, ProgramRecord):
            dtype = arguments["dtype"]
            modulus = _integer(arguments, "mod")
            assert isinstance(dtype, RegisterType)
            result = _rename_program(
                (
                    symbolic_modular_doubling_program(dtype, modulus, adjoint=True)
                    if isinstance(modulus, sympy.Expr)
                    else modular_doubling_program(dtype, modulus, adjoint=True)
                ),
                name,
            )
        root = result.functions[0] if isinstance(result, ProgramRecord) else result
        recipes[root] = dict(arguments), inverse
        return result

    def forward(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        return make(name, registers, arguments, False)

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
            values, direction = recipes[child]
            return make(name, registers, values, not direction)
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing exact modular doubling adjoint context")

    def controlled(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        spec = arguments.get("ctrl_spec")
        if isinstance(child, FunctionRecord) and child in recipes:
            values, direction = recipes[child]
            dtype = values["dtype"]
            modulus = _integer(values, "mod")
            if (
                set(arguments) == {"subbloq", "ctrl_spec"}
                and isinstance(spec, ControlSpecification)
                and len(spec.clauses) == 1
                and spec.clauses[0][0] == quantum_bit_type()
                and spec.clauses[0][1].shape == ()
                and spec.clauses[0][1].item() == 1
                and isinstance(dtype, RegisterType)
                and type(dtype.num_qubits) is int
                and (
                    type(modulus) is int
                    or (isinstance(modulus, sympy.Expr) and modulus.free_symbols)
                )
            ):
                if isinstance(modulus, sympy.Expr):
                    from .legacy_symbolic_modular_controls import (
                        symbolic_controlled_modular_doubling_program,
                    )

                    result = symbolic_controlled_modular_doubling_program(
                        dtype, modulus, adjoint=direction
                    )
                else:
                    result = controlled_modular_doubling_program(
                        dtype, modulus, adjoint=direction
                    )
                if registers is not None and registers != result.functions[0].registers:
                    result = restore_montgomery_ports(result, registers)
                return _rename_program(result, name)
        if controlled_factory is not None:
            return controlled_factory(name, registers, arguments)
        raise SerializationError(
            "Missing controlled modular doubling constructor context"
        )

    def positive(child: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        if child not in recipes:
            raise SerializationError("Missing modular doubling predicate context")
        values, direction = recipes[child]
        dtype, modulus = values["dtype"], _integer(values, "mod")
        if not isinstance(dtype, RegisterType):
            raise SerializationError("Controlled doubling requires its source dtype")
        if isinstance(modulus, sympy.Expr):
            from .legacy_symbolic_modular_controls import (
                symbolic_controlled_modular_doubling_program,
            )

            program = symbolic_controlled_modular_doubling_program(
                dtype, modulus, adjoint=direction
            )
        else:
            program = controlled_modular_doubling_program(
                dtype, modulus, adjoint=direction
            )
        module = _rename_program(
            program,
            child.name + "_positive",
        )
        return module, _stored_control_adder(module.functions[0])

    return {
        **{
            f"qualtran.bloqs.mod_arithmetic.mod_multiplication.{kind}": partial(
                _modular_factory, kind
            )
            for kind in kinds
        },
        "qualtran.bloqs.mod_arithmetic.mod_multiplication.ModDbl": forward,
        "qualtran._infra.adjoint.Adjoint": inverse,
        "qualtran._infra.controlled.Controlled": controlled,
        **controlled_via_and_factories(control_system=positive),
    }
