# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow Qualtran v0.7.0 cryptography/ecc/ec_add.py.

"""Translate original curve-addition fields into explicit owned stage calls."""

from collections.abc import Mapping

from ..algorithms.primitives.elliptic_curve_addition_flags import (
    addition_flags_signature,
)
from ..algorithms.primitives.elliptic_curve_addition_slope import (
    addition_slope_signature,
)
from ..compiler.formats.legacy_basis_vector_factories import basis_vector_factories
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.formats.montgomery_port_views import restore_montgomery_ports
from ..compiler.registers.comparison.controlled_equality_component import (
    controlled_equality_program,
)
from ..compiler.registers.comparison.equality_component import equality_program
from ..compiler.registers.control.multi_control_component import multi_control_x_program
from ..compiler.registers.elliptic_curve.addition_flags_component import (
    addition_flags_program,
)
from ..compiler.registers.elliptic_curve.addition_slope_component import (
    addition_slope_program,
)
from ..compiler.registers.modular.controlled_modular_subtraction_component import (
    controlled_modular_subtraction_program,
)
from ..compiler.registers.modular.modular_inverse_component import (
    modular_inverse_program,
)
from ..compiler.registers.modular.modular_negation_component import (
    modular_negation_program,
)
from ..compiler.registers.modular.modular_subtraction_component import (
    modular_subtraction_program,
)
from ..compiler.registers.modular.montgomery_multiplication_component import (
    montgomery_multiplication_program,
)
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.register_types import montgomery_register_type
from ..core.ir.program.registers import Register
from ..errors import SerializationError


def _flags(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if set(arguments) != {"n", "mod"}:
        raise SerializationError(
            "Curve addition flags require original n and mod fields"
        )
    width, modulus = (_integer(arguments, key) for key in ("n", "mod"))
    signature = addition_flags_signature(width)
    if registers is not None and registers != tuple(signature):
        raise SerializationError("Invalid original curve addition flag signature")
    if type(width) is not int or type(modulus) is not int:
        return FunctionRecord(name, (), registers=tuple(signature))
    dtype = montgomery_register_type(width)
    zero = basis_vector_factories()["qualtran.bloqs.basic_gates.z_basis.ZeroState"](
        "zero", None, {}
    )
    zero_program = zero if isinstance(zero, ProgramRecord) else ProgramRecord((zero,))
    program = addition_flags_program(
        signature,
        zero_program,
        equality_program(dtype),
        modular_negation_program(dtype, modulus),
        multi_control_x_program((0,) * (2 * width)),
        multi_control_x_program((0, 0, 0)),
    )
    return _rename_program(program, name)


def elliptic_curve_addition_factories() -> dict[str, ComponentFactory]:
    """Resolve implemented stages without declaring the remaining file complete."""
    return {
        "qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepOne": _flags,
        "qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepTwo": _slope,
    }


def _without_modulus(program: ProgramRecord) -> ProgramRecord:
    registers = program.functions[0].registers
    assert registers is not None
    return restore_montgomery_ports(
        program,
        tuple(
            (
                Register(
                    register.name,
                    montgomery_register_type(register.bitsize),
                    register.shape_symbolic,
                    register.side,
                )
                if register.dtype.encoding == "montgomery"
                else register
            )
            for register in registers
        ),
    )


def _slope(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if not {"n", "mod"} <= arguments.keys() or arguments.keys() - {
        "n",
        "mod",
        "window_size",
    }:
        raise SerializationError(
            "Curve addition slope requires original n, mod and window_size fields"
        )
    width, modulus = (_integer(arguments, key) for key in ("n", "mod"))
    window = _integer({"window_size": arguments.get("window_size", 1)}, "window_size")
    signature = addition_slope_signature(width)
    if registers is not None and registers != tuple(signature):
        raise SerializationError("Invalid original curve addition slope signature")
    if type(width) is not int or type(modulus) is not int or type(window) is not int:
        return FunctionRecord(name, (), registers=tuple(signature))
    dtype = montgomery_register_type(width)
    zero = basis_vector_factories()["qualtran.bloqs.basic_gates.z_basis.IntState"](
        "zero", None, {"bitsize": width, "val": 0}
    )
    zero_program = zero if isinstance(zero, ProgramRecord) else ProgramRecord((zero,))
    program = addition_slope_program(
        signature,
        zero=zero_program,
        subtract=modular_subtraction_program(dtype, modulus),
        controlled_subtract=controlled_modular_subtraction_program(dtype, modulus),
        inverse=_without_modulus(modular_inverse_program(width, modulus)),
        multiply=_without_modulus(
            montgomery_multiplication_program(width, window, modulus)
        ),
        equals=equality_program(dtype),
        select_general=multi_control_x_program((0, 1, 1)),
        select_tangent=multi_control_x_program((1, 1, 1)),
        controlled_equals=controlled_equality_program(dtype, control_value=0),
        multiply_adjoint=_without_modulus(
            montgomery_multiplication_program(width, window, modulus, adjoint=True)
        ),
        inverse_adjoint=_without_modulus(
            modular_inverse_program(width, modulus, adjoint=True)
        ),
    )
    return _rename_program(program, name)
