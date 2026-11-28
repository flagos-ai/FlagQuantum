# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow Qualtran v0.7.0 cryptography/ecc/ec_add.py.

"""Translate original curve-addition fields into explicit owned stage calls."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

import numpy as np

from ..algorithms.primitives.elliptic_curve_addition import (
    elliptic_curve_addition_signature,
)
from ..algorithms.primitives.elliptic_curve_addition_cleanup import (
    addition_cleanup_signature,
)
from ..algorithms.primitives.elliptic_curve_addition_coordinates import (
    addition_coordinates_signature,
)
from ..algorithms.primitives.elliptic_curve_addition_finish import (
    addition_finish_signature,
)
from ..algorithms.primitives.elliptic_curve_addition_flags import (
    addition_flags_signature,
)
from ..algorithms.primitives.elliptic_curve_addition_products import (
    addition_products_signature,
)
from ..algorithms.primitives.elliptic_curve_addition_slope import (
    addition_slope_signature,
)
from ..compiler.formats.legacy_basis_vector_factories import basis_vector_factories
from ..compiler.formats.legacy_component_arguments import ControlSpecification
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_controlled_factories import controlled_via_and_factories
from ..compiler.formats.legacy_fixed_gate_factories import fixed_gate_factories
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_modular_subtraction_factories import (
    modular_subtraction_factories,
)
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.registers.bits.bitwise_component import bitwise_word_program
from ..compiler.registers.bits.bitwise_component_control import (
    controlled_bitwise_word_program,
)
from ..compiler.registers.bits.fanout_component import fanout_component_program
from ..compiler.registers.comparison.controlled_equality_component import (
    controlled_equality_program,
)
from ..compiler.registers.comparison.equality_component import equality_program
from ..compiler.registers.component_control import ControlAdder, _stored_control_adder
from ..compiler.registers.control.conjunction_program import conjunction_program
from ..compiler.registers.control.multi_control_component import multi_control_x_program
from ..compiler.registers.elliptic_curve.addition_cleanup_component import (
    addition_cleanup_program,
)
from ..compiler.registers.elliptic_curve.addition_component import (
    elliptic_curve_addition_program,
)
from ..compiler.registers.elliptic_curve.addition_coordinates_component import (
    addition_coordinates_program,
)
from ..compiler.registers.elliptic_curve.addition_finish_component import (
    addition_finish_program,
)
from ..compiler.registers.elliptic_curve.addition_flags_component import (
    addition_flags_program,
)
from ..compiler.registers.elliptic_curve.addition_products_component import (
    addition_products_program,
)
from ..compiler.registers.elliptic_curve.addition_slope_component import (
    addition_slope_program,
)
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.register_types import montgomery_register_type, quantum_bit_type
from ..core.ir.program.registers import Register, RegisterType
from ..errors import SerializationError
from ._elliptic_curve_arithmetic_library import (
    curve_inverse_program,
    curve_modular_program,
    curve_multiplication_program,
)


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
    if type(width) is not int:
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
        _program(
            modular_subtraction_factories()[
                "qualtran.bloqs.mod_arithmetic.mod_subtraction.ModNeg"
            ]("negation", None, {"dtype": dtype, "mod": modulus})
        ),
        multi_control_x_program((0,) * (2 * width)),
        multi_control_x_program((0, 0, 0)),
    )
    return _rename_program(program, name)


def elliptic_curve_addition_factories() -> dict[str, ComponentFactory]:
    """Resolve all seven original constructors while retaining source promises."""
    return {
        "qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepOne": _flags,
        "qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepTwo": _slope,
        "qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepThree": partial(
            _arithmetic_stage, "coordinates"
        ),
        "qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepFour": partial(
            _arithmetic_stage, "products"
        ),
        "qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepFive": _cleanup,
        "qualtran.bloqs.cryptography.ecc.ec_add._ECAddStepSix": _finish,
        "qualtran.bloqs.cryptography.ecc.ec_add.ECAdd": _root,
    }


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
    if type(width) is not int:
        return FunctionRecord(name, (), registers=tuple(signature))
    dtype = montgomery_register_type(width)
    zero = basis_vector_factories()["qualtran.bloqs.basic_gates.z_basis.IntState"](
        "zero", None, {"bitsize": width, "val": 0}
    )
    zero_program = zero if isinstance(zero, ProgramRecord) else ProgramRecord((zero,))
    program = addition_slope_program(
        signature,
        zero=zero_program,
        subtract=curve_modular_program("ModSub", dtype, modulus),
        controlled_subtract=curve_modular_program("CModSub", dtype, modulus),
        inverse=curve_inverse_program(width, modulus),
        multiply=curve_multiplication_program(width, window, modulus),
        equals=equality_program(dtype),
        select_general=multi_control_x_program((0, 1, 1)),
        select_tangent=multi_control_x_program((1, 1, 1)),
        controlled_equals=controlled_equality_program(dtype, control_value=0),
        multiply_adjoint=curve_multiplication_program(
            width, window, modulus, adjoint=True
        ),
        inverse_adjoint=curve_inverse_program(width, modulus, adjoint=True),
    )
    return _rename_program(program, name)


def _arithmetic_stage(
    kind: Literal["coordinates", "products"],
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
            "Curve arithmetic requires original n, mod and window_size fields"
        )
    width, modulus = (_integer(arguments, key) for key in ("n", "mod"))
    window = _integer({"window_size": arguments.get("window_size", 1)}, "window_size")
    signature = (
        addition_coordinates_signature
        if kind == "coordinates"
        else addition_products_signature
    )(width)
    if registers is not None and registers != tuple(signature):
        raise SerializationError("Invalid original curve arithmetic signature")
    if type(width) is not int:
        return FunctionRecord(name, (), registers=tuple(signature))
    dtype = montgomery_register_type(width)
    zero = basis_vector_factories()["qualtran.bloqs.basic_gates.z_basis.IntState"](
        "zero", None, {"bitsize": width, "val": 0}
    )
    zero_program = zero if isinstance(zero, ProgramRecord) else ProgramRecord((zero,))
    cnot = fixed_gate_factories()["qualtran.bloqs.basic_gates.cnot.CNOT"](
        "cnot", None, {}
    )
    cnot_program = cnot if isinstance(cnot, ProgramRecord) else ProgramRecord((cnot,))
    multiply = curve_multiplication_program(width, window, modulus)
    multiply_adjoint = curve_multiplication_program(
        width, window, modulus, adjoint=True
    )
    if kind == "coordinates":
        program = addition_coordinates_program(
            signature,
            multiply=multiply,
            multiply_adjoint=multiply_adjoint,
            controlled_subtract=curve_modular_program("CModSub", dtype, modulus),
            zero=zero_program,
            cnot=cnot_program,
            double=curve_modular_program("ModDbl", dtype, modulus),
            add=curve_modular_program("ModAdd", dtype, modulus),
            controlled_add=curve_modular_program("CModAdd", dtype, modulus),
            add_adjoint=curve_modular_program("ModAdd", dtype, modulus, adjoint=True),
            double_adjoint=curve_modular_program(
                "ModDbl", dtype, modulus, adjoint=True
            ),
        )
    else:
        program = addition_products_program(
            signature,
            zero=zero_program,
            cnot=cnot_program,
            multiply=multiply,
            multiply_adjoint=multiply_adjoint,
            subtract=curve_modular_program("ModSub", dtype, modulus),
        )
    return _rename_program(program, name)


def _program(record: FunctionRecord | ProgramRecord) -> ProgramRecord:
    return record if isinstance(record, ProgramRecord) else ProgramRecord((record,))


def _clear_slope(dtype: RegisterType) -> ProgramRecord:
    xor = _rename_program(bitwise_word_program(dtype, "xor"), "xor")
    positive = _rename_program(
        controlled_bitwise_word_program(dtype, "xor"), "xor_positive"
    )

    def control(child: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        if child != xor.functions[0]:
            raise SerializationError("Slope cleanup requires the owned XOR child")
        return positive, _stored_control_adder(positive.functions[0])

    factory = controlled_via_and_factories(control_system=control)[
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
    ]
    return _program(
        factory(
            "clear_slope",
            None,
            {
                "subbloq": xor.functions[0],
                "ctrl_spec": ControlSpecification(
                    ((quantum_bit_type(), np.array(1)), (dtype, np.array(0)))
                ),
            },
        )
    )


def _cleanup(
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
            "Curve slope cleanup requires original n, mod and window_size fields"
        )
    width, modulus = (_integer(arguments, key) for key in ("n", "mod"))
    window = _integer({"window_size": arguments.get("window_size", 1)}, "window_size")
    signature = addition_cleanup_signature(width)
    if registers is not None and registers != tuple(signature):
        raise SerializationError("Invalid original curve slope cleanup signature")
    if type(width) is not int:
        return FunctionRecord(name, (), registers=tuple(signature))
    dtype = montgomery_register_type(width)
    graph = addition_cleanup_program(
        signature,
        inverse=curve_inverse_program(width, modulus),
        multiply=curve_multiplication_program(width, window, modulus),
        select=multi_control_x_program((1, 1)),
        clear_slope=_clear_slope(dtype),
        multiply_adjoint=curve_multiplication_program(
            width, window, modulus, adjoint=True
        ),
        inverse_adjoint=curve_inverse_program(width, modulus, adjoint=True),
        controlled_negate=curve_modular_program("CModNeg", dtype, modulus),
        add=curve_modular_program("ModAdd", dtype, modulus),
        controlled_subtract=curve_modular_program("CModSub", dtype, modulus),
    )
    return _rename_program(graph, name)


def _finish(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if set(arguments) != {"n", "mod"}:
        raise SerializationError(
            "Curve flag cleanup requires original n and mod fields"
        )
    width, modulus = (_integer(arguments, key) for key in ("n", "mod"))
    signature = addition_finish_signature(width)
    if registers is not None and registers != tuple(signature):
        raise SerializationError("Invalid original curve flag cleanup signature")
    if type(width) is not int:
        return FunctionRecord(name, (), registers=tuple(signature))
    dtype = montgomery_register_type(width)
    gates = fixed_gate_factories()
    x = gates["qualtran.bloqs.basic_gates.x_basis.XGate"]("x", None, {})
    assert isinstance(x, FunctionRecord)
    triple = controlled_via_and_factories()[
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
    ](
        "zero_three_words",
        None,
        {
            "subbloq": x,
            "ctrl_spec": ControlSpecification(((dtype, np.array([0, 0, 0])),)),
        },
    )
    graph = addition_finish_program(
        signature,
        zero_flags=multi_control_x_program((0, 0, 0)),
        zero_three_words=_program(triple),
        toffoli=_program(
            gates["qualtran.bloqs.basic_gates.toffoli.Toffoli"]("toffoli", None, {})
        ),
        zero=_program(
            basis_vector_factories()["qualtran.bloqs.basic_gates.z_basis.ZeroState"](
                "zero", None, {}
            )
        ),
        controlled_subtract=curve_modular_program("CModSub", dtype, modulus),
        controlled_add=curve_modular_program("CModAdd", dtype, modulus),
        equals_pair=equality_program(montgomery_register_type(2 * width)),
        zero_pair=multi_control_x_program((0,) * (2 * width)),
        conjunction=conjunction_program((0,) * (2 * width)),
        fanout=fanout_component_program(2),
        conjunction_adjoint=conjunction_program((0,) * (2 * width), uncompute=True),
    )
    return _rename_program(graph, name)


def _root(
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
            "Curve addition requires original n, mod and window_size fields"
        )
    width, modulus = (_integer(arguments, key) for key in ("n", "mod"))
    window = _integer({"window_size": arguments.get("window_size", 1)}, "window_size")
    signature = elliptic_curve_addition_signature(width)
    if registers is not None and registers != tuple(signature):
        raise SerializationError("Invalid original curve addition signature")
    scalar = {"n": width, "mod": modulus}
    windowed = {**scalar, "window_size": window}
    stages = (
        _program(_flags("flags", None, scalar)),
        _program(_slope("slope", None, windowed)),
        _program(_arithmetic_stage("coordinates", "coordinates", None, windowed)),
        _program(_arithmetic_stage("products", "products", None, windowed)),
        _program(_cleanup("cleanup", None, windowed)),
        _program(_finish("finish", None, scalar)),
    )
    return _rename_program(elliptic_curve_addition_program(signature, stages), name)
