# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor control follows locked Qualtran v0.7.0 ecc/ec_add.py.

"""Bind explicit controlled curve-stage dependencies without source SDK objects."""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Literal

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
from ..algorithms.primitives.elliptic_curve_fixed_point import (
    fixed_point_addition_signature,
)
from ..compiler.formats.legacy_basis_vector_factories import basis_vector_factories
from ..compiler.formats.legacy_component_arguments import (
    AffinePointArgument,
    ControlSpecification,
)
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_controlled_factories import controlled_via_and_factories
from ..compiler.formats.legacy_fixed_gate_factories import fixed_gate_factories
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.registers.bits.bitwise_component_control import (
    _controlled_cx,
    _toffoli_function,
)
from ..compiler.registers.component_control import (
    ControlAdder,
    _stored_control_adder,
    component_program_control_adder,
    control_component_program,
    quantum_control_signature,
)
from ..compiler.registers.control.component_bookkeeping import (
    bookkeeping_scalar_control_adder,
)
from ..compiler.registers.elliptic_curve.addition_component import (
    controlled_elliptic_curve_addition_program,
)
from ..compiler.registers.elliptic_curve.addition_coordinates_component import (
    addition_coordinates_program,
)
from ..compiler.registers.elliptic_curve.addition_products_component import (
    addition_products_program,
)
from ..compiler.registers.modular.additional_control_component import (
    additional_modular_addition_program,
    additional_modular_subtraction_program,
)
from ..compiler.registers.modular.controlled_doubling_component import (
    controlled_modular_doubling_program,
)
from ..compiler.registers.modular.montgomery_signature import (
    montgomery_product_signature,
    validate_dirty_montgomery_constructor,
)
from ..core.ir.program import FunctionRecord, ProgramRecord, verify_program_record
from ..core.ir.program.register_types import montgomery_register_type, quantum_bit_type
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..core.value_math import SymbolicInt
from ..errors import SerializationError
from ._elliptic_curve_window_control_library import controlled_curve_window_addition

_KINDS = (
    "_ECAddStepOne",
    "_ECAddStepTwo",
    "_ECAddStepThree",
    "_ECAddStepFour",
    "_ECAddStepFive",
    "_ECAddStepSix",
)
_SIGNATURES = {
    kind: signature
    for kind, signature in zip(
        _KINDS,
        (
            addition_flags_signature,
            addition_slope_signature,
            addition_coordinates_signature,
            addition_products_signature,
            addition_cleanup_signature,
            addition_finish_signature,
        ),
        strict=True,
    )
}


def _refused_control(name: str, signature: Signature) -> ProgramRecord:
    """Declare one explicitly identified constructor's original control refusal."""
    return ProgramRecord(
        (
            FunctionRecord(
                name,
                (),
                registers=tuple(
                    quantum_control_signature(signature, ((quantum_bit_type(), ()),))
                ),
            ),
        )
    )


def controlled_curve_arithmetic_stage(
    kind: Literal["coordinates", "products"],
    width: int,
    modulus: int,
    window: SymbolicInt = 1,
) -> ProgramRecord:
    """Compose all source-available positive-controlled coordinate/product calls.

    Data words retain source MSB order and the original optional inner control.
    Dirty multiplication and IntState controls are genuine directional refusals,
    not substitutes for available arithmetic. Rounded multiplier history retains
    the source rounded expression for symbolic windows. Concrete constructor
    constraints remain enforced. Unresolved source leaves prevent physical binding.
    """
    validate_dirty_montgomery_constructor(width, window, modulus)
    dtype = montgomery_register_type(width)
    signature = (
        addition_coordinates_signature
        if kind == "coordinates"
        else addition_products_signature
    )(width)
    adders: dict[str, ControlAdder] = {}
    transient: set[str] = set()

    def owner(name: str, controlled: ProgramRecord) -> ProgramRecord:
        ports = controlled.functions[0].registers
        assert ports is not None
        transient.add(name)
        adders[name] = component_program_control_adder(controlled)
        return ProgramRecord((FunctionRecord(name, (), registers=ports[1:]),))

    multiply, multiply_adjoint = (
        owner(
            name,
            _refused_control(
                f"controlled_{name}_window_{window}",
                montgomery_product_signature(width, window, modulus, adjoint=inverse),
            ),
        )
        for name, inverse in (("multiply", False), ("multiply_inverse", True))
    )
    state = basis_vector_factories()["qualtran.bloqs.basic_gates.z_basis.IntState"](
        "zero", None, {"bitsize": width, "val": 0}
    )
    state_root = state.functions[0] if isinstance(state, ProgramRecord) else state
    assert state_root.registers is not None
    zero = owner(
        "zero", _refused_control("controlled_zero", Signature(state_root.registers))
    )
    cnot = fixed_gate_factories()["qualtran.bloqs.basic_gates.cnot.CNOT"](
        "cnot", None, {}
    )
    assert isinstance(cnot, FunctionRecord)
    cnot_program = ProgramRecord((cnot,))
    adders["cnot"] = _controlled_cx
    if kind == "coordinates":
        program = addition_coordinates_program(
            signature,
            multiply=multiply,
            multiply_adjoint=multiply_adjoint,
            zero=zero,
            cnot=cnot_program,
            controlled_subtract=owner(
                "controlled_subtract",
                additional_modular_subtraction_program(dtype, modulus, control_value=1),
            ),
            double=owner("double", controlled_modular_doubling_program(dtype, modulus)),
            add=owner(
                "add",
                additional_modular_addition_program(
                    montgomery_register_type(width, modulus), modulus
                ),
            ),
            controlled_add=owner(
                "controlled_add",
                additional_modular_addition_program(dtype, modulus, control_value=1),
            ),
            add_adjoint=owner(
                "add_inverse",
                additional_modular_addition_program(
                    montgomery_register_type(width, modulus), modulus, adjoint=True
                ),
            ),
            double_adjoint=owner(
                "double_inverse",
                controlled_modular_doubling_program(dtype, modulus, adjoint=True),
            ),
        )
    else:
        program = addition_products_program(
            signature,
            zero=zero,
            cnot=cnot_program,
            multiply=multiply,
            multiply_adjoint=multiply_adjoint,
            subtract=owner(
                "subtract", additional_modular_subtraction_program(dtype, modulus)
            ),
        )
    functions = {function.name: function for function in program.functions}
    for operation in program.functions[0].blocks[0].operations:
        assert operation.callee is not None
        if operation.callee not in adders:
            adders[operation.callee] = bookkeeping_scalar_control_adder(
                functions[operation.callee]
            )
    program = control_component_program(
        replace(program, functions=(*program.functions, _toffoli_function())), adders
    )
    # These temporary ordinary declarations only supplied the canonical stage layout.
    program = replace(
        program,
        functions=tuple(
            function for function in program.functions if function.name not in transient
        ),
    )
    diagnostics = verify_program_record(program)
    if diagnostics:
        raise ValueError(f"Invalid controlled curve stage: {diagnostics}")
    return program


def _controlled_stage_or_refusal(
    kind: str, width: SymbolicInt, modulus: SymbolicInt, window: SymbolicInt
) -> ProgramRecord:
    if kind not in {"_ECAddStepThree", "_ECAddStepFour"} or type(width) is not int:
        return _refused_control("controlled_" + kind, _SIGNATURES[kind](width))
    try:
        validate_dirty_montgomery_constructor(width, window, modulus)
    except (AssertionError, ZeroDivisionError):
        # The source stage fails while constructing its first dirty multiplier;
        # its six-call parent remains available and must not lose that layer.
        return _refused_control("controlled_" + kind, _SIGNATURES[kind](width))
    if type(modulus) is not int:
        raise SerializationError(
            "Symbolic-modulus arithmetic control bodies remain unsupported"
        )
    return controlled_curve_arithmetic_stage(
        "coordinates" if kind == "_ECAddStepThree" else "products",
        width,
        modulus,
        window,
    )


def controlled_curve_addition(
    width: SymbolicInt, modulus: SymbolicInt, window: SymbolicInt = 1
) -> ProgramRecord:
    """Keep the six available root calls and only genuine stage-control refusals."""
    stages = tuple(
        _controlled_stage_or_refusal(kind, width, modulus, window) for kind in _KINDS
    )
    return controlled_elliptic_curve_addition_program(
        elliptic_curve_addition_signature(width), stages
    )


def elliptic_curve_control_factories() -> dict[str, ComponentFactory]:
    """Resolve controlled curve roots using explicit source constructor references.

    This control-only catalogue declares ordinary children as constructor
    references; plain forward execution uses elliptic_curve_addition_factories.
    Requested controlled parents retain all available bodies. Arithmetic bodies
    require concrete n/mod; genuine refusal declarations retain symbolic metadata.
    """
    recipes: dict[FunctionRecord, tuple[str, dict[str, ResolvedArgument]]] = {}

    def reference(
        kind: str,
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord:
        if kind == "ECWindowAddR":
            if (
                not {"n", "R", "add_window_size"} <= arguments.keys()
                or arguments.keys() - {"n", "R", "add_window_size", "mul_window_size"}
                or not isinstance(arguments["R"], AffinePointArgument)
            ):
                raise SerializationError(
                    "Invalid curve window control constructor fields"
                )
            width = _integer(arguments, "n")
            window = _integer(arguments, "add_window_size")
            _integer({"window": arguments.get("mul_window_size", 1)}, "window")
            signature = fixed_point_addition_signature(width, window)
            if registers is not None and registers != tuple(signature):
                raise SerializationError(
                    "Invalid curve window control reference signature"
                )
            result = FunctionRecord(name, (), registers=tuple(signature))
            recipes[result] = kind, dict(arguments)
            return result
        allowed = {"n", "mod"} | (
            {"window_size"} if kind not in {"_ECAddStepOne", "_ECAddStepSix"} else set()
        )
        if not {"n", "mod"} <= arguments.keys() or arguments.keys() - allowed:
            raise SerializationError("Invalid curve control constructor fields")
        width = _integer(arguments, "n")
        _integer(arguments, "mod")
        if "window_size" in arguments:
            _integer(arguments, "window_size")
        signature = (
            elliptic_curve_addition_signature if kind == "ECAdd" else _SIGNATURES[kind]
        )(width)
        if registers is not None and registers != tuple(signature):
            raise SerializationError("Invalid curve control reference signature")
        result = FunctionRecord(name, (), registers=tuple(signature))
        recipes[result] = kind, dict(arguments)
        return result

    def positive(child: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        if child not in recipes:
            raise SerializationError("Missing curve control constructor context")
        kind, arguments = recipes[child]
        if kind == "ECWindowAddR":
            width = _integer(arguments, "n")
            window = _integer(arguments, "add_window_size")
            if type(window) is not int:
                raise SerializationError(
                    "Symbolic curve address window cannot decompose"
                )
            point = arguments["R"]
            assert isinstance(point, AffinePointArgument)
            mul_window = _integer(
                {"window": arguments.get("mul_window_size", 1)}, "window"
            )
            program = controlled_curve_window_addition(
                width,
                point,
                window,
                addition=controlled_curve_addition(width, int(point.mod), mul_window),
            )
            program = _rename_program(program, child.name + "_positive")
            return program, _stored_control_adder(program.functions[0])
        width, modulus = (_integer(arguments, key) for key in ("n", "mod"))
        window = _integer(
            {"window_size": arguments.get("window_size", 1)}, "window_size"
        )
        if kind == "ECAdd":
            program = controlled_curve_addition(width, modulus, window)
        else:
            program = _controlled_stage_or_refusal(kind, width, modulus, window)
        program = _rename_program(program, child.name + "_positive")
        return program, _stored_control_adder(program.functions[0])

    def controlled(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        child, spec = arguments.get("subbloq"), arguments.get("ctrl_spec")
        if (
            set(arguments) != {"subbloq", "ctrl_spec"}
            or not isinstance(child, FunctionRecord)
            or not isinstance(spec, ControlSpecification)
            or len(spec.clauses) != 1
            or spec.clauses[0][0] != quantum_bit_type()
            or spec.clauses[0][1].shape != ()
            or spec.clauses[0][1].item() != 1
        ):
            raise SerializationError("Curve control requires one positive scalar QBit")
        program, _ = positive(child)
        if registers is not None and registers != program.functions[0].registers:
            raise SerializationError("Invalid controlled curve signature")
        return _rename_program(program, name)

    return {
        "qualtran.bloqs.cryptography.ecc.ec_add_r.ECWindowAddR": partial(
            reference, "ECWindowAddR"
        ),
        **{
            "qualtran.bloqs.cryptography.ecc.ec_add." + kind: partial(reference, kind)
            for kind in (*_KINDS, "ECAdd")
        },
        "qualtran._infra.controlled.Controlled": controlled,
        **controlled_via_and_factories(
            control_system=positive, allow_symbolic_targets=True
        ),
    }
