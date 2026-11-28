# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Bindings follow locked Qualtran v0.7.0 ecc/ec_add.py and mod_arithmetic.

"""Bind explicit shared arithmetic owners for the curve-addition stage adapters."""

from typing import Literal

from ..compiler.formats.legacy_component_resolution import ResolvedArgument
from ..compiler.formats.legacy_modular_addition_factories import _symbolic_addition
from ..compiler.formats.legacy_modular_inverse_factories import (
    _symbolic_inverse_program,
)
from ..compiler.formats.legacy_modular_multiplication_factories import (
    modular_multiplication_factories,
    symbolic_modular_doubling_program,
)
from ..compiler.formats.legacy_modular_subtraction_factories import (
    modular_subtraction_factories,
)
from ..compiler.formats.montgomery_port_views import restore_montgomery_ports
from ..compiler.registers.modular.controlled_modular_addition_component import (
    controlled_modular_addition_program,
)
from ..compiler.registers.modular.controlled_modular_negation_component import (
    controlled_modular_negation_program,
)
from ..compiler.registers.modular.controlled_modular_subtraction_component import (
    controlled_modular_subtraction_program,
)
from ..compiler.registers.modular.modular_addition_component import (
    modular_addition_program,
)
from ..compiler.registers.modular.modular_doubling_component import (
    modular_doubling_program,
)
from ..compiler.registers.modular.modular_inverse_component import (
    modular_inverse_program,
)
from ..compiler.registers.modular.modular_subtraction_component import (
    modular_subtraction_program,
)
from ..compiler.registers.modular.montgomery_multiplication_component import (
    montgomery_multiplication_program,
)
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.register_types import montgomery_register_type
from ..core.ir.program.registers import Register, RegisterType
from ..core.value_math import SymbolicInt
from ..errors import SerializationError

_ArithmeticKind = Literal["ModAdd", "CModAdd", "ModSub", "CModSub", "CModNeg", "ModDbl"]


def curve_modular_program(
    kind: _ArithmeticKind,
    dtype: RegisterType,
    modulus: SymbolicInt,
    *,
    adjoint: bool = False,
) -> ProgramRecord:
    """Choose concrete or source-defined logical children, without numeric execution.

    The known constructor supplies the actual constant/control semantics. A
    source-refused leaf stays explicit; its ABI never implies a gate or a price.
    """
    width = dtype.num_qubits
    assert type(width) is int
    if adjoint and kind not in {"ModAdd", "CModAdd", "ModDbl"}:
        raise SerializationError("Unsupported curve arithmetic adjoint constructor")
    if type(modulus) is int:
        if kind == "ModAdd":
            return modular_addition_program(width, modulus, adjoint=adjoint)
        if kind == "CModAdd":
            return controlled_modular_addition_program(dtype, modulus, adjoint=adjoint)
        if kind == "ModSub":
            return modular_subtraction_program(dtype, modulus)
        if kind == "CModSub":
            return controlled_modular_subtraction_program(dtype, modulus)
        if kind == "CModNeg":
            return controlled_modular_negation_program(dtype, modulus)
        return modular_doubling_program(dtype, modulus, adjoint=adjoint)
    if kind in {"ModAdd", "CModAdd"}:
        word = montgomery_register_type(width, modulus) if kind == "ModAdd" else dtype
        return _symbolic_addition(
            word, modulus, controlled=kind == "CModAdd", adjoint=adjoint
        )
    if kind == "ModDbl":
        return symbolic_modular_doubling_program(dtype, modulus, adjoint=adjoint)
    result = modular_subtraction_factories()[
        "qualtran.bloqs.mod_arithmetic.mod_subtraction." + kind
    ]("main", None, {"dtype": dtype, "mod": modulus})
    return result if isinstance(result, ProgramRecord) else ProgramRecord((result,))


def curve_inverse_program(
    width: int, modulus: SymbolicInt, *, adjoint: bool = False
) -> ProgramRecord:
    """Keep published finite views and retain live symbolic source metadata."""
    program = (
        modular_inverse_program(width, modulus, adjoint=adjoint)
        if type(modulus) is int
        else _symbolic_inverse_program(
            "KaliskiModInverse", width, modulus, adjoint=adjoint
        )
    )
    return without_modulus(program) if type(modulus) is int else program


def without_modulus(program: ProgramRecord) -> ProgramRecord:
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


def curve_multiplication_program(
    width: int, window: SymbolicInt, modulus: SymbolicInt, *, adjoint: bool = False
) -> ProgramRecord:
    if type(window) is int and type(modulus) is int:
        return without_modulus(
            montgomery_multiplication_program(width, window, modulus, adjoint=adjoint)
        )
    factory = modular_multiplication_factories()[
        "qualtran.bloqs.mod_arithmetic.mod_multiplication.DirtyOutOfPlaceMontgomeryModMul"
    ]
    arguments: dict[str, ResolvedArgument] = {
        "bitsize": width,
        "window_size": window,
        "mod": modulus,
        "uncompute": adjoint,
    }
    declaration = factory("multiply", None, arguments)
    assert isinstance(declaration, FunctionRecord) and not declaration.blocks
    assert declaration.registers is not None
    if type(modulus) is not int:
        # The modulus is recoverable from this constructor. Preserve the
        # original live word views inside the surrounding available stages.
        return ProgramRecord((declaration,))
    # The source codec omits Montgomery modulus metadata, but keeps the exact
    # symbolic history width and directional ports. Ask the existing factory
    # to validate that source view instead of inserting a synthetic Cast body.
    ports = tuple(
        Register(
            r.name,
            (
                montgomery_register_type(r.dtype.num_qubits)
                if r.dtype.encoding == "montgomery"
                else r.dtype
            ),
            r.shape_symbolic,
            r.side,
        )
        for r in declaration.registers
    )
    result = factory("multiply", ports, arguments)
    return result if isinstance(result, ProgramRecord) else ProgramRecord((result,))
