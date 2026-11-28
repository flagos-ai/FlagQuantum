# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor control follows locked Qualtran v0.7.0 ecc/ec_add_r.py.

"""Bind windowed control composition to actual table and curve dependencies."""

from dataclasses import replace

import sympy

from ..algorithms.primitives.elliptic_curve_fixed_point import (
    fixed_point_addition_signature,
)
from ..algorithms.primitives.elliptic_curve_point import ECPoint
from ..algorithms.primitives.elliptic_curve_window_addition import windowed_point_table
from ..compiler.formats.legacy_component_arguments import AffinePointArgument
from ..compiler.registers.component_control import (
    component_program_control_adder,
    control_component_program,
    quantum_control_signature,
)
from ..compiler.registers.control.component_bookkeeping import (
    bookkeeping_scalar_control_adder,
)
from ..compiler.registers.elliptic_curve.window_addition_component import (
    windowed_curve_addition_program,
)
from ..compiler.registers.lookup.qroam_clean_metadata import qroam_clean_signature
from ..core.ir.program import FunctionRecord, ProgramRecord, verify_program_record
from ..core.ir.program.register_types import quantum_bit_type
from ..core.ir.program.signatures import Signature
from ..core.value_math import SymbolicInt
from ..errors import SerializationError


def controlled_curve_window_addition(
    width: SymbolicInt,
    point: AffinePointArgument,
    window: int,
    *,
    addition: ProgramRecord,
) -> ProgramRecord:
    """Keep the five source calls around genuinely refused lookup controls.

    The outer scalar control is distinct from the MSB-first address array.
    Table words and retained banks belong to this root until exact erasure.
    Concrete table construction retains original coordinate/domain failures;
    symbolic n uses shape metadata without evaluating point coordinates.
    """
    logs: tuple[SymbolicInt, ...]
    if type(width) is not int:
        logs = (0,)
    else:
        if any(
            type(value) is not int and not isinstance(value, sympy.Integer)
            for value in (point.x, point.y, point.curve_a)
        ):
            raise SerializationError("Symbolic curve point table remains unsupported")
        if type(point.mod) is not int:
            # Original uint_to_montgomery rejects every symbolic modulus,
            # after the first source coordinate conversion, even Integer(p).
            int(point.x)
            raise AssertionError
        table = windowed_point_table(
            width, window, ECPoint(point.x, point.y, point.mod, point.curve_a)
        )
        logs = table.log_block_sizes
    signature = qroam_clean_signature(
        (1 << window,), (width,) * 3, logs, selection_widths=(window,)
    )

    def refused(name: str, ports: Signature) -> ProgramRecord:
        return ProgramRecord(
            (
                FunctionRecord(
                    name,
                    (),
                    registers=tuple(
                        quantum_control_signature(ports, ((quantum_bit_type(), ()),))
                    ),
                ),
            )
        )

    controlled = {
        "lookup": refused("controlled_point_table", signature),
        "addition": addition,
        "erasure": refused("controlled_point_table_adjoint", signature.adjoint()),
    }
    ordinary = {
        name: ProgramRecord(
            (
                FunctionRecord(
                    name,
                    (),
                    registers=(
                        record.functions[0].registers[1:]
                        if record.functions[0].registers
                        else ()
                    ),
                ),
            )
        )
        for name, record in controlled.items()
    }
    program = windowed_curve_addition_program(
        fixed_point_addition_signature(width, window), **ordinary
    )
    adders = {
        name: component_program_control_adder(record)
        for name, record in controlled.items()
    }
    functions = {function.name: function for function in program.functions}
    for operation in program.functions[0].blocks[0].operations:
        assert operation.callee is not None
        if operation.callee not in adders:
            adders[operation.callee] = bookkeeping_scalar_control_adder(
                functions[operation.callee]
            )
    program = control_component_program(program, adders)
    program = replace(
        program,
        functions=tuple(
            function for function in program.functions if function.name not in ordinary
        ),
    )
    diagnostics = verify_program_record(program)
    if diagnostics:
        raise ValueError(f"Invalid controlled curve window: {diagnostics}")
    return program
