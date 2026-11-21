# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 gf_poly_arithmetic.

"""Present complete packed polynomial addition using existing typed field owners."""

from collections.abc import Mapping

from examples.algorithms.word_arithmetic_decomposition import word_symbols
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.finite_field.addition_component import (
    field_addition_program,
)
from flagquantum.compiler.registers.finite_field.field_polynomial_layout import (
    FieldPolynomialLayout,
)
from flagquantum.compiler.registers.finite_field.polynomial_addition_component import (
    field_polynomial_addition_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.partition_symbols import (
    partition_wire_symbol,
    split_join_wire_symbol,
)
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    RegisterWireSymbolSupplier,
    SymbolClass,
    register_score_data,
)
from flagquantum.simulation.finite_field_polynomial_metadata import (
    resolve_finite_field_polynomial_encoding,
)


def field_polynomial_program(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    """Reuse packed addition and retain original symbolic/atomic view refusals."""
    if operation not in {"GF2PolyAdd", "GF2PolyAddK"}:
        return None
    dtype = ports[0].dtype
    if type(dtype.num_qubits) is not int:
        return None
    encoding = resolve_finite_field_polynomial_encoding(dtype)
    degree, width = encoding.degree, encoding.field.bitsize
    assert type(degree) is int and type(width) is int
    constants = (
        tuple(int(c) for c in encoding.to_gf_coefficients(parameters["g_x"]))
        if operation == "GF2PolyAddK"
        else None
    )
    return field_polynomial_addition_program(
        FieldPolynomialLayout(degree, width),
        encoding.field.register_type(),
        constants=constants,
        packed_dtype=dtype,
    )


def field_polynomial_definition(
    record: ProgramRecord,
    entry: str,
    ports: Signature,
    parameters: Mapping[str, object],
) -> tuple[str, ProgramRecord] | None:
    """Identify complete coefficient addition bodies by literal definition match."""
    function = next(f for f in record.functions if f.name == entry)
    registers = function.registers or ()
    names = tuple(r.name for r in registers)
    if names not in {("x",), ("x", "y")}:
        return None
    encoding = resolve_finite_field_polynomial_encoding(ports[0].dtype)
    coefficient = encoding.field.register_type()
    constants: tuple[int | None, ...] = (
        tuple(int(c) for c in encoding.to_gf_coefficients(parameters["g_x"]))
        if names == ("x",)
        else (None,)
    )
    for constant in dict.fromkeys(constants):
        template = field_addition_program(coefficient, constant=constant)
        if matches_component_definition(record, entry, template):
            return "GF2Addition" if constant is None else "GF2AddK", template
    return None


def field_polynomial_symbols(record: ProgramRecord) -> RegisterWireSymbolSupplier:
    """Keep atomic coefficient views and original named field addition glyphs."""
    functions = {f.name: f for f in record.functions}

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        names = tuple(r.name for r in functions[step.name].registers or ())
        if names in {("x",), ("x", "y")}:
            if register is None:
                return "Text", {
                    "text": "GF2AddK" if names == ("x",) else "GF2Addition"[:8] + "..",
                    "fontsize": 10,
                }
            return partition_wire_symbol(step, register, index)
        return split_join_wire_symbol(step, register, index)

    return supply


def field_polynomial_score(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> MusicalScoreData | None:
    """Project complete source-ordered packed addition graphs into catalog scores."""
    record = field_polynomial_program(operation, ports, parameters)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=field_polynomial_symbols(record),
    )


def field_polynomial_child(
    record: ProgramRecord,
    entry: str,
    ports: Signature,
    parameters: Mapping[str, object],
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    """Reuse complete field addition drawings after whole-definition qualification."""
    selected = field_polynomial_definition(record, entry, ports, parameters)
    if selected is None:
        return None
    label, template = selected
    return (
        label,
        template,
        word_symbols(template, "Xor" if label == "GF2Addition" else "XorK"),
    )
