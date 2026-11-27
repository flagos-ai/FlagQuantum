# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 gf2_square.py and gf2_inverse.py.

"""Present complete Frobenius and inverse graphs through existing field owners."""

from collections.abc import Mapping
from typing import SupportsIndex

from examples.algorithms.field_product_decomposition import field_product_symbols
from examples.algorithms.field_product_definitions import field_product_definition
from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.finite_field.addition_component import (
    field_addition_program,
)
from flagquantum.compiler.registers.finite_field.frobenius_component import (
    field_frobenius_program,
)
from flagquantum.compiler.registers.finite_field.inverse_component import (
    field_inverse_program,
)
from flagquantum.compiler.registers.finite_field.karatsuba_component import (
    karatsuba_multiplication_program,
)
from flagquantum.compiler.registers.finite_field.linear_component import (
    binary_linear_program,
)
from flagquantum.compiler.registers.finite_field.square_component import (
    field_square_decomposition_matrix,
    field_square_decomposition_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.partition_symbols import partition_wire_symbol
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    RegisterWireSymbolSupplier,
    SymbolClass,
    register_score_data,
)
from flagquantum.simulation.finite_field_frobenius import field_frobenius_matrix
from flagquantum.simulation.finite_field_metadata import (
    resolve_finite_field_encoding,
    resolve_quantum_field_storage,
)


def field_power_program(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    """Reuse concrete executable owners and preserve original symbolic refusal."""
    if operation not in {"GF2Square", "GF2Inverse"}:
        return None
    dtype = ports[0].dtype
    if type(dtype.num_qubits) is not int:
        return None
    dtype = resolve_quantum_field_storage(dtype)
    modulus = int(resolve_finite_field_encoding(dtype).gf_type.irreducible_poly)
    if operation == "GF2Inverse":
        return field_inverse_program(dtype, modulus=modulus)
    count, adjoint = parameters["k"], parameters["uncompute"]
    if not isinstance(count, SupportsIndex) or type(adjoint) is not bool:
        return None
    return field_square_decomposition_program(
        dtype, count, modulus=modulus, uncompute=adjoint
    )


def field_power_definition(
    record: ProgramRecord,
    entry: str,
    ports: Signature,
    parameters: Mapping[str, object],
) -> tuple[str, ProgramRecord] | None:
    """Match complete power children before reusing product dependency labels."""
    dtype = ports[0].dtype
    width = dtype.num_qubits
    if dtype.encoding != "finite_field" or type(width) is not int:
        return None
    dtype = resolve_quantum_field_storage(dtype)
    encoding = resolve_finite_field_encoding(dtype)
    modulus = int(encoding.gf_type.irreducible_poly)
    function = next(f for f in record.functions if f.name == entry)
    names = tuple(r.name for r in function.registers or ())
    if {"k", "uncompute"} <= parameters.keys() and names in {("x",), ("q",)}:
        count, uncompute = parameters["k"], parameters["uncompute"]
        if not isinstance(count, SupportsIndex) or type(uncompute) is not bool:
            return None
        if names == ("x",):
            label = "GF2Square"
            template = field_square_decomposition_program(
                dtype, count, modulus=modulus, uncompute=uncompute
            )
        else:
            # The frozen flag changes logical identity even though its physical
            # PLU builder ignores it. Keep the caller's declared direction in
            # the label only after matching the complete literal child body.
            label = "SynthesizeLRCircuit" + ("†" if uncompute else "")
            template = binary_linear_program(
                field_square_decomposition_matrix(width, count, modulus=modulus)
            )
        return (
            (label, template)
            if matches_component_definition(record, entry, template)
            else None
        )
    candidates: list[tuple[str, ProgramRecord]] = []
    if names == ("x", "y"):
        candidates.append(("GF2Addition", field_addition_program(dtype)))
    elif names == ("x", "y", "result"):
        candidates.append(
            (
                "GF2MulViaKaratsuba",
                karatsuba_multiplication_program(dtype, modulus=modulus),
            )
        )
    elif names in {("x",), ("q",)}:
        for count in sorted({1, *(2**i for i in range((width - 1).bit_length()))}):
            for adjoint in (False, True):
                if names == ("x",):
                    label = "GF2Square"
                    template = field_frobenius_program(
                        dtype, count, modulus=modulus, adjoint=adjoint
                    )
                else:
                    matrix = tuple(
                        tuple(int(bit) for bit in row)
                        for row in field_frobenius_matrix(encoding, count)
                    )
                    label = "SynthesizeLRCircuit" + ("†" if adjoint else "")
                    template = binary_linear_program(matrix, adjoint=adjoint)
                candidates.append((label, template))
    selected = next(
        (c for c in candidates if matches_component_definition(record, entry, c[1])),
        None,
    )
    return selected or field_product_definition(record, entry, ports, parameters)


def field_power_symbols(
    record: ProgramRecord, ports: Signature, parameters: Mapping[str, object]
) -> RegisterWireSymbolSupplier:
    """Reuse product glyphs and override only complete power child labels."""
    root = record.functions[0]
    context_dtype = ports[0].dtype
    if context_dtype.encoding == "finite_field":
        stored = resolve_quantum_field_storage(context_dtype)
        if stored is not context_dtype:
            ports = Signature(
                tuple(
                    Register(
                        r.name,
                        stored if r.dtype == context_dtype else r.dtype,
                        r.shape_symbolic,
                        r.side,
                    )
                    for r in ports
                )
            )
    fallback = field_product_symbols(record, ports, parameters)
    labels = {
        op.callee: selected[0]
        for block in root.blocks
        for op in block.operations
        if op.callee is not None
        and (selected := field_power_definition(record, op.callee, ports, parameters))
        is not None
    }

    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        label = labels.get(step.name)
        if label not in {
            "GF2Addition",
            "GF2Square",
            "GF2MulViaKaratsuba",
            "SynthesizeLRCircuit",
            "SynthesizeLRCircuit†",
        }:
            return fallback(step, register, index)
        if register is None:
            return "Text", {
                "text": label if len(label) <= 10 else label[:8] + "..",
                "fontsize": 10,
            }
        return partition_wire_symbol(step, register, index)

    return supply


def field_power_score(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> MusicalScoreData | None:
    """Draw the stored complete power root without allocating execution state."""
    record = field_power_program(operation, ports, parameters)
    if record is None:
        return None
    return register_score_data(
        component_register_score(record, record.functions[0].name),
        register_wire_symbol=field_power_symbols(record, ports, parameters),
    )


def field_power_child(
    record: ProgramRecord,
    entry: str,
    ports: Signature,
    parameters: Mapping[str, object],
) -> tuple[str, ProgramRecord, RegisterWireSymbolSupplier] | None:
    """Qualify complete children, retaining native exact inverse labels."""
    selected = field_power_definition(record, entry, ports, parameters)
    if selected is None:
        return None
    label, template = selected
    return label, template, field_power_symbols(template, ports, parameters)
