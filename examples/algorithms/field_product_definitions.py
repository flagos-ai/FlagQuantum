# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 gf_arithmetic/gf2_multiplication.py.

"""Select existing field product definitions from typed ports and source parameters."""

from collections.abc import Mapping

from flagquantum.compiler.registers.component_definition_match import (
    matches_component_definition,
)
from flagquantum.compiler.registers.finite_field.constant_multiplication_component import (
    constant_multiplication_decomposition_program,
    constant_multiplication_program,
)
from flagquantum.compiler.registers.finite_field.karatsuba_component import (
    karatsuba_multiplication_program,
)
from flagquantum.compiler.registers.finite_field.linear_component import (
    binary_linear_decomposition_program,
    binary_linear_program,
)
from flagquantum.compiler.registers.finite_field.monomial_component import (
    monomial_decomposition_program,
    monomial_multiplication_program,
)
from flagquantum.compiler.registers.finite_field.multiplication_component import (
    field_multiplication_program,
)
from flagquantum.compiler.registers.finite_field.polynomial_product_component import (
    binary_polynomial_program,
    one_plus_polynomial_program,
)
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.finite_field_metadata import resolve_finite_field_encoding
from flagquantum.simulation.finite_field_multiplication import field_scale_matrix


def field_product_catalog_program(
    operation: str, ports: Signature, parameters: Mapping[str, object]
) -> ProgramRecord | None:
    """Reuse the complete concrete owners and retain original symbolic refusal."""
    if operation in {"BinaryPolynomialMultiplication", "MultiplyPolyByOnePlusXk"}:
        width = parameters["n"]
        if type(width) is not int:
            return None
        if operation == "BinaryPolynomialMultiplication":
            return binary_polynomial_program(width)
        shift = parameters["k"]
        return one_plus_polynomial_program(width, shift) if type(shift) is int else None
    if operation not in {
        "GF2Multiplication",
        "GF2MulK",
        "GF2MulViaKaratsuba",
        "GF2ShiftLeft",
        "GF2ShiftRight",
    }:
        return None
    dtype = ports[0].dtype
    if type(dtype.num_qubits) is not int:
        return None
    modulus = int(resolve_finite_field_encoding(dtype).gf_type.irreducible_poly)
    if operation == "GF2Multiplication":
        accumulate = parameters["plus_equal_prod"]
        assert type(accumulate) is bool
        return field_multiplication_program(
            dtype, modulus=modulus, plus_equal_prod=accumulate
        )
    if operation == "GF2MulViaKaratsuba":
        return karatsuba_multiplication_program(dtype, modulus=modulus)
    if operation == "GF2MulK":
        constant = parameters["const"]
        assert type(constant) is int
        return constant_multiplication_decomposition_program(
            dtype, constant, modulus=modulus
        )
    shift = parameters["k"]
    if type(shift) is not int:
        return None
    return monomial_decomposition_program(
        dtype, shift, modulus=modulus, adjoint=operation == "GF2ShiftLeft"
    )


def field_product_definition(
    record: ProgramRecord,
    entry: str,
    root_ports: Signature,
    parameters: Mapping[str, object],
) -> tuple[str, ProgramRecord] | None:
    """Qualify whole definitions without relying on generated function names.

    Polynomial shapes determine their original recursion parameters. Field
    shifts and constants come from the owning factory's declared values.
    Exact native linear inverses retain a dagger label because the frozen
    SynthesizeLRCircuit ignores its adjoint flag in its decomposition.
    """
    function = next(f for f in record.functions if f.name == entry)
    registers = function.registers or ()
    ports = tuple(r.name for r in registers)
    candidates: list[tuple[str, ProgramRecord]] = []
    if ports == ("f", "g", "h") and len(registers[0].shape) == 1:
        width, target = registers[0].shape[0], registers[2].shape[0]
        if target == 2 * width - 1:
            candidates.append(
                ("BinaryPolynomialMultiplication", binary_polynomial_program(width))
            )
        else:
            shift = target // 2 if width == 1 else target - 2 * width + 1
            if shift > 0 and (width == 1 or shift <= 2 * width - 1):
                candidates.append(
                    (
                        "MultiplyPolyByOnePlusXk",
                        one_plus_polynomial_program(width, shift),
                    )
                )
    elif ports in {("f",), ("g",), ("q",)}:
        dtype = root_ports[0].dtype
        width = dtype.num_qubits
        if dtype.encoding != "finite_field" or type(width) is not int:
            return None
        encoding = resolve_finite_field_encoding(dtype)
        modulus = int(encoding.gf_type.irreducible_poly)
        if ports == ("f",):
            shifts = {width, (width + 1) // 2}
            factory_shift = parameters.get("k")
            if type(factory_shift) is int:
                shifts.add(factory_shift)
            for shift in sorted(shifts):
                for power, label in (
                    (shift, "GF2ShiftRight"),
                    (-shift, "GF2ShiftLeft"),
                ):
                    candidates.append(
                        (
                            label,
                            monomial_multiplication_program(
                                dtype, power, modulus=modulus
                            ),
                        )
                    )
        else:
            constant = parameters.get("const", (1 << ((width + 1) // 2)) | 1)
            assert type(constant) is int
            matrix = tuple(
                tuple(int(bit) for bit in row)
                for row in field_scale_matrix(encoding, constant)
            )
            for adjoint in (False, True):
                label = "GF2MulK" if ports == ("g",) else "SynthesizeLRCircuit"
                template = (
                    (
                        constant_multiplication_decomposition_program(
                            dtype, constant, modulus=modulus, adjoint=adjoint
                        )
                        if constant == 0
                        else constant_multiplication_program(
                            dtype, constant, modulus=modulus, adjoint=adjoint
                        )
                    )
                    if ports == ("g",)
                    else (
                        binary_linear_decomposition_program(matrix, adjoint=adjoint)
                        if constant == 0
                        else binary_linear_program(matrix, adjoint=adjoint)
                    )
                )
                candidates.append((label + ("†" if adjoint else ""), template))
    return next(
        (
            candidate
            for candidate in candidates
            if matches_component_definition(record, entry, candidate[1])
        ),
        None,
    )
