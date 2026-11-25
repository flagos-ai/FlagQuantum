# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 rsa_mod_exp.ModExp.build_composite_bloq.

"""Compose modular exponentiation from the shared initializer and multipliers."""

from dataclasses import replace
from math import gcd

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type, unsigned_register_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_adjoint import adjoint_component_calls
from ..component_basis_vectors import integer_basis_function
from ..component_builder import ComponentProgramBuilder
from ..component_graph import _namespace_helpers
from .modular_multiplication_component import modular_multiplication_program


def modular_exponentiation_program(
    base: int,
    modulus: int,
    exponent_width: int,
    output_width: int,
    *,
    executable: bool = True,
    adjoint: bool = False,
) -> ProgramRecord:
    """Retain source call order and a fresh caller-owned MSB-first result word.

    Executable graphs include the existing multiplication bodies. The component
    tutorial explicitly requests declarations for its large one-level diagrams;
    those leaves fail physical binding rather than pretending to be identities.
    Each multiplication owns and erases its private work. The RIGHT result is
    logical output and is not released. No coefficient or gate synthesis changes.
    The adjoint consumes a result in this prepared image and returns the exponent.
    """
    if type(adjoint) is not bool or type(executable) is not bool:
        raise ValueError("Modular exponentiation body policies require booleans")
    if adjoint and not executable:
        raise ValueError("Exact exponentiation adjoint requires executable bodies")
    if any(type(v) is not int for v in (base, modulus, exponent_width, output_width)):
        raise ValueError("Modular exponentiation graph requires concrete integers")
    if exponent_width < 0 or output_width < 1:
        raise ValueError("Invalid modular exponentiation widths or body policy")
    if gcd(base, modulus) != 1:
        raise ValueError("Modular exponentiation base and modulus must be coprime")
    exponent_type = unsigned_register_type(exponent_width)
    output_type = unsigned_register_type(output_width)
    signature = Signature(
        (
            Register("exponent", exponent_type),
            Register("x", output_type, side=RegisterSide.RIGHT),
        )
    )
    builder = ComponentProgramBuilder(
        signature, (integer_basis_function("IntState", output_type, 1),), name="main"
    )
    x = builder.call_dict("IntState", {})["val"]
    exponent = builder.inputs["exponent"]
    assert isinstance(exponent, RegisterBinding)
    bits = builder.split(exponent)
    multiplier = base % modulus
    defined: set[str] = set()
    state = ProgramRecord((integer_basis_function("IntState", output_type, 1),))
    effect = ProgramRecord(
        (integer_basis_function("IntEffect", output_type, 1, state=False),)
    )
    pairs: list[tuple[ProgramRecord, ProgramRecord]] = [(state, effect)]
    for index in reversed(range(exponent_width)):
        name = f"CModMulK({multiplier})"
        if name not in defined:
            if executable:
                child = modular_multiplication_program(output_type, modulus, multiplier)
                if adjoint:
                    pairs.append(
                        (
                            child,
                            modular_multiplication_program(
                                output_type, modulus, multiplier, adjoint=True
                            ),
                        )
                    )
                child = replace(
                    child,
                    functions=(
                        replace(child.functions[0], name=name),
                        *child.functions[1:],
                    ),
                )
                child = _namespace_helpers(child, name)
                for function in child.functions:
                    builder.define(function)
            else:
                builder.define(
                    FunctionRecord(
                        name,
                        (),
                        registers=(
                            Register("ctrl", quantum_bit_type()),
                            Register("x", output_type),
                        ),
                    )
                )
            defined.add(name)
        outputs = builder.call_dict(name, {"ctrl": bits[index], "x": x})
        bits[index], x = outputs["ctrl"], outputs["x"]
        multiplier = multiplier * multiplier % modulus
    record = builder.finalize(
        {"exponent": builder.join(bits, dtype=exponent_type), "x": x}
    )
    if adjoint:
        record = adjoint_component_calls(record, tuple(pairs))
    return replace(
        record,
        functions=(replace(record.functions[0], name="ModExp"), *record.functions[1:]),
    )
