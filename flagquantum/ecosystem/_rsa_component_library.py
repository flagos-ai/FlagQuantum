# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow Qualtran v0.7.0 RSA and _factoring_shims source files.

"""Bind original RSA roots while retaining the terminal source CBit failure."""

from collections.abc import Mapping

from ..algorithms.primitives.rsa_phase_estimation import RSAPhaseEstimation
from ..compiler.formats.legacy_basis_vector_factories import (
    basis_vector_factories,
    x_basis_vector_factories,
)
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_modular_multiplication_factories import (
    modular_multiplication_factories,
)
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.registers.phase.rsa_phase_component import rsa_phase_estimation_program
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.register_types import unsigned_register_type
from ..core.ir.program.registers import Register
from ..core.value_math import SymbolicInt
from ..errors import SerializationError
from ._fourier_measurement_library import _measure, fourier_measurement_factories

_RSA = "qualtran.bloqs.cryptography.rsa.rsa_phase_estimate.RSAPhaseEstimate"


def _as_program(record: FunctionRecord | ProgramRecord) -> ProgramRecord:
    return record if isinstance(record, ProgramRecord) else ProgramRecord((record,))


def _rsa(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if set(arguments) != {"n", "mod", "base"}:
        raise SerializationError("RSA requires original n, mod and base fields")
    width, modulus, base = (
        _integer(arguments, field) for field in ("n", "mod", "base")
    )
    declaration = RSAPhaseEstimation(width, modulus, base)
    ports = tuple(declaration.source_signature)
    if registers is not None and registers != ports:
        raise SerializationError("Original RSA phase estimation has no public ports")
    if type(width) is not int:
        return FunctionRecord(name, (), registers=ports)
    if width < 1:
        return FunctionRecord(name, (), registers=ports)
    plus = x_basis_vector_factories()["qualtran.bloqs.basic_gates.x_basis.PlusState"](
        "plus", None, {}
    )
    one = basis_vector_factories()["qualtran.bloqs.basic_gates.z_basis.IntState"](
        "one", None, {"bitsize": width, "val": 1}
    )
    factory = modular_multiplication_factories()[
        "qualtran.bloqs.mod_arithmetic.mod_multiplication.CModMulK"
    ]
    by_factor: dict[SymbolicInt, ProgramRecord] = {}
    children = []
    factor = base % modulus
    for _ in range(2 * width):
        if factor not in by_factor:
            by_factor[factor] = _as_program(
                factory(
                    f"multiply_{factor}",
                    None,
                    {
                        "dtype": unsigned_register_type(width),
                        "mod": modulus,
                        "k": factor,
                    },
                )
            )
        children.append(by_factor[factor])
        factor = factor * factor % modulus
    terminal = _measure("measure_qft", None, {"n": 2 * width})
    program = rsa_phase_estimation_program(
        width,
        _as_program(plus),
        _as_program(one),
        tuple(children),
        ProgramRecord((terminal,)),
    )
    return _rename_program(program, name)


def rsa_component_factories() -> dict[str, ComponentFactory]:
    """Resolve both original identifiers with source-success graph boundaries."""
    return {_RSA: _rsa, **fourier_measurement_factories()}
