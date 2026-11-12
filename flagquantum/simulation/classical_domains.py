# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Domain composition adapted from Qualtran v0.7.0 dtype get_classical_domain methods.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Connect native register declarations to existing classical value domains."""

from collections.abc import Iterable

from ..core.ir.program.registers import Register
from .bit_encoding import bit_value_domain
from .bounded_integer_values import bounded_integer_domain
from .classical_register_metadata import value_register_metadata
from .finite_field_metadata import resolve_finite_field_encoding
from .finite_field_polynomial_metadata import resolve_finite_field_polynomial_encoding
from .fixed_point_encoding import fixed_point_domain
from .integer_values import integer_word_domain
from .montgomery_values import montgomery_value_domain
from .ones_complement_encoding import ones_complement_domain


def register_classical_domain(register: Register) -> Iterable[object]:
    """Return the declared scalar domain without repairing it to the bit capacity.

    Fields retain typed elements, bounded values use their exclusive bound,
    fixed point yields stored integers, and Montgomery omits zero when a modulus
    is declared. Opaque words have no unambiguous domain. Shapes do not change a
    dtype's scalar domain; the truth-table owner separately refuses array inputs.
    Symbolic widths and parameters reach their existing numerical owners without
    evaluation; this adapter does not expand executable Program admission.
    """
    kind = register.dtype.encoding
    if kind == "finite_field_polynomial":
        return resolve_finite_field_polynomial_encoding(register.dtype).get_domain()
    if kind == "finite_field":
        return resolve_finite_field_encoding(register.dtype).get_domain()
    width, parameters = value_register_metadata(register)
    if kind == "opaque":
        raise TypeError("Ambiguous domain for an opaque register. Use a specific type.")
    if kind == "bit":
        return bit_value_domain()
    if kind == "uint" and "bound" in parameters:
        return bounded_integer_domain(width, parameters["bound"])
    if kind == "fixed_point":
        return fixed_point_domain(
            width, parameters["fractional_width"], signed=bool(parameters["signed"])
        )
    if kind == "ones_complement":
        return ones_complement_domain(width)
    if kind == "montgomery":
        return montgomery_value_domain(width, parameters.get("modulus"))
    if kind in {"uint", "signed"}:
        return integer_word_domain(width, signed=kind == "signed")
    raise ValueError(f"no native classical domain for {kind!r}")
