# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Codec composition adapted from Qualtran v0.7.0 data_types.py and _tensor_from_classical.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_tensor_from_classical.py

"""Connect native register metadata to existing numerical bit codecs."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.ir.program.registers import Register
from .bit_encoding import bit_array_from_bits, bit_array_to_bits, bit_from_bits
from .bounded_integer_encoding import (
    bounded_integer_array_from_bits,
    bounded_integer_array_to_bits,
    bounded_integer_from_bits,
)
from .classical_register_metadata import concrete_register_metadata
from .finite_field_metadata import resolve_finite_field_encoding
from .finite_field_polynomial_metadata import resolve_finite_field_polynomial_encoding
from .fixed_point_encoding import (
    fixed_point_array_from_bits,
    fixed_point_array_to_bits,
    fixed_point_from_bits,
)
from .integer_encoding import (
    integer_array_from_bits,
    integer_array_to_bits,
    integer_from_bits,
    vectorized_integer_array_from_bits,
    vectorized_integer_array_to_bits,
)
from .ones_complement_encoding import (
    ones_complement_array_from_bits,
    ones_complement_array_to_bits,
    ones_complement_from_bits,
)


def decode_register_bits(register: Register, bits: NDArray[Any]) -> object:
    """Decode a scalar or shaped register without adding validation or bit repair.

    Input layout is supplied by the classical tensor enumerator. Scalar bits
    are flattened, while arrays use the existing dtype-specific array codec.
    Metadata support here does not admit that type into executable Program IR.
    """
    if register.dtype.encoding in {"finite_field", "finite_field_polynomial"}:
        field = (
            resolve_finite_field_polynomial_encoding(register.dtype)
            if register.dtype.encoding == "finite_field_polynomial"
            else resolve_finite_field_encoding(register.dtype)
        )
        return (
            field.from_bits_array(bits)
            if register.shape
            else field.from_bits(list(bits.flat))
        )
    width, parameters = concrete_register_metadata(register)
    kind = register.dtype.encoding
    scalar = list(bits.flat)
    if kind == "bit":
        return bit_array_from_bits(bits) if register.shape else bit_from_bits(scalar)
    if kind == "uint" and "bound" in parameters:
        bound = parameters["bound"]
        return (
            bounded_integer_array_from_bits(bits, width, bound)
            if register.shape
            else bounded_integer_from_bits(scalar, width, bound)
        )
    if kind == "fixed_point":
        fractional, signed = parameters["fractional_width"], bool(parameters["signed"])
        return (
            fixed_point_array_from_bits(bits, width, fractional, signed=signed)
            if register.shape
            else fixed_point_from_bits(scalar, width, fractional, signed=signed)
        )
    if kind == "ones_complement":
        return (
            ones_complement_array_from_bits(bits, width)
            if register.shape
            else ones_complement_from_bits(scalar, width)
        )
    if kind == "montgomery" and register.shape:
        return vectorized_integer_array_from_bits(bits, width)
    if kind in {"uint", "signed", "opaque", "montgomery"}:
        return (
            integer_array_from_bits(bits, width, signed=kind == "signed")
            if register.shape
            else integer_from_bits(scalar, width, signed=kind == "signed")
        )
    raise ValueError(f"no native classical bit codec for {kind!r}")


def encode_register_bits(register: Register, values: NDArray[Any]) -> NDArray[np.uint8]:
    """Encode output values through the existing scalar-vectorized array codec.

    The source always invokes to_bits_array, even for scalar outputs. Preserve
    that behavior, NumPy result inference, empty-array failures and dtype errors.
    Montgomery values remain stored raw words; no residue conversion occurs.
    """
    if register.dtype.encoding == "finite_field_polynomial":
        return resolve_finite_field_polynomial_encoding(register.dtype).to_bits_array(
            values
        )
    if register.dtype.encoding == "finite_field":
        return resolve_finite_field_encoding(register.dtype).to_bits_array(values)
    width, parameters = concrete_register_metadata(register)
    kind = register.dtype.encoding
    if kind == "bit":
        return bit_array_to_bits(values)
    if kind == "uint" and "bound" in parameters:
        return bounded_integer_array_to_bits(values, width, parameters["bound"])
    if kind == "fixed_point":
        return fixed_point_array_to_bits(
            values,
            width,
            parameters["fractional_width"],
            signed=bool(parameters["signed"]),
        )
    if kind == "ones_complement":
        return ones_complement_array_to_bits(values, width)
    if kind == "montgomery":
        return vectorized_integer_array_to_bits(values, width)
    if kind in {"uint", "signed", "opaque"}:
        return integer_array_to_bits(values, width, signed=kind == "signed")
    raise ValueError(f"no native classical bit codec for {kind!r}")
