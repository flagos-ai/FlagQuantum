# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Field declaration widths follow Qualtran v0.7.0 data_types.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Validate field storage separately from unresolved logical declarations."""

from typing import cast

from ...value_math import SymbolicInt, bit_length
from ...value_parameters import is_symbolic_value
from .registers import RegisterType, TypeParameter


def _field_register_width(dtype: RegisterType) -> int:
    """Validate structural field metadata independently of numerical construction."""
    width = dtype.quantum_bits
    if (
        type(width) is not int
        or width <= 0
        or type(dtype.classical_bits) is not int
        or dtype.classical_bits != 0
        or not isinstance(dtype.parameters, tuple)
        or any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            or type(pair[1]) is not int
            for pair in dtype.parameters
        )
    ):
        raise ValueError(f"unsupported executable register descriptor {dtype!r}")
    names = tuple(name for name, _ in dtype.parameters)
    parameters = {name: cast(int, value) for name, value in dtype.parameters}
    if dtype.encoding == "finite_field_polynomial":
        if not names or names[0] != "polynomial_degree":
            raise ValueError(f"unsupported executable register descriptor {dtype!r}")
        degree = parameters["polynomial_degree"]
        order = parameters.get("order")
        if degree < 0 or order is None or order < 2:
            raise ValueError(f"unsupported executable register descriptor {dtype!r}")
        coefficient_width = (order - 1).bit_length()
        if width != (degree + 1) * coefficient_width:
            raise ValueError(f"unsupported executable register descriptor {dtype!r}")
        coefficient = RegisterType(
            "finite_field", coefficient_width, parameters=dtype.parameters[1:]
        )
        _field_register_width(coefficient)
        return width
    base = ("characteristic", "degree", "order")
    polynomial = (
        "polynomial",
        "polynomial_field_characteristic",
        "polynomial_field_degree",
        "polynomial_field_primitive_element",
        "polynomial_field_modulus",
    )
    if names in (base, base + polynomial):
        characteristic, degree, order = (parameters[name] for name in base)
        if (
            characteristic >= 2
            and degree > 0
            and order == characteristic**degree
            and width == (order - 1).bit_length()
        ):
            return width
    raise ValueError(f"unsupported executable register descriptor {dtype!r}")


def _dimension(value: TypeParameter, minimum: int) -> SymbolicInt:
    if type(value) is int:
        if value < minimum:
            raise ValueError("field dimension is below its minimum")
        return value
    from sympy import Expr

    if isinstance(value, Expr) and value.is_integer is not False:
        if value.is_number and value < minimum:
            raise ValueError("field dimension is below its minimum")
        return value
    raise ValueError("field dimensions must be integral or symbolic")


def logical_field_register_width(dtype: RegisterType) -> SymbolicInt:
    """Keep canonical symbolic requirements without admitting physical storage."""
    if (
        type(dtype.classical_bits) is not int
        or dtype.classical_bits != 0
        or not isinstance(dtype.parameters, tuple)
        or any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not isinstance(pair[0], str)
            for pair in dtype.parameters
        )
    ):
        raise ValueError("malformed quantum field declaration")
    if not is_symbolic_value(
        dtype.quantum_bits, *(value for _, value in dtype.parameters)
    ):
        return _field_register_width(dtype)
    names = tuple(name for name, _ in dtype.parameters)
    parameters = dict(dtype.parameters)
    polynomial = (
        "polynomial",
        "polynomial_field_characteristic",
        "polynomial_field_degree",
        "polynomial_field_primitive_element",
        "polynomial_field_modulus",
    )
    base = ("characteristic", "degree", "order")
    extent: SymbolicInt = 1
    if dtype.encoding == "finite_field_polynomial":
        if not names or names[0] != "polynomial_degree":
            raise ValueError("polynomial declaration requires its degree first")
        extent = _dimension(parameters["polynomial_degree"], 0) + 1
        names = names[1:]
    if names not in (base, base + polynomial):
        raise ValueError("noncanonical field declaration parameters")
    if names == base + polynomial and any(
        type(parameters[name]) is not int for name in polynomial
    ):
        raise ValueError("field polynomial metadata must remain concrete integers")
    characteristic = _dimension(parameters["characteristic"], 2)
    degree = _dimension(parameters["degree"], 1)
    order = _dimension(parameters["order"], 2)
    if order != characteristic**degree:
        raise ValueError("field order must equal characteristic raised to degree")
    width = extent * bit_length(order - 1)
    if _dimension(dtype.quantum_bits, 1) != width:
        raise ValueError(
            "field declaration width must match its exact order and extent"
        )
    return width
