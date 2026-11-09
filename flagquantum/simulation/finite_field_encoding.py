# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _GF and QGF/CGF polynomial construction.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Lazy finite-field numerical codecs with Simulation-owned galois objects."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from importlib import import_module
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..core.value_math import SymbolicInt, bit_length
from ..core.value_parameters import is_symbolic_value
from .integer_encoding import (
    integer_array_from_bits,
    integer_from_bits,
    integer_to_bits,
)
from .value_encoding import encode_value_array


class _DefaultPolynomial(Enum):
    OMITTED = auto()


@dataclass(frozen=True, eq=False)
class FiniteFieldEncoding:
    """Retain field construction and lazy codecs without admitting a Core dtype.

    Polynomial and FieldArray types are dynamic galois classes. Their Any types
    are confined to this optional numerical-library boundary. Scalar decoding
    returns a field scalar and array decoding returns a field array.
    """

    characteristic: SymbolicInt
    degree: SymbolicInt
    irreducible_poly: Any = _DefaultPolynomial.OMITTED

    def __post_init__(self) -> None:
        polynomial = self.irreducible_poly
        if polynomial is _DefaultPolynomial.OMITTED:
            if is_symbolic_value(self.characteristic, self.degree):
                polynomial = None
            else:
                polynomial = (
                    import_module("galois")
                    .GF(
                        int(self.characteristic),
                        int(self.degree),
                        compile="python-calculate",
                    )
                    .irreducible_poly
                )
        # Import precedes the None check, exactly as in the source converter.
        galois = import_module("galois")
        if polynomial is not None and not isinstance(polynomial, galois.Poly):
            polynomial = galois.Poly.Degrees(polynomial)
        object.__setattr__(self, "irreducible_poly", polynomial)

    @cached_property
    def order(self) -> SymbolicInt:
        """Compute field order using the declared operands without coercion."""
        return self.characteristic**self.degree

    @cached_property
    def bitsize(self) -> SymbolicInt:
        """Resolve the original shared order-minus-one bit-width expression."""
        return bit_length(self.order - 1)

    @cached_property
    def gf_type(self) -> Any:
        """Resolve the numerical field class only when a consumer requests it."""
        polynomial = self.irreducible_poly if self.degree > 1 else None
        return import_module("galois").GF(
            int(self.characteristic),
            int(self.degree),
            irreducible_poly=polynomial,
            verify=False,
            repr="poly",
            compile="python-calculate",
        )

    def get_domain(self) -> Iterable[Any]:
        """Yield the numerical field elements in their original integer order."""
        yield from self.gf_type.elements

    def assert_valid_val(self, value: object, debug_str: str = "val") -> None:
        """Require the resolved field class without coercing ordinary integers."""
        if not isinstance(value, self.gf_type):
            raise ValueError(f"{debug_str} should be a {self.gf_type}, not {value!r}")

    def assert_valid_val_array(
        self, values: NDArray[Any], debug_str: str = "val"
    ) -> None:
        """Retain range-only array validation and the original check order."""
        if np.any(values < 0):
            raise ValueError(f"Negative classical values encountered in {debug_str}")
        if np.any(values >= self.order):
            raise ValueError(f"Too-large classical values encountered in {debug_str}")

    def to_bits(self, value: object) -> list[int]:
        """Validate the field scalar before shared big-endian integer encoding."""
        self.assert_valid_val(value)
        return integer_to_bits(int(cast(Any, value)), self.bitsize)

    def from_bits(self, bits: Sequence[int]) -> Any:
        """Decode the original unchecked bit string into a field scalar."""
        # Resolve the field before decoding, preserving Python call evaluation.
        field = self.gf_type
        return field(integer_from_bits(bits, self.bitsize))

    def to_bits_array(self, values: NDArray[Any]) -> NDArray[np.uint8]:
        """Retain the default scalar vectorizer and its field-type failures."""
        return encode_value_array(values, self.to_bits)

    def from_bits_array(self, bits: NDArray[np.uint8]) -> Any:
        """Reuse unsigned array decoding before the original field conversion."""
        field = self.gf_type
        return field(integer_array_from_bits(bits, self.bitsize))
