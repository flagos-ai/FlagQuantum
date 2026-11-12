# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _GFPoly and QGFPoly/CGFPoly numerical methods.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Finite-field polynomial codecs, independent of executable register admission."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from itertools import product
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..core.ir.program.registers import RegisterType, TypeParameter
from ..core.value_math import SymbolicInt
from .finite_field_encoding import FiniteFieldEncoding
from .value_encoding import decode_value_array, encode_value_array


@dataclass(frozen=True)
class FiniteFieldPolynomialEncoding:
    """Represent descending coefficients over one existing numerical field.

    Galois polynomial and field objects stay at this optional numerical boundary.
    Construction remains lazy, with no new degree or field normalization. This
    codec does not declare a Core dtype or authorize physical execution.
    """

    degree: SymbolicInt
    field: FiniteFieldEncoding

    @cached_property
    def bitsize(self) -> SymbolicInt:
        """Return coefficient width times degree plus one without coercion."""
        return self.field.bitsize * (self.degree + 1)

    def register_type(self, *, classical: bool = False) -> RegisterType:
        """Project numerical declaration into existing neutral register metadata.

        Reuse the field projection so polynomial/coefficient-field identity stays
        with its existing owner. This does not admit a polynomial Program port.
        """
        field_type = self.field.register_type()
        parameters: tuple[tuple[str, TypeParameter], ...] = (
            (("storage", "classical"),) if classical else ()
        )
        parameters += (("polynomial_degree", self.degree),) + field_type.parameters
        width = self.bitsize
        return RegisterType(
            "finite_field_polynomial",
            0 if classical else width,
            width if classical else 0,
            parameters,
        )

    def get_domain(self) -> Iterable[Any]:
        """Yield source Cartesian coefficient order, including leading zeros."""
        galois = import_module("galois")
        for coefficients in product(
            self.field.gf_type.elements, repeat=cast(int, self.degree + 1)
        ):
            yield galois.Poly(
                self.field.gf_type(coefficients), field=self.field.gf_type
            )

    def to_gf_coefficients(self, polynomial: Any) -> Any:
        """Left-pad descending coefficients to the declared degree plus one."""
        coefficients = self.field.gf_type.Zeros(self.degree + 1)
        coefficients[self.degree - polynomial.degree :] = polynomial.coeffs
        return coefficients

    def from_gf_coefficients(self, coefficients: Any) -> Any:
        """Construct the original field polynomial without a length guard."""
        return import_module("galois").Poly(coefficients, field=self.field.gf_type)

    def assert_valid_val(self, value: object, debug_str: str = "val") -> None:
        """Check polynomial type, nominal field identity, then degree in order."""
        galois = import_module("galois")
        if not isinstance(value, galois.Poly):
            raise ValueError(f"{debug_str} should be a {galois.Poly}, not {value!r}")
        polynomial = cast(Any, value)
        if polynomial.field is not self.field.gf_type:
            raise ValueError(
                f"{debug_str} should be defined over {self.field.gf_type}, not {polynomial.field}"
            )
        if polynomial.degree > self.degree:
            raise ValueError(
                f"{debug_str} should have a degree <= {self.degree}, not {polynomial.degree}"
            )

    def assert_valid_val_array(
        self, values: NDArray[Any], debug_str: str = "val"
    ) -> None:
        """Reuse scalar validation in source flattened order, including empty arrays."""
        for value in values.reshape(-1):
            self.assert_valid_val(value, debug_str)

    def to_bits(self, value: object) -> list[int]:
        """Validate then reuse the field array codec, retaining its original failures."""
        self.assert_valid_val(value)
        return (
            self.field.to_bits_array(self.to_gf_coefficients(value))
            .reshape(-1)
            .tolist()
        )

    def from_bits(self, bits: Sequence[int] | NDArray[Any]) -> Any:
        """Reshape big-endian coefficient words before the field array decoder."""
        words = np.array(bits).reshape((int(self.degree) + 1, int(self.field.bitsize)))
        return self.from_gf_coefficients(self.field.from_bits_array(words))

    def to_bits_array(self, values: NDArray[Any]) -> NDArray[np.uint8]:
        """Delegate the source default scalar vectorizer without type repair."""
        return encode_value_array(values, self.to_bits)

    def from_bits_array(self, bits: NDArray[np.uint8]) -> NDArray[Any]:
        """Delegate source default output inference and empty core-dimension errors."""
        return decode_value_array(bits, self.from_bits)
