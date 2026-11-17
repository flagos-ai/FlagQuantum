# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ScaledChebyshevPolynomial.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/chebyshev_polynomial.py

"""Scale Chebyshev coefficients and use the existing signed block composition."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ...compiler._component_trace.symbolic_math import SymbolicInt, SymbolicReal
from ...core.ir.program.register_types import opaque_register_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ._block_metadata import _BlockMetadata
from .black_box_prepare import BlackBoxPrepare
from .chebyshev_block_encoding import ChebyshevBlockEncoding
from .linear_combination_block_encoding import LinearCombinationBlockEncoding


@dataclass(frozen=True)
class ScaledChebyshevBlockEncoding:
    """Use the source polynomial coefficient transform with native LCU terms."""

    block_encoding: _BlockMetadata
    order: int
    coefficient_bits: SymbolicInt = 5

    def __post_init__(self) -> None:
        # Apply the same order and signal admission as the unscaled term.
        ChebyshevBlockEncoding(self.block_encoding, self.order)

    @cached_property
    def linear_combination(
        self,
    ) -> LinearCombinationBlockEncoding | ChebyshevBlockEncoding:
        """Return the native child with its existing operation binding interface."""
        if self.order <= 1:
            return ChebyshevBlockEncoding(self.block_encoding, self.order)

        alpha = self.block_encoding.alpha
        if not isinstance(alpha, (int, float)):
            raise ValueError(
                "Scaled Chebyshev coefficients need a concrete normalization"
            )
        coefficients = np.polynomial.chebyshev.cheb2poly([0] * self.order + [1])
        for index in range(len(coefficients)):
            coefficients[index] *= alpha**index
        scaled = np.polynomial.chebyshev.poly2cheb(coefficients)
        return LinearCombinationBlockEncoding(
            tuple(
                ChebyshevBlockEncoding(self.block_encoding, index)
                for index in range(self.order + 1)
            ),
            tuple(float(value) for value in scaled),
            self.coefficient_bits,
        )

    @property
    def system_bitsize(self) -> int:
        return self.linear_combination.system_bitsize

    @property
    def ancilla_bitsize(self) -> int:
        return self.linear_combination.ancilla_bitsize

    @property
    def resource_bitsize(self) -> int:
        return self.linear_combination.resource_bitsize

    @cached_property
    def signature(self) -> Signature:
        """Declare packed system, ancilla and resource ports in source order."""
        return Signature.build_from_dtypes(
            system=opaque_register_type(self.system_bitsize),
            ancilla=opaque_register_type(self.ancilla_bitsize),
            resource=opaque_register_type(self.resource_bitsize),
        )

    @property
    def target_registers(self) -> tuple[Register, ...]:
        return tuple(self.signature.rights())

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return (self.signature.get_right("ancilla"),) if self.ancilla_bitsize else ()

    @property
    def junk_registers(self) -> tuple[Register, ...]:
        return (self.signature.get_right("resource"),) if self.resource_bitsize else ()

    @property
    def alpha(self) -> SymbolicReal:
        return self.linear_combination.alpha

    @property
    def epsilon(self) -> SymbolicReal:
        return self.linear_combination.epsilon

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return self.linear_combination.signal_state
