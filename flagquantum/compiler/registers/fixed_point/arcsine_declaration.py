# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/trigonometric/arcsin.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/trigonometric/arcsin.py

"""Own the original arcsine parameters, typed ports and independent estimate."""

from dataclasses import dataclass

from ....core.ir.program.register_types import fixed_point_register_type
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ....core.value_parameters import is_symbolic_value
from .trigonometric_resource_model import source_arcsine_toffoli_count


@dataclass(frozen=True)
class ArcsineDeclaration:
    """Retain unsigned fixed-point storage and original iteration/degree defaults."""

    bitsize: SymbolicInt
    num_frac: SymbolicInt
    num_iters: SymbolicInt = 4
    degree: SymbolicInt = 4

    def __post_init__(self) -> None:
        if (
            not is_symbolic_value(self.num_frac, self.bitsize)
            and self.num_frac > self.bitsize
        ):
            raise ValueError(
                "Fractional width must be less than or equal to word width"
            )

    @property
    def signature(self) -> Signature:
        word = fixed_point_register_type(self.bitsize, self.num_frac)
        return Signature((Register("x", word), Register("result", word)))

    def direct_calls(self) -> dict[str, SymbolicInt]:
        """Declare original Toffoli demand without inventing a source gate body."""
        return {
            "toffoli": source_arcsine_toffoli_count(
                self.bitsize,
                self.num_frac,
                iterations=self.num_iters,
                degree=self.degree,
            )
        }
