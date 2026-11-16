# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 mod_division.py and mod_division.ipynb.

"""Source metadata for the concrete and symbolic Kaliski inverse examples."""

from dataclasses import dataclass

from sympy import Symbol

from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.core.ir.program.register_types import (
    montgomery_register_type,
    opaque_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt


@dataclass(frozen=True)
class ModularInverseExample:
    width: SymbolicInt
    modulus: SymbolicInt

    def signature(self, *, adjoint: bool = False) -> Signature:
        """The target passes through and the history changes direction."""
        side = RegisterSide.LEFT if adjoint else RegisterSide.RIGHT
        return Signature(
            (
                Register("x", montgomery_register_type(self.width, self.modulus)),
                Register("junk", opaque_register_type(4 * self.width), side=side),
            )
        )

    def call_graph(self) -> SymbolicCallGraph[str]:
        """Declare the source root and its eight direct operation identities."""
        root = "KaliskiModInverse"
        n, p = self.width, self.modulus
        return symbolic_call_graph(
            root,
            {
                root: {
                    f"KaliskiIteration({n}, {p})": 2 * n,
                    f"BitwiseNot({n})": 1,
                    f"AddK({n}, {p + 1})": 1,
                    "X": 1,
                    f"XorK({n}, {p})": 2,
                    f"XorK({n}, 1)": 2,
                    f"C[XorK]({n}, {p}, 1)": 1,
                    f"C[XorK]({n}, {p - 1}, 1)": 2,
                }
            },
            max_depth=1,
        )


def modular_inverse_examples() -> dict[str, ModularInverseExample]:
    return {
        "kaliskimodinverse_example": ModularInverseExample(32, 10**9 + 7),
        "kaliskimodinverse_symbolic": ModularInverseExample(Symbol("n"), Symbol("p")),
    }
