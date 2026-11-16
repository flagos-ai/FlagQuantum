# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 mod_multiplication.py and its notebook.

"""Register signatures and direct calls for the modular multiplication tutorial."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sympy import Symbol

from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt

MultiplicationKind = Literal["ModDbl", "CModMulK", "DirtyOutOfPlaceMontgomeryModMul"]


@dataclass(frozen=True)
class ModularMultiplicationExample:
    kind: MultiplicationKind
    width: SymbolicInt
    modulus: SymbolicInt
    constant: SymbolicInt | None = None
    window_size: int | None = None

    def signature(self, *, uncompute: bool = False) -> Signature:
        if self.window_size is not None:
            dtype = montgomery_register_type(self.width, self.modulus)
            side = RegisterSide.LEFT if uncompute else RegisterSide.RIGHT
            windows = (self.width + self.window_size - 1) // self.window_size
            indices = montgomery_register_type(windows * self.window_size, self.modulus)
            return Signature(
                (
                    Register("x", dtype),
                    Register("y", dtype),
                    Register("target", dtype, side=side),
                    Register("qrom_indices", indices, side=side),
                    Register("reduced", quantum_bit_type(), side=side),
                )
            )
        x = Register("x", unsigned_register_type(self.width))
        if self.kind == "CModMulK":
            return Signature((Register("ctrl", quantum_bit_type()), x))
        return Signature((x,))

    def call_graph(self) -> SymbolicCallGraph[str]:
        """Declare the original forward cost graph at tutorial depth one."""
        n, p = self.width, self.modulus
        calls: dict[str, SymbolicInt]
        if self.window_size is not None:
            w = self.window_size
            calls = {
                f"C[AddK](QUInt({n}), {p})": 1,
                f"LessThanConstant({n}, {p})": 1,
                "X": 1,
                f"SingleWindowModMul({w}, {n}, {p})": (n + w - 1) // w,
            }
        elif self.kind == "CModMulK":
            calls = {f"CtrlScaleModAdd({n}, {p}, _k0)": 2, f"CSwap({n})": 1}
        else:
            calls = {
                f"AddK(QInt({n + 2}), {-p})": 1,
                f"C[AddK](QUInt({n + 1}), {p})": 1,
                "CNOT": 1,
                "X": 2,
            }
        return symbolic_call_graph(self.kind, {self.kind: calls}, max_depth=1)


def modular_multiplication_examples() -> dict[str, ModularMultiplicationExample]:
    k, modulus, width = (Symbol(name) for name in ("k", "N", "n_x"))
    return {
        "moddbl_small": ModularMultiplicationExample("ModDbl", 4, 13),
        "moddbl_large": ModularMultiplicationExample("ModDbl", 32, 10**9 + 7),
        "modmul_symb": ModularMultiplicationExample("CModMulK", width, modulus, k),
        "modmul": ModularMultiplicationExample("CModMulK", 8, 13 * 17, 123),
        "dirtyoutofplacemontgomerymodmul_small": ModularMultiplicationExample(
            "DirtyOutOfPlaceMontgomeryModMul", 6, 7, window_size=2
        ),
        "dirtyoutofplacemontgomerymodmul_medium": ModularMultiplicationExample(
            "DirtyOutOfPlaceMontgomeryModMul", 16, 2**15 - 1, window_size=4
        ),
    }
