# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 arithmetic/subtraction.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/subtraction.py

"""Own subtraction type defaults and direct models before physical lowering."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_types import unsigned_register_type
from ....core.ir.program.registers import RegisterType
from ....core.ir.program.signatures import Signature
from ....core.value_math import SymbolicInt
from ....core.value_parameters import is_symbolic_value
from ..._component_trace.arithmetic_call import ArithmeticCall, ArithmeticEncoding
from ..._component_trace.resource_generalization import CliffordCall, OnEachCall
from .subtraction_component import subtract_from_program, subtraction_program

_ENCODINGS: dict[str, ArithmeticEncoding] = {
    "uint": "unsigned",
    "signed": "twos_complement",
    "montgomery": "montgomery",
}


def _validate_integer_type(dtype: RegisterType) -> None:
    if (
        dtype.encoding not in _ENCODINGS
        or dtype.classical_bits != 0
        or any(
            name != "modulus" or dtype.encoding != "montgomery"
            for name, _ in dtype.parameters
        )
    ):
        raise ValueError(
            "Subtraction supports quantum signed, unsigned and Montgomery types"
        )


@dataclass(frozen=True, init=False)
class SubtractionDeclaration:
    """Retain source dtype defaults, unsigned views and ordered direct calls."""

    a_dtype: RegisterType
    b_dtype: RegisterType

    def __init__(
        self, a_dtype: RegisterType, b_dtype: RegisterType | None = None
    ) -> None:
        target = a_dtype if b_dtype is None else b_dtype
        _validate_integer_type(a_dtype)
        _validate_integer_type(target)
        if (
            not is_symbolic_value(a_dtype.num_qubits)
            and a_dtype.num_qubits > target.num_qubits
        ):
            raise ValueError(
                "Left dtype width must be less than or equal to right dtype width"
            )
        object.__setattr__(self, "a_dtype", a_dtype)
        object.__setattr__(self, "b_dtype", target)

    @property
    def dtype(self) -> RegisterType:
        if self.a_dtype != self.b_dtype:
            raise ValueError(
                "Subtraction dtype requires both operands to have the same dtype"
            )
        return self.a_dtype

    @property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(a=self.a_dtype, b=self.b_dtype)

    @property
    def a_dtype_as_unsigned(self) -> RegisterType:
        return (
            unsigned_register_type(self.a_dtype.num_qubits)
            if self.a_dtype.encoding == "signed"
            else self.a_dtype
        )

    @property
    def b_dtype_as_unsigned(self) -> RegisterType:
        return (
            unsigned_register_type(self.b_dtype.num_qubits)
            if self.b_dtype.encoding == "signed"
            else self.b_dtype
        )

    def direct_calls(self) -> dict[Hashable, SymbolicInt]:
        width = self.b_dtype.num_qubits
        delta = width - self.a_dtype.num_qubits
        calls: dict[Hashable, SymbolicInt] = {
            OnEachCall(CliffordCall("x", 1), width): 3,
            ArithmeticCall("add", (width, width)): 1,
        }
        if delta:
            calls[("allocate", delta)] = 1
            calls[("free", delta)] = 1
            if self.a_dtype.encoding == "signed":
                calls[("multi_target_cx", delta)] = 2
        return calls

    def program(self) -> ProgramRecord:
        """Delegate finite descriptors and clean prefix/carry ownership unchanged."""
        return subtraction_program(self.a_dtype, self.b_dtype)


@dataclass(frozen=True)
class SubtractFromDeclaration:
    """Retain equal-width typed b-a through complement/Add/complement."""

    dtype: RegisterType

    def __post_init__(self) -> None:
        _validate_integer_type(self.dtype)

    @property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(a=self.dtype, b=self.dtype)

    def direct_calls(self) -> dict[Hashable, SymbolicInt]:
        width, encoding = self.dtype.num_qubits, _ENCODINGS[self.dtype.encoding]
        return {
            ArithmeticCall("bitwise_not", (width,), (encoding,)): 2,
            ArithmeticCall("add", (width, width), (encoding, encoding)): 1,
        }

    def program(self) -> ProgramRecord:
        """Delegate finite typed subtraction and shared Add carry erasure."""
        return subtract_from_program(self.dtype)
