# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ctrl_spec_and.ipynb and ctrl_spec_and.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mcmt/ctrl_spec_and.py

"""Grouped equality predicates from the locked control-specification tutorial."""

from __future__ import annotations

from dataclasses import dataclass

from sympy import Symbol

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.resource_decomposition import (
    decomposed_resource_callees,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    ConjunctionCall,
    ControlPredicateCall,
)
from flagquantum.compiler.registers.control.component_predicate import (
    control_predicate_signature,
)
from flagquantum.compiler.registers.control.control_predicate import (
    ControlRegister,
    control_predicate_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt, ssum


@dataclass(frozen=True)
class PredicateExample:
    """Pass through control groups and expose cleanly allocated prefix outputs.

    Append binds groups in row-major order with big-endian words, then junk and
    target. Erase consumes these outputs in the reverse measured ladder and
    returns them to zero. Unknown activation is available for resource queries.
    """

    controls: tuple[Register, ...]
    patterns: tuple[ControlRegister, ...] | None

    @property
    def width(self) -> SymbolicInt:
        return ssum(register.total_qubits() for register in self.controls)

    @property
    def signature(self) -> Signature:
        return control_predicate_signature(self.controls)

    @property
    def activation(self) -> tuple[int, ...] | SymbolicInt:
        if self.patterns is None:
            return self.width
        return tuple(bit for group in self.patterns for bit in group.bits())

    def source_child(self) -> ConjunctionCall:
        """Return the source's sole And or MultiAnd call identity."""
        ((child, _),) = decomposed_resource_callees(
            ControlPredicateCall(self.activation), None
        )
        assert isinstance(child, ConjunctionCall)
        if self.patterns is None and self.width == 2:
            return ConjunctionCall((Symbol("cv1"), Symbol("cv2")))
        return child

    def append_to(self, circuit: Circuit) -> Circuit:
        return self._append(circuit, erase=False)

    def adjoint(self) -> PredicateErase:
        return PredicateErase(self)

    def _append(self, circuit: Circuit, *, erase: bool) -> Circuit:
        n = self.width
        if self.patterns is None or not isinstance(n, int):
            raise ValueError("Predicate append requires a concrete activation")
        if circuit.n_qubits < 2 * n - 1:
            raise ValueError("Predicate append requires its clean prefix output wires")
        groups = []
        offset = 0
        for group in self.patterns:
            end = offset + len(group.bits())
            groups.append(tuple(range(offset, end)))
            offset = end
        body = control_predicate_ir(
            tuple(groups),
            self.patterns,
            2 * n - 2,
            tuple(range(n, 2 * n - 2)),
            classical_bits=tuple(range(n - 1)) if erase else (),
            uncompute=erase,
        )
        return circuit.compose(
            CircuitIR(circuit.n_qubits, body, metadata={"batch_size": circuit.bsz})
        )


@dataclass(frozen=True)
class PredicateErase:
    forward: PredicateExample

    def append_to(self, circuit: Circuit) -> Circuit:
        return self.forward._append(circuit, erase=True)


def predicate_examples() -> dict[str, PredicateExample]:
    """Return all seven original example factories in declaration order."""
    bit = quantum_bit_type()
    uint, sint = unsigned_register_type(4), signed_register_type(4)
    return {
        "ctrl_on_int": PredicateExample(
            (Register("ctrl_0", uint, (1,)),), (ControlRegister(4, (5,)),)
        ),
        "ctrl_on_bits": PredicateExample(
            (Register("ctrl_0", bit, (4,)),), (ControlRegister(1, (0, 1, 0, 1)),)
        ),
        "ctrl_on_nd_bits": PredicateExample(
            (Register("ctrl_0", bit, (2, 2)),), (ControlRegister(1, (0, 1, 1, 0)),)
        ),
        "ctrl_on_multiple_values": PredicateExample(
            (Register("ctrl_0", uint, (2,)), Register("ctrl_1", sint, (2,))),
            (
                ControlRegister(4, (5, 12)),
                ControlRegister(4, (2, -2), "twos_complement"),
            ),
        ),
        "ctrl_on_symbolic_cv": PredicateExample((Register("ctrl_0", bit, (2,)),), None),
        "ctrl_on_symbolic_cv_multi": PredicateExample(
            (Register("ctrl_0", bit, (3,)),), None
        ),
        "ctrl_on_symbolic_n_ctrls": PredicateExample(
            (Register("ctrl_0", bit, (Symbol("n"),)),), None
        ),
    }
