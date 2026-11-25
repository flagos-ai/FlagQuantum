# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ZPowProgrammedAncilla and ZPowUsingProgrammedAncilla.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/programmable_ancilla_rotation.py

"""Declared programmed-rotation resources before adaptive gate lowering."""

from dataclasses import dataclass, fields
from math import isfinite

from ....core.ir.program.register_types import quantum_bit_type
from ..._component_trace.call_graph import SymbolAllocator
from ..._component_trace.qec_gate_cost import query_qec_gate_cost
from ..._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    MeasurementCall,
    ResourceCall,
    RotationCall,
)
from ..._component_trace.resource_model import ComponentCost
from ..._component_trace.symbolic_math import SymbolicInt, SymbolicReal, ceil, log2
from ..._component_trace.terminal_t_cost import estimate_terminal_t_cost
from ..resources.symbolic_resource_model import SymbolicResourceFormula


@dataclass(frozen=True)
class ProgrammedAncillaCall:
    """The exponent and tolerance distinguish prepared resource states."""

    exponent: SymbolicReal
    precision: SymbolicReal = 1e-11

    def direct_calls(self) -> dict[ResourceCall, SymbolicInt]:
        return {
            BookkeepingCall("allocate", quantum_bit_type()): 1,
            CliffordCall("h", 1): 1,
            RotationCall("zpow", self.exponent, self.precision): 1,
        }


@dataclass(frozen=True)
class ProgrammedRotationModel:
    """Worst-case source declaration for one bounded programmed rotation."""

    exponent: SymbolicReal
    precision: SymbolicReal = 1e-11
    rounds: SymbolicInt = 2
    apply_final_correction: bool = False

    def __post_init__(self) -> None:
        rounds = self.rounds
        if isinstance(rounds, (int, float)) and (type(rounds) is not int or rounds < 0):
            raise ValueError("Programmed rotation requires nonnegative integer rounds")
        if type(self.apply_final_correction) is not bool:
            raise ValueError("Programmed rotation correction flag must be boolean")

    @classmethod
    def from_failure_probability(
        cls,
        exponent: SymbolicReal,
        *,
        max_fail_probability: SymbolicReal,
        precision: SymbolicReal = 1e-11,
    ) -> "ProgrammedRotationModel":
        p = max_fail_probability
        if isinstance(p, (int, float)) and (not isfinite(p) or not 0 < p <= 1):
            raise ValueError("Programmed rotation failure bound must lie in (0, 1]")
        return cls(exponent, precision, ceil(log2(1 / p)))

    def direct_calls(
        self, *, allocator: SymbolAllocator | None = None
    ) -> dict[ResourceCall | ProgrammedAncillaCall, SymbolicInt]:
        """Retain fresh angle identities throughout a containing graph query."""
        n = self.rounds
        calls: dict[ResourceCall | ProgrammedAncillaCall, SymbolicInt] = {
            CliffordCall("cx", 2): n,
            CliffordCall("x", 1): n,
            MeasurementCall("z"): n,
        }
        if n == 0 and not self.apply_final_correction:
            return calls
        tolerance = self.precision / (n + int(self.apply_final_correction))
        if isinstance(n, int):
            for i in range(n):
                child = ProgrammedAncillaCall(2**i * self.exponent, tolerance)
                calls[child] = calls.get(child, 0) + 1
        else:
            symbols = allocator if allocator is not None else SymbolAllocator()
            calls[ProgrammedAncillaCall(symbols.new_symbol(r"\phi"), tolerance)] = n
        if self.apply_final_correction:
            calls[RotationCall("zpow", 2**n * self.exponent, tolerance)] = 1
        return calls

    def terminal_calls(
        self, *, allocator: SymbolAllocator | None = None
    ) -> dict[ResourceCall, SymbolicInt]:
        terminals: dict[ResourceCall, SymbolicInt] = {}
        for call, count in self.direct_calls(allocator=allocator).items():
            children = (
                call.direct_calls()
                if isinstance(call, ProgrammedAncillaCall)
                else {call: 1}
            )
            for child, repetitions in children.items():
                terminals[child] = terminals.get(child, 0) + repetitions * count
        return terminals

    def resource_formula(self) -> SymbolicResourceFormula:
        root = ComponentCost("ZPowUsingProgrammedAncilla", 2)
        gates = query_qec_gate_cost(root, declared_calls={root: self.terminal_calls()})
        return SymbolicResourceFormula(
            logical_qubits=root.qubits,
            **{field.name: getattr(gates, field.name) for field in fields(gates)},
        )

    def synthesis_t_count(self) -> SymbolicInt:
        return estimate_terminal_t_cost(self.terminal_calls())
