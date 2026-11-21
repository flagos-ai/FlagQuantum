# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 RSAPhaseEstimate.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/rsa/rsa_phase_estimate.py

"""Compose the source's modular exponentiation and Fourier readout."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from math import ceil, log2

from numpy.random import RandomState

from ...circuit import Circuit
from ...compiler._component_trace.call_graph import (
    SymbolAllocator,
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...core.value_parameters import is_symbolic_value
from ...dynamic import DynamicCircuit
from .measure_qft import MeasureQFT
from .modular_exponentiation import ModularExponentiation, shor_factoring_parameters

PhaseCall = tuple[str, SymbolicInt] | tuple[str, SymbolicInt, SymbolicInt, SymbolicInt]


def generalize_rsa_multiplier_call(call: Hashable) -> Hashable:
    """Replace only a controlled modular multiplier's factor, as in the source."""
    if isinstance(call, tuple) and len(call) == 4 and call[0] == "CModMulK":
        from sympy import Symbol

        return (call[0], call[1], Symbol("k_exp"), call[3])
    return call


@dataclass(frozen=True)
class _BoundRSAPhaseEstimation:
    wires: tuple[int, ...]
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError("RSA phase estimation requires a DynamicCircuit")
        local = CircuitIR(
            len(self.wires), self.instructions, metadata={"batch_size": circuit.bsz}
        ).validate()
        return circuit.compose(local, qubits=self.wires)


@dataclass(frozen=True)
class RSAPhaseEstimation:
    """Prepare 2n fresh phase qubits and measure after modular exponentiation.

    Phase and target words are big-endian. The caller supplies both in zero,
    with clean arithmetic work and disjoint readout and arithmetic slots.
    The modular result stays on the caller's target wires after readout;
    the source declares a dirty release for that word.
    """

    width: SymbolicInt
    modulus: SymbolicInt
    base: SymbolicInt

    def __post_init__(self) -> None:
        if not is_symbolic_value(self.width, self.modulus) and self.width != ceil(
            log2(int(self.modulus))
        ):
            raise ValueError("RSA phase width must equal ceil(log2(modulus))")

    @classmethod
    def for_shor(
        cls,
        modulus: SymbolicInt,
        base: SymbolicInt | None = None,
        *,
        random_state: RandomState | None = None,
    ) -> RSAPhaseEstimation:
        """Reuse the source factoring width and seeded base selection."""
        width, base = shor_factoring_parameters(modulus, base, random_state)
        return cls(width, modulus, base)

    @property
    def source_signature(self) -> Signature:
        return Signature(())

    @property
    def phase_width(self) -> SymbolicInt:
        return 2 * self.width

    def source_callees(self, allocator: SymbolAllocator) -> dict[Hashable, SymbolicInt]:
        """Declare the original generic factor and smaller Fourier model width."""
        factor = allocator.new_symbol("k")
        multiplier: PhaseCall = ("CModMulK", self.width, factor, self.modulus)
        return {("MeasureQFT", self.width): 1, multiplier: self.phase_width}

    def source_call_graph(self) -> SymbolicCallGraph[Hashable]:
        """Retain complete source identities without instantiating physical gates."""
        return symbolic_call_graph(self, {self: self.source_callees(SymbolAllocator())})

    def _modular_exponentiation(self) -> ModularExponentiation:
        return ModularExponentiation(
            self.base, self.modulus, self.phase_width, self.width
        )

    def workspace_requirements(self) -> tuple[int, int]:
        """Count arithmetic scratch and reusable feedback slots."""
        return self._modular_exponentiation().workspace_requirements()

    def make_on(
        self,
        *,
        phase: tuple[int, ...],
        target: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        result_bits: tuple[int, ...],
    ) -> _BoundRSAPhaseEstimation:
        modexp = self._modular_exponentiation()
        q, c = modexp.workspace_requirements()
        if (
            len(phase) != self.phase_width
            or len(result_bits) != len(phase)
            or len(target) != self.width
        ):
            raise ValueError(
                "RSA phase estimation requires 2n phase and result bits and n target bits"
            )
        if len(workspace) != q or len(classical_bits) != c:
            raise ValueError(
                "RSA phase estimation requires exact arithmetic workspace sizes"
            )
        wires = (*phase, *target, *workspace)
        for values in (wires, (*classical_bits, *result_bits)):
            if len(set(values)) != len(values) or any(
                type(v) is not int or v < 0 for v in values
            ):
                raise ValueError(
                    "RSA phase estimation requires distinct nonnegative wires and slots"
                )

        e, n = len(phase), len(target)
        exponent = tuple(range(e))
        result = tuple(range(e, e + n))
        scratch = tuple(range(e + n, len(wires)))
        circuit = DynamicCircuit(len(wires))
        for wire in exponent:
            circuit.h(wire)
        circuit.append(
            modexp.make_on(
                exponent=exponent,
                target=result,
                workspace=scratch,
                classical_bits=classical_bits,
            )
        )
        circuit.append(MeasureQFT(e).make_on(x=exponent, result_bits=result_bits))
        return _BoundRSAPhaseEstimation(wires, circuit.to_ir().instructions)
