# SPDX-License-Identifier: Apache-2.0
"""Bind the grid inverse-square-root Newton update to coherent registers."""

from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler.registers.fixed_point.inverse_sqrt_newton import (
    inverse_sqrt_newton_requirements,
    inverse_sqrt_newton_step_ir,
)
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR
from ...dynamic import DynamicCircuit


@dataclass(frozen=True)
class GridInverseSqrtNewton:
    """XOR a truncated Newton estimate into an arbitrary target word.

    The unsigned big-endian inputs represent a and x at the specified
    fractional widths. The update is x*(3-a*x*x)/2. Each populated input
    must satisfy a*x*x <= 3 and yield an unsigned result fitting target,
    at the estimate's fractional scale. Caller work starts and ends zero.
    Source, estimate and correction inputs are preserved. Reapplication
    restores a matching target.

    An optional correction adds delta to the factor 3. Its word is
    interpreted with source_fraction_bits + 2*estimate_fraction_bits
    fractional bits, as in the shared grid update.
    """

    x_sq: tuple[int, ...]
    poly: tuple[int, ...]
    target: tuple[int, ...]
    workspace: tuple[int, ...]
    source_fraction_bits: int
    estimate_fraction_bits: int
    correction: tuple[int, ...] = ()

    @staticmethod
    def requirements(
        source: int,
        estimate: int,
        *,
        source_fraction_bits: int,
        estimate_fraction_bits: int,
    ) -> tuple[int, int]:
        return inverse_sqrt_newton_requirements(
            source,
            estimate,
            source_fraction_bits=source_fraction_bits,
            estimate_fraction_bits=estimate_fraction_bits,
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError("Grid Newton update requires a DynamicCircuit")
        start = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        _, slots = self.requirements(
            len(self.x_sq),
            len(self.poly),
            source_fraction_bits=self.source_fraction_bits,
            estimate_fraction_bits=self.estimate_fraction_bits,
        )
        gates = inverse_sqrt_newton_step_ir(
            self.x_sq,
            self.poly,
            self.target,
            self.workspace,
            tuple(range(start, start + slots)),
            source_fraction_bits=self.source_fraction_bits,
            estimate_fraction_bits=self.estimate_fraction_bits,
            correction=self.correction,
        )
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz}
            ).validate()
        )
