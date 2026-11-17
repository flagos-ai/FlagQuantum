# SPDX-License-Identifier: Apache-2.0
"""Coherent coefficient binding for the grid inverse-square-root polynomial."""

from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler.registers.fixed_point.register_polynomial import (
    register_polynomial_ir,
    register_polynomial_requirements,
)
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR
from ...dynamic import DynamicCircuit


@dataclass(frozen=True)
class GridInverseSqrtPolynomial:
    """Bind argument, four coefficient words and an arbitrary XOR output.

    Coefficients can carry quantum data. The alternating polynomial uses
    explicit fractional precision and wraps each intermediate at coefficient
    width. The caller owns initially clean work. Ports are big-endian.
    """

    x_sq: tuple[int, ...]
    in_coeff: tuple[tuple[int, ...], ...]
    out: tuple[int, ...]
    workspace: tuple[int, ...]
    argument_fraction_bits: int

    @staticmethod
    def requirements(argument: int, coefficient: int) -> tuple[int, int]:
        return register_polynomial_requirements(argument, coefficient)

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError("Grid polynomial requires a DynamicCircuit")
        start = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        width = len(self.in_coeff[0]) if self.in_coeff else 0
        _, slots = self.requirements(len(self.x_sq), width)
        gates = register_polynomial_ir(
            self.x_sq,
            self.in_coeff,
            self.out,
            self.workspace,
            tuple(range(start, start + slots)),
            argument_fraction_bits=self.argument_fraction_bits,
        )
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz}
            ).validate()
        )
