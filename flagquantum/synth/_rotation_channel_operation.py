"""Bind the executable rotation channel forms to native circuit operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..runtime.dynamic.circuit import DynamicCircuit
from ._clifford_t_operation import CliffordTOperation
from ._rotation_channels import ProjectiveChannel, UnitaryChannel

if TYPE_CHECKING:
    from ..circuit import Circuit


@dataclass(frozen=True)
class UnitaryChannelOperation:
    """Append a non-twirled synthesized channel on an existing wire."""

    channel: UnitaryChannel
    target: int
    fmt: str = "xz"

    def append_to(self, circuit: Circuit) -> Circuit:
        if self.channel.twirl:
            raise ValueError("Unitary channel execution requires twirl=False")
        return CliffordTOperation(
            self.channel.to_matrix(), self.target, self.fmt
        ).append_to(circuit)


@dataclass(frozen=True)
class ProjectiveChannelOperation:
    """Append the locked executable path with an outcome bit and clean ancilla.

    The caller supplies a zero ancilla. The measured bit retains the outcome
    after the conditional ancilla reset. Both channel factors must be unitary
    and non-twirled, matching the source circuit export.
    """

    channel: ProjectiveChannel
    target: int
    ancilla: int
    classical_bit: int
    fmt: str = "xz"

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(circuit, DynamicCircuit):
            raise TypeError("Projective rotation requires a DynamicCircuit")
        if self.target == self.ancilla or not all(
            0 <= wire < circuit.n_qubits for wire in (self.target, self.ancilla)
        ):
            raise ValueError("Projective rotation requires two distinct circuit wires")
        if self.classical_bit < 0:
            raise ValueError("Projective rotation requires a nonnegative outcome bit")
        correction = self.channel.correction
        if not isinstance(correction, UnitaryChannel):
            raise ValueError("Projective execution requires a unitary correction")
        if self.channel.rotation.twirl or correction.twirl:
            raise ValueError("Projective execution requires non-twirled channels")
        rotation_gates = CliffordTOperation(
            self.channel.rotation.to_matrix(), self.target, self.fmt
        ).instructions()
        correction_gates = CliffordTOperation(
            correction.to_matrix(), self.target, self.fmt
        ).instructions()

        circuit.cx(self.target, self.ancilla)
        for instruction in rotation_gates:
            circuit.gate(instruction.name, instruction.wires, params=instruction.params)
        circuit.cx(self.target, self.ancilla)
        circuit.measure(self.ancilla, classical_bit=self.classical_bit)
        for instruction in correction_gates:
            circuit.conditional(
                instruction.name,
                instruction.wires,
                classical_bit=self.classical_bit,
                params=instruction.params,
            )
        circuit.conditional("x", self.ancilla, classical_bit=self.classical_bit)
        return circuit
