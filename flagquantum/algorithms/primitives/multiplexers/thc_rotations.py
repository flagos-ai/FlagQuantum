"""Explicit address-selected Givens networks for THC basis rotations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite, tau

from ....circuit import Circuit
from ....compiler._arithmetic_kernels import ripple_add_modular
from ....compiler.unary_iteration import append_unary_iteration
from ....core.ir import CircuitIR, Instruction
from ....errors import ValidationError


@dataclass(frozen=True)
class THCRotations:
    """Apply one Givens network for each big-endian selection address.

    A rotation is (left, right, angle) with adjacent target offsets and an
    angle in radians. It maps |01> to cos(angle)|01> + sin(angle)|10>.
    Unlisted addresses act as identity. Angles are compiled as constants.
    """

    selection: tuple[int, ...]
    target: tuple[int, ...]
    networks: tuple[tuple[tuple[int, int, float], ...], ...]
    is_adjoint: bool = False

    @classmethod
    def make_on(
        cls,
        *,
        selection: Sequence[int],
        target: Sequence[int],
        networks: Sequence[Sequence[tuple[int, int, float]]],
    ) -> THCRotations:
        """Bind an ordered network table to the circuit's registers."""
        return cls(tuple(selection), tuple(target), tuple(map(tuple, networks)))

    def adjoint(self) -> THCRotations:
        """Reverse the Givens network selected by each address."""
        return replace(self, is_adjoint=not self.is_adjoint)

    def append_to(self, circuit: Circuit) -> Circuit:
        """Append exact gates using the existing controlled rotation synthesis."""
        qubits = (*self.selection, *self.target)
        if any(type(q) is not int or not 0 <= q < circuit.n_qubits for q in qubits):
            raise ValidationError("THC rotation qubits must be valid circuit indices.")
        if len(set(qubits)) != len(qubits):
            raise ValidationError("THC rotation registers must use disjoint qubits.")
        width = len(self.selection)
        if not self.target or not 0 < len(self.networks) <= 1 << width:
            raise ValidationError("THC rotation table must fit its selection register.")

        # Compose only after synthesis succeeds, so invalid inputs cannot
        # leave an incomplete rotation on the caller's circuit.
        fragment = Circuit(len(qubits), bsz=circuit.bsz, dtype=circuit.dtype)
        controls = tuple(range(1, width + 2))
        for address, network in enumerate(self.networks):
            if not network:
                continue
            flips = [q for q in range(width) if not (address >> (width - q - 1)) & 1]
            for q in flips:
                fragment.x(q)
            rotations = reversed(network) if self.is_adjoint else iter(network)
            for left, right, angle in rotations:
                self._validate_rotation(left, right, angle)
                theta = -2 * angle if self.is_adjoint else 2 * angle
                rotation = Circuit(1, bsz=circuit.bsz, dtype=circuit.dtype).ry(
                    0, theta=theta
                )
                controlled = rotation.control(len(controls), controls)
                fragment.cx(width + left, width + right)
                fragment.compose(
                    controlled,
                    qubits=(width + left, width + right, *range(width)),
                )
                fragment.cx(width + left, width + right)
            for q in flips:
                fragment.x(q)
        return circuit.compose(fragment, qubits=qubits)

    def _append_qrom(
        self,
        circuit: Circuit,
        angle_data: tuple[int, ...],
        gradient: tuple[int, ...],
        carry: int,
        scratch: tuple[int, ...],
    ) -> None:
        width = len(self.selection)
        modulus = 1 << len(angle_data)
        tables: dict[tuple[int, int], list[int]] = {}
        for address, network in enumerate(self.networks):
            for slot, (left, right, angle) in enumerate(network):
                self._validate_rotation(left, right, angle)
                # Reduce before scaling so any finite input angle is supported.
                word = round((angle % tau) / tau * modulus) % modulus
                if word:
                    table = tables.setdefault((slot, left), [0] * len(self.networks))
                    table[address] = (-word if self.is_adjoint else word) % modulus

        adder = ripple_add_modular(angle_data, gradient, carry)
        for slot in sorted(tables, reverse=self.is_adjoint):
            _, left = slot
            target, parity = self.target[left : left + 2]
            operands = (parity, *self.selection, *angle_data, *scratch[:width])
            qrom = Circuit.from_ir(
                _angle_lookup(width, len(angle_data), tables[slot], len(operands)),
                dtype=circuit.dtype,
            )
            circuit.cx(target, parity).compose(qrom, qubits=operands)
            circuit.sdg(target).h(target)
            # Complementing both sides of the modular add implements subtraction.
            for wire in gradient:
                circuit.cx(target, wire)
            for gate in adder:
                circuit.gate(gate.name, gate.wires)
            for wire in gradient:
                circuit.cx(target, wire)
            circuit.h(target).s(target)
            # XOR lookup is its own inverse and the address and parity are intact.
            circuit.compose(qrom, qubits=operands).cx(target, parity)

    def _validate_rotation(self, left: int, right: int, angle: float) -> None:
        if (
            type(left) is not int
            or type(right) is not int
            or not 0 <= left < right == left + 1 < len(self.target)
            or not isfinite(angle)
        ):
            raise ValidationError(
                "THC needs adjacent orbital offsets and finite angles."
            )


def _angle_lookup(
    width: int, angle_bits: int, table: list[int], wires: int
) -> CircuitIR:
    instructions: list[Instruction] = []

    def load(address: int, predicate: int | None) -> None:
        assert predicate is not None  # The orbital parity bit is always a control.
        for bit in range(angle_bits):
            if table[address] & (1 << (angle_bits - 1 - bit)):
                instructions.append(Instruction("cx", (predicate, width + 1 + bit)))

    append_unary_iteration(
        instructions,
        operand_count=width + 1 + angle_bits,
        width=width,
        entries=len(table),
        control_value=1,
        apply_branch=load,
        coherent=True,
    )
    return CircuitIR(wires, tuple(instructions))
