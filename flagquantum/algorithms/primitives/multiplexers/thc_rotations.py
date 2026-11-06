"""Explicit address-selected Givens networks for THC basis rotations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite, tau

from ....circuit import Circuit
from ....compiler.registers.arithmetic import (
    controlled_add_or_subtract_ir,
    measurement_add_or_subtract_ir,
)
from ....compiler.registers.qrom import controlled_qrom_ir
from ....core.ir import CircuitIR
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
        carries: tuple[int, ...],
        scratch: tuple[int, ...],
        measurement_bits: tuple[int, ...],
        junk: tuple[int, ...],
    ) -> None:
        width = len(self.selection)
        modulus = 1 << len(angle_data)
        relative_phase_swap = bool(junk and measurement_bits)
        traversal_bits: tuple[int, ...] = ()
        if measurement_bits:
            start = measurement_bits[-1] + 1 + (len(carries) if len(carries) > 1 else 0)
            traversal_bits = tuple(range(start, start + width - int(bool(junk))))
        tables: dict[tuple[int, int], list[int]] = {}
        for address, network in enumerate(self.networks):
            for slot, (left, right, angle) in enumerate(network):
                self._validate_rotation(left, right, angle)
                # Reduce before scaling so any finite input angle is supported.
                word = round((angle % tau) / tau * modulus) % modulus
                if word:
                    table = tables.setdefault((slot, left), [0] * len(self.networks))
                    table[address] = (-word if self.is_adjoint else word) % modulus

        for table_key in sorted(tables, reverse=self.is_adjoint):
            _, left = table_key
            target, parity = self.target[left : left + 2]
            operands = (parity, *self.selection, *angle_data, *junk, *scratch[:width])
            lookup_words = tables[table_key]
            if traversal_bits:
                # Match upstream traversal over the full address register while
                # preserving the zero word on FlagQuantum's padded addresses.
                lookup_words = [
                    *lookup_words,
                    *([0] * ((1 << width) - len(lookup_words))),
                ]
            qrom = Circuit.from_ir(
                controlled_qrom_ir(
                    width,
                    len(angle_data),
                    lookup_words,
                    blocked=bool(junk),
                    traversal_bits=traversal_bits,
                    relative_phase_swap=relative_phase_swap,
                ),
                dtype=circuit.dtype,
                bsz=circuit.bsz,
            )
            circuit.cx(target, parity).compose(qrom, qubits=operands)
            circuit.sdg(target).h(target)
            if len(carries) > 1:
                first_bit = measurement_bits[-1] + 1
                bits = tuple(range(first_bit, first_bit + len(carries)))
                adder = measurement_add_or_subtract_ir(
                    target, angle_data, gradient, carries, bits
                )
                circuit.compose(
                    Circuit.from_ir(
                        CircuitIR(circuit.n_qubits, adder),
                        dtype=circuit.dtype,
                        bsz=circuit.bsz,
                    )
                )
            else:
                for gate in controlled_add_or_subtract_ir(
                    target, angle_data, gradient, carries[0]
                ):
                    circuit.gate(gate.name, gate.wires)
            circuit.h(target).s(target)
            if measurement_bits or junk:
                qrom = Circuit.from_ir(
                    controlled_qrom_ir(
                        width,
                        len(angle_data),
                        tables[table_key],
                        measurement_bits=measurement_bits,
                        blocked=bool(junk),
                        inverse=True,
                        relative_phase_swap=relative_phase_swap,
                    ),
                    dtype=circuit.dtype,
                    bsz=circuit.bsz,
                )
            # The address and orbital parity still select the same word.
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
