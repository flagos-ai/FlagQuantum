"""Explicit address-selected Givens networks for THC basis rotations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite

from ....circuit import Circuit
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
                if (
                    type(left) is not int
                    or type(right) is not int
                    or not 0 <= left < right == left + 1 < len(self.target)
                    or not isfinite(angle)
                ):
                    raise ValidationError(
                        "THC needs adjacent orbital offsets and finite angles."
                    )
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
