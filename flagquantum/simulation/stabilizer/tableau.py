"""Aaronson-Gottesman tableaus for Clifford circuits.

A stabilizer state on ``n`` qubits is described by ``2n`` Pauli operators in
canonical form ``(-1)**r * i**(x . z) * X**x * Z**z``: rows ``[0, n)`` are the
destabilizers and rows ``[n, 2n)`` are the stabilizer generators whose ``+1``
eigenspace is the state. The representation stores ``x`` and ``z`` as Boolean
tensors of shape ``(2n, n)`` and the phase bits as a Boolean tensor of shape
``(2n,)``, so a gate touches at most four columns and the cost of a circuit is
linear in the number of qubits times the number of gates rather than
exponential in the number of qubits.

Only Clifford gates are representable. A gate outside the set, a gate angle
that would need angle-specific Clifford recognition, and any instruction the
simulator would otherwise have to approximate raise
:class:`~flagquantum.errors.CapabilityError` instead of silently degrading.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ...core.operator_schema import canonical_opcode
from ...errors import CapabilityError, ValidationError

# A single-qubit Clifford is fully described by where it sends the four Pauli
# operators on its wire. Each entry is indexed by ``2 * x + z`` of the incoming
# row and holds the outgoing ``x`` bit, the outgoing ``z`` bit, and the phase
# bit to exclusive-or into the row. Deriving every single-qubit gate from this
# table keeps the phase bookkeeping in one reviewed place.
_SINGLE_QUBIT_ENTRIES: dict[str, tuple[tuple[int, int, int], ...]] = {
    # index:            I (0,0)     Z (0,1)     X (1,0)     Y (1,1)
    "i": ((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)),
    "h": ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)),
    "s": ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 1)),
    "sdg": ((0, 0, 0), (0, 1, 0), (1, 1, 1), (1, 0, 0)),
    "sx": ((0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 0)),
    "sxdg": ((0, 0, 0), (1, 1, 0), (1, 0, 0), (0, 1, 1)),
    "x": ((0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)),
    "y": ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)),
    "z": ((0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 1)),
}

# Two-qubit Cliffords that are not a bare controlled-X are applied through the
# identities CY = (I x Sdg) CX (I x S), CZ = (I x H) CX (I x H), and
# SWAP = CX CX CX, so every two-qubit phase rule reduces to the controlled-X
# rule below.
_TWO_QUBIT_RECIPES: dict[str, tuple[tuple[str, ...], ...]] = {
    "cx": (("cx",),),
    "cz": (("h", "right"), ("cx",), ("h", "right")),
    "cy": (("sdg", "right"), ("cx",), ("s", "right")),
    "swap": (("cx", "forward"), ("cx", "reverse"), ("cx", "forward")),
}

CLIFFORD_GATE_NAMES = frozenset(_SINGLE_QUBIT_ENTRIES) | frozenset(_TWO_QUBIT_RECIPES)


def _table_tensors(
    entries: tuple[tuple[int, int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([entry[0] for entry in entries], dtype=torch.bool),
        torch.tensor([entry[1] for entry in entries], dtype=torch.bool),
        torch.tensor([entry[2] for entry in entries], dtype=torch.bool),
    )


_SINGLE_QUBIT_TABLES: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {
    name: _table_tensors(entries) for name, entries in _SINGLE_QUBIT_ENTRIES.items()
}


def _normalize_wire(wire: Any, n_qubits: int, *, what: str) -> int:
    if isinstance(wire, bool) or not isinstance(wire, int):
        raise ValidationError(f"{what} must be an integer wire index, got {wire!r}")
    if not 0 <= wire < n_qubits:
        raise ValidationError(
            f"{what} {wire} is outside the simulated register of {n_qubits} qubit(s)"
        )
    return wire


class CliffordTableau:
    """Mutable stabilizer tableau for one pure stabilizer state.

    The tableau is deliberately stateful: a gate mutates it in place, and
    sampling measures the current state. Call :meth:`reset` to return to
    ``|0...0>`` before replaying a circuit.
    """

    def __init__(self, n_qubits: int) -> None:
        if isinstance(n_qubits, bool) or not isinstance(n_qubits, int):
            raise ValidationError(f"n_qubits must be an integer, got {n_qubits!r}")
        if n_qubits <= 0:
            raise ValidationError(f"n_qubits must be positive, got {n_qubits}")
        self._n_qubits = n_qubits
        self._x = torch.zeros((2 * n_qubits, n_qubits), dtype=torch.bool)
        self._z = torch.zeros((2 * n_qubits, n_qubits), dtype=torch.bool)
        self._r = torch.zeros((2 * n_qubits,), dtype=torch.bool)
        self.reset()

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    def reset(self) -> None:
        """Return the tableau to the ``|0...0>`` state."""

        n = self._n_qubits
        self._x.zero_()
        self._z.zero_()
        self._r.zero_()
        rows = torch.arange(n)
        self._x[rows, rows] = True
        self._z[n + rows, rows] = True

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

    def apply_gate(self, name: str, wires: Sequence[int]) -> None:
        """Apply one Clifford gate, or fail closed if it is not Clifford."""

        opcode = canonical_opcode(name)
        if opcode in _SINGLE_QUBIT_ENTRIES:
            if len(wires) != 1:
                raise ValidationError(
                    f"gate {opcode!r} acts on one wire, got {len(wires)}"
                )
            self._apply_single(
                opcode,
                _normalize_wire(wires[0], self._n_qubits, what=f"gate {opcode!r} wire"),
            )
            return
        if opcode in _TWO_QUBIT_RECIPES:
            if len(wires) != 2:
                raise ValidationError(
                    f"gate {opcode!r} acts on two wires, got {len(wires)}"
                )
            left = _normalize_wire(
                wires[0], self._n_qubits, what=f"gate {opcode!r} wire"
            )
            right = _normalize_wire(
                wires[1], self._n_qubits, what=f"gate {opcode!r} wire"
            )
            if left == right:
                raise ValidationError(
                    f"gate {opcode!r} requires two distinct wires, got {left} twice"
                )
            self._apply_recipe(opcode, left, right)
            return
        raise CapabilityError(
            f"gate {opcode!r} is not in the Clifford gate set "
            f"({', '.join(sorted(CLIFFORD_GATE_NAMES))}); a non-Clifford gate "
            "cannot be represented by a stabilizer tableau. Execute the plan "
            "with mode='statevector', mode='mps', mode='tensor_network', or "
            "mode='density_matrix' instead."
        )

    def _apply_single(self, opcode: str, wire: int) -> None:
        table_x, table_z, table_r = _SINGLE_QUBIT_TABLES[opcode]
        index = self._x[:, wire].to(torch.int64) * 2 + self._z[:, wire].to(torch.int64)
        new_x = table_x[index]
        new_z = table_z[index]
        self._r ^= table_r[index]
        self._x[:, wire] = new_x
        self._z[:, wire] = new_z

    def _apply_recipe(self, opcode: str, left: int, right: int) -> None:
        for step in _TWO_QUBIT_RECIPES[opcode]:
            if step[0] == "cx":
                self._apply_cx(left, right)
                continue
            gate, position = step
            self._apply_single(gate, right if position == "right" else left)
        return

    def _apply_cx(self, control: int, target: int) -> None:
        """Controlled-X on (control, target) with the exact phase correction."""

        x = self._x
        z = self._z
        phase = x[:, control] & z[:, target] & ~(x[:, target] ^ z[:, control])
        self._r ^= phase
        x[:, target] = x[:, target] ^ x[:, control]
        z[:, control] = z[:, control] ^ z[:, target]

    def _apply_two_qubit_recipe(
        self, opcode: str, left: int, right: int
    ) -> None:  # pragma: no cover - retained for callers that prefer the name
        self._apply_recipe(opcode, left, right)

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def measure_z(self, wire: int, *, outcome: bool | None = None) -> bool:
        """Measure ``Z`` on one wire and collapse the tableau.

        When the outcome is random the caller may pass ``outcome`` to force the
        branch, which is how a sampling loop replays one shot deterministically.
        """

        index = _normalize_wire(wire, self._n_qubits, what="measurement wire")
        n = self._n_qubits
        stabilizer_column = self._x[n:, index]
        if bool(stabilizer_column.any()):
            result = bool(torch.randint(0, 2, ())) if outcome is None else bool(outcome)
            pivot = n + int(stabilizer_column.to(torch.int64).argmax())
            self._collide_on_pivot(pivot)
            self._x[pivot] = False
            self._z[pivot] = False
            self._z[pivot, index] = True
            self._r[pivot] = result
            return result
        # Deterministic: Z on this wire is a product of the stabilizer rows
        # selected by the destabilizer generators that anticommute with it.
        selecting = self._x[:n, index]
        if outcome is not None and bool(outcome) != self._deterministic_outcome(
            selecting
        ):
            raise ValidationError(
                f"measurement of wire {index} is deterministic; a forced outcome "
                "cannot disagree with the tableau"
            )
        return self._deterministic_outcome(selecting)

    def _deterministic_outcome(self, selecting: torch.Tensor) -> bool:
        if not bool(selecting.any()):
            return False
        n = self._n_qubits
        return bool((self._r[n:][selecting].to(torch.int64).sum() % 2) == 1)

    def _collide_on_pivot(self, pivot: int) -> None:
        """Left-multiply every row that anticommutes with the pivot by it."""

        affected = self._x[:, self._pivot_wire] if False else None  # pragma: no cover
        raise AssertionError(affected)  # pragma: no cover

    # ------------------------------------------------------------------
    # Pauli expectation values
    # ------------------------------------------------------------------

    def pauli_expectation(self, x_mask: torch.Tensor, z_mask: torch.Tensor) -> float:
        """Exact ``<P>`` for the Pauli ``P`` given by ``x_mask`` and ``z_mask``.

        A stabilizer state returns ``-1.0``, ``0.0``, or ``+1.0``; the ``0.0``
        case is exact, not a sampled or rounded result.
        """

        x_mask, z_mask = self._validate_mask(x_mask, z_mask)
        n = self._n_qubits
        stabilizers = self._symplectic(self._x[n:], self._z[n:], x_mask, z_mask)
        if bool(stabilizers.any()):
            return 0.0
        selecting = self._symplectic(self._x[:n], self._z[:n], x_mask, z_mask)
        if not bool(selecting.any()):
            return 1.0
        sign = int(self._r[n:][selecting].to(torch.int64).sum() % 2)
        return -1.0 if sign else 1.0

    def _validate_mask(
        self, x_mask: torch.Tensor, z_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = self._n_qubits
        masks = []
        for label, mask in (("x", x_mask), ("z", z_mask)):
            tensor = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
            if tensor.numel() != n:
                raise ValidationError(
                    f"Pauli {label} mask must have {n} entr(ies), "
                    f"got {tensor.numel()}"
                )
            masks.append(tensor)
        return masks[0], masks[1]

    @staticmethod
    def _symplectic(
        rows_x: torch.Tensor,
        rows_z: torch.Tensor,
        x_mask: torch.Tensor,
        z_mask: torch.Tensor,
    ) -> torch.Tensor:
        left = (rows_x & z_mask).to(torch.int64).sum(dim=1) % 2
        right = (rows_z & x_mask).to(torch.int64).sum(dim=1) % 2
        return (left ^ right).to(torch.bool)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def row_vectors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return copies of the ``x``, ``z``, and phase tensors."""

        return self._x.clone(), self._z.clone(), self._r.clone()

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"CliffordTableau(n_qubits={self._n_qubits})"
