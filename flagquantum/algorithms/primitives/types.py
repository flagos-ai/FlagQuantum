"""Protocols shared by the algorithm primitives.

The protocols describe operators by the gates they append, so a caller can supply any object
that can emit its own gates. Nothing here defines circuit semantics, and no protocol requires
an object from an external framework.

``Predicate`` is how a marked-state search is described: a decidable property of a bit
string, read one register value at a time, so the callable is handed the bit string's
integer value and never a sequence of bits.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from ...circuit import Circuit

Predicate = Callable[[int], bool]


@runtime_checkable
class ControlledUnitary(Protocol):
    """An operator that can be applied plainly, under control, and as an integer power.

    Phase estimation needs the controlled powers of an operator, so the protocol requires
    the power form and not only the bare one.
    """

    @property
    def n_wires(self) -> int:
        """The number of wires the operator acts on."""
        ...

    def apply(self, circuit: Circuit, wires: Sequence[int]) -> None:
        """Append the bare operator to ``circuit`` on ``wires``."""
        ...

    def apply_controlled(
        self, circuit: Circuit, control: int, wires: Sequence[int]
    ) -> None:
        """Append the operator controlled on the wire ``control``."""
        ...

    def apply_power_controlled(
        self, circuit: Circuit, control: int, wires: Sequence[int], power: int
    ) -> None:
        """Append the ``power``-th power of the operator, controlled on ``control``."""
        ...


@runtime_checkable
class StatePreparationOperator(Protocol):
    """A state-preparation unitary paired with the subspace whose amplitude is estimated."""

    @property
    def n_wires(self) -> int:
        """The number of wires the operator acts on."""
        ...

    def prepare(self, circuit: Circuit, wires: Sequence[int]) -> None:
        """Append the state-preparation unitary to ``circuit`` on ``wires``."""
        ...

    def mark(self, circuit: Circuit, wires: Sequence[int]) -> None:
        """Append the phase marker for the good subspace."""
        ...
