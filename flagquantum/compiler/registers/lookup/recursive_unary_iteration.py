# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Kickmix UnaryIterationCursor and gen_unary_iteration.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/gen/iteration/gen_unary_iteration.cc

"""Source-ordered address traversal with explicit measured cleanup."""

from collections.abc import Callable, Sequence

from ...artifacts.reversible_phase import ReversiblePhaseOperation
from ..finite_field.polynomial import _validate_registers
from ..layout.workspace import RegisterWorkspace


class UnaryAddressCursor:
    """Compile visits over a big-endian address and caller-owned clean work.

    Construction reserves work but emits nothing. Concatenate the operations
    returned by move_to and close. The caller must close a live cursor before
    reusing its workspace; there is no destructor-based gate emission.
    """

    def __init__(
        self,
        address: tuple[int, ...],
        workspace: tuple[int, ...],
        *,
        control: int | None = None,
        measurement_bit: int = 0,
    ) -> None:
        _validate_registers(address, workspace, () if control is None else (control,))
        if len(address) > 63:
            raise ValueError("Unary addresses support at most 63 bits")
        if type(measurement_bit) is not int or measurement_bit < 0:
            raise ValueError("Unary erasure requires a nonnegative result slot")
        self.address = address[::-1]
        pool = RegisterWorkspace(workspace)
        self.workspace = pool.allocate(max(1, len(address))).wires
        self.control = control
        self.measurement_bit = measurement_bit
        self.address_space_size = 1 << len(address)
        self.current = self.address_space_size

    @property
    def match_qubit(self) -> int:
        """Return the wire marking the current live address and control."""
        return self.workspace[0]

    def _flip(
        self, target: int, parent: int | None, address: int | None = None
    ) -> ReversiblePhaseOperation:
        controls = tuple(q for q in (parent, address) if q is not None)
        return ReversiblePhaseOperation(
            ("x", "cx", "ccx")[len(controls)], (*controls, target)
        )

    def _level(
        self, level: int, value: int, *, erase: bool
    ) -> tuple[ReversiblePhaseOperation, ...]:
        address = self.address[level] if self.address else None
        parent = (
            self.control
            if level == len(self.address) - 1 or not self.address
            else self.workspace[level + 1]
        )
        work = self.workspace[level]
        negative = address is not None and not ((value >> level) & 1)
        prefix = (
            (ReversiblePhaseOperation("x", (address,)),)
            if negative and address is not None
            else ()
        )
        if erase:
            controls = tuple(q for q in (parent, address) if q is not None)
            middle = (
                ReversiblePhaseOperation(
                    "hmr", (work,), result_bit=self.measurement_bit
                ),
                ReversiblePhaseOperation(
                    ("neg", "z", "cz")[len(controls)],
                    controls,
                    condition_bit=self.measurement_bit,
                ),
            )
        else:
            middle = (
                ReversiblePhaseOperation("r", (work,)),
                self._flip(work, parent, address),
            )
        return (*prefix, *middle, *prefix)

    def _levels(
        self, value: int, count: int, *, erase: bool
    ) -> tuple[ReversiblePhaseOperation, ...]:
        levels = range(count) if erase else reversed(range(count))
        return tuple(
            op for level in levels for op in self._level(level, value, erase=erase)
        )

    def move_to(self, value: int) -> tuple[ReversiblePhaseOperation, ...]:
        """Visit a uint64 value, preserving source jump and parking behavior."""
        if type(value) is not int or not 0 <= value < 1 << 64:
            raise ValueError("Unary visits require unsigned 64-bit integers")
        was_live = self.current < self.address_space_size
        now_live = value < self.address_space_size
        operations: tuple[ReversiblePhaseOperation, ...]
        if value == self.current or not (was_live or now_live):
            operations = ()
        elif not was_live:
            operations = self._levels(value, max(1, len(self.address)), erase=False)
        elif not now_live:
            operations = self._levels(
                self.current, max(1, len(self.address)), erase=True
            )
        else:
            level = (self.current ^ value).bit_length() - 1
            parent = (
                self.control
                if level == len(self.address) - 1
                else self.workspace[level + 1]
            )
            operations = (
                *self._levels(self.current, level, erase=True),
                self._flip(self.workspace[level], parent),
                *self._levels(value, level, erase=False),
            )
        self.current = value
        return operations

    def close(self) -> tuple[ReversiblePhaseOperation, ...]:
        """Clear all owned work and park; repeated close emits nothing."""
        return self.move_to(self.address_space_size)


def unary_iteration_operations(
    address: tuple[int, ...],
    workspace: tuple[int, ...],
    values: Sequence[int],
    body: Callable[[int, int], tuple[ReversiblePhaseOperation, ...]],
    *,
    control: int | None = None,
    measurement_bit: int = 0,
) -> tuple[ReversiblePhaseOperation, ...]:
    """Visit values in order, invoke the body even when parked, then close."""
    if not values:
        return ()
    cursor = UnaryAddressCursor(
        address, workspace, control=control, measurement_bit=measurement_bit
    )
    operations: list[ReversiblePhaseOperation] = []
    for value in values:
        operations.extend(cursor.move_to(value))
        operations.extend(body(value, cursor.match_qubit))
    operations.extend(cursor.close())
    return tuple(operations)
