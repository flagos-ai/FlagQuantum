# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Lane reservations adapted from Qualtran v0.7.0 LineManager.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/musical_score.py

"""Display lane allocation, independent of graph and numerical execution."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from typing import Literal, TypeAlias

from ..core.ir.program.registers import Register

LaneLocation: TypeAlias = int | Literal["input", "output"]
ReleasePredicate: TypeAlias = Callable[[LaneLocation, Register], bool]


@dataclass(frozen=True, order=True)
class LaneSegment:
    """A geometric lane interval visible during allocation and release.

    An unfinished interval has no end. Register identities and word widths
    remain owned by the layout's completed RegisterLine records.
    """

    y: int
    seq_x_start: int
    seq_x_end: int | None = None
    flavor: Literal["HLineFlavor.QUANTUM", "HLineFlavor.CLASSICAL"] = (
        "HLineFlavor.QUANTUM"
    )


class RegisterLaneManager:
    """Own a score's lane heap, reservations and geometric interval lifecycle.

    Locations are original call ordinals or the input/output boundary. Hooks
    receive native Core registers and row-major element indices. Subclasses can
    reserve lanes from ``maybe_reserve`` after each allocation or release.
    Capacity grows by default. An explicit finite capacity matches the source
    heap within that capacity. Create one manager per independent layout.
    The layout sets seq_x before allocation and release. Hooks may inspect the
    immutable open/closed intervals in hlines without modifying that collection.
    """

    def __init__(self, max_n_lines: int | None = None) -> None:
        if max_n_lines is not None and (
            type(max_n_lines) is not int or max_n_lines < 0
        ):
            raise ValueError("Lane capacity must be a nonnegative integer or None")
        self.available = list(range(max_n_lines or 0))
        self._capacity = max_n_lines
        self._next_lane = max_n_lines or 0
        self._reserved: list[tuple[tuple[int, ...], ReleasePredicate]] = []
        self.seq_x = 0
        self.hlines: set[LaneSegment] = set()

    def new_y(self) -> int:
        """Take the lowest free lane, growing an unbounded pool when needed."""
        if self.available:
            return heappop(self.available)
        if self._capacity is not None:
            raise IndexError("No display lanes available")
        lane = self._next_lane
        self._next_lane += 1
        return lane

    def new_y_for(
        self,
        location: LaneLocation,
        register: Register,
        index: tuple[int, ...] | None = None,
    ) -> int:
        """Allocate a lane with the source register-element context.

        Scalar allocation receives None and array allocation receives its
        row-major index. Overrides own selection from the available heap.
        The default delegates to new_y, retaining existing allocation policies.
        Reservations use new_y directly because they have no element context.
        """
        return self.new_y()

    def reserve_n(self, n: int, until: ReleasePredicate) -> None:
        """Hold the next n lanes until a later allocation satisfies until.

        Invalid requests and finite-capacity exhaustion leave the pool intact.
        """
        if type(n) is not int or n < 0:
            raise ValueError("Reservation size must be a nonnegative integer")
        if not callable(until):
            raise TypeError("Reservation release predicate must be callable")
        if self._capacity is not None and n > len(self.available):
            raise IndexError("Insufficient display lanes for reservation")
        self._reserved.append((tuple(self.new_y() for _ in range(n)), until))

    def unreserve(self, location: LaneLocation, register: Register) -> None:
        """Evaluate reservations once before a register's elements allocate."""
        available = self.available.copy()
        kept = []
        try:
            for lanes, until in self._reserved:
                if until(location, register):
                    for lane in lanes:
                        heappush(self.available, lane)
                else:
                    kept.append((lanes, until))
        except Exception:
            # Callback failures must not return the same reserved lane twice on retry.
            self.available[:] = available
            raise
        self._reserved = kept

    def maybe_reserve(
        self, location: LaneLocation, register: Register, index: tuple[int, ...]
    ) -> None:
        """Override to reserve lanes after an element allocates or releases."""

    def new(self, location: LaneLocation, register: Register) -> tuple[int, ...]:
        """Open each element at seq_x before invoking its allocation hook."""
        self.unreserve(location, register)
        classical = register.dtype.num_qubits == 0 and (
            register.dtype.num_cbits != 0
            or ("storage", "classical") in register.dtype.parameters
        )
        lanes = []
        for index in register.all_idxs():
            lane = self.new_y_for(location, register, index or None)
            lanes.append(lane)
            self.hlines.add(
                LaneSegment(
                    lane,
                    self.seq_x,
                    flavor=(
                        "HLineFlavor.CLASSICAL" if classical else "HLineFlavor.QUANTUM"
                    ),
                )
            )
            self.maybe_reserve(location, register, index)
        return tuple(lanes)

    def _partial_hline(self, y: int) -> LaneSegment:
        (line,) = (h for h in self.hlines if h.y == y and h.seq_x_end is None)
        return line

    def finish_hline(self, y: int, seq_x_end: int) -> None:
        """Replace the unique unfinished interval for y with its closed value.

        Missing or ambiguous intervals raise ValueError without changing lines.
        """
        partial = self._partial_hline(y)
        self.hlines.remove(partial)
        self.hlines.add(replace(partial, seq_x_end=seq_x_end))

    def free(
        self, location: LaneLocation, register: Register, lanes: tuple[int, ...]
    ) -> None:
        """Return lanes, close their intervals at seq_x, then invoke each hook.

        Invalid element counts, duplicate lanes and missing intervals are
        refused before releasing any lane. Hook failures propagate.
        """
        elements = tuple(zip(register.all_idxs(), lanes, strict=True))
        if len(set(lanes)) != len(lanes):
            raise ValueError("Released display lanes must be unique")
        for _, lane in elements:
            self._partial_hline(lane)
        for index, lane in elements:
            heappush(self.available, lane)
            self.finish_hline(lane, self.seq_x)
            self.maybe_reserve(location, register, index)
