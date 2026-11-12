# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Lane reservations adapted from Qualtran v0.7.0 LineManager.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/musical_score.py

"""Display lane allocation, independent of graph and numerical execution."""

from collections.abc import Callable
from heapq import heappop, heappush
from typing import Literal, TypeAlias

from ..core.ir.program.registers import Register

LaneLocation: TypeAlias = int | Literal["input", "output"]
ReleasePredicate: TypeAlias = Callable[[LaneLocation, Register], bool]


class RegisterLaneManager:
    """Own a score's minimum-lane heap and temporary reservations.

    Locations are original call ordinals or the input/output boundary. Hooks
    receive native Core registers and row-major element indices. Subclasses can
    reserve lanes from ``maybe_reserve`` after each allocation or release.
    Capacity grows by default. An explicit finite capacity matches the source
    heap within that capacity. Create one manager per independent layout.
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
        """Allocate all elements, checking release predicates once per register."""
        self.unreserve(location, register)
        lanes = []
        for index in register.all_idxs():
            lanes.append(self.new_y_for(location, register, index or None))
            self.maybe_reserve(location, register, index)
        return tuple(lanes)

    def free(
        self, location: LaneLocation, register: Register, lanes: tuple[int, ...]
    ) -> None:
        """Release elements in row-major order without evaluating predicates."""
        for index, lane in zip(register.all_idxs(), lanes, strict=True):
            heappush(self.available, lane)
            self.maybe_reserve(location, register, index)
