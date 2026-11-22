# SPDX-License-Identifier: Apache-2.0
# Adapted from Kickmix operation_statistics.pybind.cc and Sim operation counters
# at 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Accumulate raw physical instruction and rotation counts by packed lane."""

from collections import defaultdict

from .physical_basis_shots import BasisRecord

_MASK = (1 << 64) - 1


class PhysicalOperationCounters:
    """Own counters independently of the caller's packed simulation state."""

    def __init__(self, shots: int, *, enabled: bool = True) -> None:
        if type(shots) is not int or shots < 0:
            raise ValueError("Counter shot count must be a nonnegative integer")
        self._shots = shots
        self._enabled = enabled
        self._operations: defaultdict[str, list[int]] = defaultdict(lambda: [0] * shots)
        self._levels: defaultdict[int, list[int]] = defaultdict(lambda: [0] * shots)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def clear(self) -> None:
        """Reset counts while leaving the separately owned state untouched."""
        self._operations.clear()
        self._levels.clear()

    def observe(self, record: BasisRecord, active: int) -> None:
        """Count classically active lanes before quantum controls are applied."""
        if not self._enabled:
            return
        if active < 0 or active.bit_length() > self._shots:
            raise ValueError("Active counter mask exceeds the configured shot count")
        name = record.opcode
        if record.condition_bit is not None and name != "PUSH_CONDITION":
            name += "_IF"
        counts = self._operations[name]
        levels: list[int] | None = None
        if record.angle is not None:
            units = record.angle.units
            level = 0 if not units else 129 - (units & -units).bit_length()
            levels = self._levels[level]
        while active:
            low = active & -active
            lane = low.bit_length() - 1
            counts[lane] = (counts[lane] + 1) & _MASK
            if levels is not None:
                levels[lane] = (levels[lane] + 1) & _MASK
            active ^= low

    def snapshot(
        self, shot_index: int | None = None
    ) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[int, int], ...]]:
        """Return independent raw-name and rotation-level count tuples."""
        if not self._enabled:
            raise ValueError("Operation counting is not enabled")
        if shot_index is not None and not 0 <= shot_index < self._shots:
            raise IndexError("Need 0 <= shot_index < counter shot count")

        def total(lanes: list[int]) -> int:
            return sum(lanes) & _MASK if shot_index is None else lanes[shot_index]

        return (
            tuple((name, total(lanes)) for name, lanes in self._operations.items()),
            tuple((level, total(lanes)) for level, lanes in self._levels.items()),
        )
