# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked physical builder's scalar allocation and free stacks.
# https://github.com/quantumlib/Qualtran/blob/9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453/kickmix/src/kickmix/py/circuit/circuit_builder.pybind.h

"""Grow one physical ID space while reusing authoritative workspace leases."""

from typing import SupportsIndex, SupportsInt

from ...artifacts._operand_tags import MAX_NUM_QUBITS
from .workspace import RegisterWorkspace


def allocation_count(value: object) -> int:
    """Apply the source integer protocol before any allocation side effects."""
    if isinstance(value, int):
        result = int.__int__(value)
    elif isinstance(value, (SupportsInt, SupportsIndex)) and not isinstance(
        value, float
    ):
        result = int(value)
    else:
        raise TypeError("Physical allocation requires an integer count")
    if not 0 <= result < 1 << 64:
        raise TypeError("Physical allocation count must fit unsigned 64-bit storage")
    return result


class WireAllocation:
    """Keep permanent register IDs separate from releasable temporary storage."""

    def __init__(self) -> None:
        self._workspace = RegisterWorkspace(())
        self._next_wire = 0
        self._reserved_count = 0

    @property
    def allocated_count(self) -> int:
        return self._reserved_count + self._workspace.live_count

    def reserve(self, count: int) -> tuple[int, ...]:
        """Reserve new permanent IDs without consuming reusable temporary IDs."""
        count = allocation_count(count)
        self._check_growth(count)
        result = tuple(range(self._next_wire, self._next_wire + count))
        self._next_wire += count
        self._reserved_count += count
        return result

    def allocate(self, count: int) -> tuple[int, ...]:
        """Reuse the free stack before adopting new IDs into live workspace."""
        count = allocation_count(count)
        reused = min(count, self._workspace.available_count())
        self._check_growth(count - reused)
        previous = self._workspace.allocate(reused).wires if reused else ()
        fresh = tuple(range(self._next_wire, self._next_wire + count - reused))
        if fresh:
            self._workspace.adopt(fresh)
        self._next_wire += len(fresh)
        return previous + fresh

    def free(self, wire: int) -> None:
        """Release a single live temporary wire onto the LIFO free stack."""
        self._workspace.free_wires((wire,))

    def _check_growth(self, count: int) -> None:
        if count > MAX_NUM_QUBITS - self._next_wire:
            raise ValueError("Physical allocation exceeds the operand ID limit")
