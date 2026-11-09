# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Allocation and free contracts adapted from Qualtran v0.7.0.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping

"""Compile-time ownership of caller-provided clean and borrowed wires."""

from dataclasses import dataclass


@dataclass(frozen=True, eq=False)
class WorkspaceLease:
    """An ordered, whole-register binding with a distinct allocation lifetime."""

    wires: tuple[int, ...]
    dirty: bool


class RegisterWorkspace:
    """Allocate and release wires without emitting quantum instructions.

    The emitter proves that it returns clean wires to zero and borrowed
    wires to their original joint state before freeing them.
    """

    def __init__(
        self, clean: tuple[int, ...], *, borrowed: tuple[int, ...] = ()
    ) -> None:
        wires = (*clean, *borrowed)
        if any(type(wire) is not int or wire < 0 for wire in wires) or len(
            set(wires)
        ) != len(wires):
            raise ValueError("Workspace requires distinct nonnegative integer wires")
        self._available = {False: list(reversed(clean)), True: list(reversed(borrowed))}
        self._live: set[WorkspaceLease] = set()
        self._used: set[int] = set()

    def allocate(self, count: int, *, dirty: bool = False) -> WorkspaceLease:
        """Take the next wires in order from the selected pool."""
        if type(count) is not int or count < 0 or type(dirty) is not bool:
            raise ValueError(
                "Allocation requires a nonnegative width and a boolean dirty flag"
            )
        available = self._available[dirty]
        if count > len(available):
            kind = "borrowed" if dirty else "clean"
            raise ValueError(f"Insufficient {kind} workspace for {count} wires")
        lease = WorkspaceLease(tuple(available.pop() for _ in range(count)), dirty)
        self._live.add(lease)
        self._used.update(lease.wires)
        return lease

    def free(
        self, lease: WorkspaceLease, *, order: tuple[int, ...] | None = None
    ) -> None:
        """Release one live binding, optionally in its current virtual wire order."""
        if lease not in self._live:
            raise ValueError("Workspace release requires a live lease from this pool")
        wires = lease.wires if order is None else order
        if (
            len(wires) != len(lease.wires)
            or any(type(wire) is not int for wire in wires)
            or set(wires) != set(lease.wires)
        ):
            raise ValueError("Workspace release order must permute the leased wires")
        self._live.remove(lease)
        self._available[lease.dirty].extend(reversed(wires))

    @property
    def used_wires(self) -> tuple[int, ...]:
        """Return all touched workspace wires in physical wire order."""
        return tuple(sorted(self._used))
