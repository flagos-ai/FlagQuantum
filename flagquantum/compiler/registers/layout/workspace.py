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
        self._live: dict[WorkspaceLease, tuple[int, ...]] = {}
        self._used: set[int] = set()

    def adopt(self, wires: tuple[int, ...], *, dirty: bool = False) -> WorkspaceLease:
        """Register an existing caller-owned lifetime without allocating wires."""
        if (
            not isinstance(wires, tuple)
            or any(type(wire) is not int or wire < 0 for wire in wires)
            or len(set(wires)) != len(wires)
            or type(dirty) is not bool
        ):
            raise ValueError(
                "Adoption requires distinct nonnegative integer wire tuples "
                "and a boolean dirty flag"
            )
        occupied = {wire for remaining in self._live.values() for wire in remaining}
        occupied.update(self._available[False])
        occupied.update(self._available[True])
        if occupied.intersection(wires):
            raise ValueError("Adopted wires must not overlap available or live wires")
        lease = WorkspaceLease(wires, dirty)
        self._live[lease] = wires
        self._used.update(wires)
        return lease

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
        self._live[lease] = lease.wires
        self._used.update(lease.wires)
        return lease

    def free(
        self, lease: WorkspaceLease, *, order: tuple[int, ...] | None = None
    ) -> None:
        """Release the live remainder, optionally in its current virtual wire order."""
        if lease not in self._live:
            raise ValueError("Workspace release requires a live lease from this pool")
        remaining = self._live[lease]
        wires = remaining if order is None else order
        if (
            len(wires) != len(remaining)
            or any(type(wire) is not int for wire in wires)
            or set(wires) != set(remaining)
        ):
            raise ValueError(
                "Workspace release order must permute the leased wires still live"
            )
        del self._live[lease]
        self._available[lease.dirty].extend(reversed(wires))

    def free_wires(self, wires: tuple[int, ...]) -> None:
        """Release ordered live wires, including parts of distinct leases.

        Every wire returns to its allocation pool. Validate the complete request
        before releasing any prefix. Empty words retain their separate lease
        identity and must be released with ``free(lease)``.
        """
        if any(type(wire) is not int for wire in wires) or len(set(wires)) != len(
            wires
        ):
            raise ValueError(
                "Partial workspace release requires distinct integer wires"
            )
        owners = {
            wire: lease for lease, remaining in self._live.items() for wire in remaining
        }
        if any(wire not in owners for wire in wires):
            raise ValueError(
                "Partial workspace release requires live wires from this pool"
            )
        released = set(wires)
        for lease in {owners[wire] for wire in wires}:
            remaining = tuple(
                wire for wire in self._live[lease] if wire not in released
            )
            if remaining:
                self._live[lease] = remaining
            else:
                del self._live[lease]
        for dirty in (False, True):
            ordered = tuple(wire for wire in wires if owners[wire].dirty is dirty)
            self._available[dirty].extend(reversed(ordered))

    @property
    def used_wires(self) -> tuple[int, ...]:
        """Return all touched workspace wires in physical wire order."""
        return tuple(sorted(self._used))
