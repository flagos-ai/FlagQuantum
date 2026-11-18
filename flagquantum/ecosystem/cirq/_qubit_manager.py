# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 cirq_interop/_interop_qubit_manager.py.

"""Reserve caller-owned Qids around a local Cirq decomposition."""

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from .conversion import _cirq


class InteropQubitManager:
    """Delegate Cirq work allocation while keeping external inputs reserved."""

    def __init__(self, base: Any = None) -> None:
        self._base = _cirq().SimpleQubitManager() if base is None else base
        self._reserved: set[Any] = set()

    def qalloc(self, n: int, dim: int = 2) -> list[Any]:
        return self._take(self._base.qalloc, n, dim)

    def qborrow(self, n: int, dim: int = 2) -> list[Any]:
        return self._take(self._base.qborrow, n, dim)

    def _take(
        self, allocate: Callable[[int, int], list[Any]], n: int, dim: int
    ) -> list[Any]:
        result: list[Any] = []
        skipped: list[Any] = []
        try:
            while len(result) < n:
                candidates = allocate(n - len(result), dim)
                if not candidates:
                    raise RuntimeError("Cirq base manager returned no requested qubits")
                for qubit in candidates:
                    if qubit in self._reserved:
                        skipped.append(qubit)
                    else:
                        result.append(qubit)
        except Exception:
            if skipped or result:
                self._base.qfree([*skipped, *result])
            raise
        if skipped:
            self._base.qfree(skipped)
        return result

    def manage_qubits(self, qubits: Iterable[Any]) -> None:
        self._reserved.update(qubits)

    @contextmanager
    def reserve_qubits(self, qubits: Iterable[Any]) -> Iterator[None]:
        """Reserve a decomposition frame's inputs until all descendants finish."""
        previous = self._reserved.copy()
        self.manage_qubits(qubits)
        try:
            yield
        finally:
            self._reserved = previous

    def qfree(self, qubits: Iterable[Any]) -> None:
        released = tuple(qubits)
        external = self._reserved.intersection(released)
        internal = [qubit for qubit in released if qubit not in external]
        if internal:
            self._base.qfree(internal)
        self._reserved.difference_update(external)
