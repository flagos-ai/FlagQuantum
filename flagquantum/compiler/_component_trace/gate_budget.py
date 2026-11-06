# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GateCounts.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/_bloq_counts.py

"""Convert declared gate categories to the locked resource budget conventions."""

from typing import Protocol

from .symbolic_math import SymbolicInt


class GateCategories(Protocol):
    """Read-only categories shared by concrete costs and symbolic formulas."""

    @property
    def t(self) -> SymbolicInt: ...

    @property
    def toffoli(self) -> SymbolicInt: ...

    @property
    def cswap(self) -> SymbolicInt: ...

    @property
    def and_bloq(self) -> SymbolicInt: ...

    @property
    def clifford(self) -> SymbolicInt: ...

    @property
    def rotation(self) -> SymbolicInt: ...

    @property
    def measurement(self) -> SymbolicInt: ...


def _ccz_units(gates: GateCategories) -> SymbolicInt:
    return gates.toffoli + gates.cswap + gates.and_bloq


def t_and_ccz_budget(
    gates: GateCategories, *, ts_per_rotation: int = 11
) -> dict[str, SymbolicInt]:
    """Keep Toffoli-like costs as CCZ units and charge synthesized rotations to T."""
    return {
        "n_t": gates.t + ts_per_rotation * gates.rotation,
        "n_ccz": _ccz_units(gates),
    }


def _is_zero(count: SymbolicInt) -> bool:
    if isinstance(count, int):
        return count == 0
    try:
        return count.equals(0) is True
    except ValueError:
        return False


def toffoli_only_budget(gates: GateCategories) -> SymbolicInt:
    """Count Toffoli-like operations when T and rotation counts are zero."""
    if not _is_zero(gates.t):
        raise ValueError("Gate categories contain T counts")
    if not _is_zero(gates.rotation):
        raise ValueError("Gate categories contain rotations")
    return _ccz_units(gates)


def legacy_gate_budget(
    gates: GateCategories,
    *,
    ts_per_toffoli: int = 4,
    ts_per_cswap: int = 7,
    ts_per_and: int = 4,
    cliffords_per_and: int = 9,
    cliffords_per_cswap: int = 10,
) -> dict[str, SymbolicInt]:
    """Fold gates into the source's historical T, Clifford, and rotation fields."""
    return {
        "t": gates.t
        + ts_per_toffoli * gates.toffoli
        + ts_per_cswap * gates.cswap
        + ts_per_and * gates.and_bloq,
        "rotations": gates.rotation,
        "clifford": gates.clifford
        + gates.measurement
        + cliffords_per_and * gates.and_bloq
        + cliffords_per_cswap * gates.cswap,
    }


def beverland_gate_budget(gates: GateCategories) -> dict[str, SymbolicInt]:
    """Return the source Beverland inputs with rotation count as the depth proxy."""
    return {
        "meas": gates.measurement,
        "R": gates.rotation,
        "T": gates.t,
        "Tof": _ccz_units(gates),
        "D_R": gates.rotation,
    }
