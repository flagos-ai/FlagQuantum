"""Checks code definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..pauli import Pauli
from ._compat import _retain_module


def ancilla_bands(
    checks: tuple[CodeCheck, ...] | list[CodeCheck],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split a code's check ancillas into the X band and the Z band.

    A check's ancilla measures one basis, which its stabilizer's type fixes, so
    the two bands are a function of the checks and are derived here rather than
    declared beside them. That is deliberate: a record that stated its bands as
    well as its checks could state two answers, and the drift would be silent
    because every consumer reads one of them. A record that derives them cannot.

    An ancilla that measures neither basis -- a flag or an idle ancilla -- is in
    neither band, so the two bands need not cover a record's declared ancillas.
    Both bands are returned in ascending qubit order, which is the order a
    detector band and a schedule are read in.

    Args:
        checks: The checks whose ancillas are being split.

    Returns:
        The ``(x_qubits, z_qubits)`` pair, each sorted and without duplicates.
    """

    x_qubits = tuple(
        sorted({check.ancilla_qubit for check in checks if check.stabilizer.x_qubits})
    )
    z_qubits = tuple(
        sorted({check.ancilla_qubit for check in checks if check.stabilizer.z_qubits})
    )
    overlap = set(x_qubits) & set(z_qubits)
    if overlap:
        raise ValueError(
            "ancillas "
            + ", ".join(str(qubit) for qubit in sorted(overlap))
            + " measure a check of each basis, so they belong to neither band: one "
            "ancilla measures one basis"
        )
    return x_qubits, z_qubits


def ancilla_band_counts(
    checks: tuple[CodeCheck, ...] | list[CodeCheck],
) -> tuple[int, int]:
    """Count the ancillas in each band, tolerating an ancilla that reads both.

    :func:`ancilla_bands` answers with the distinct ancilla each check measures,
    which is the set a detector band is laid out against, and it refuses an
    ancilla that reads both bases because one ancilla cannot be read in two
    bands. This accessor answers the same question -- how wide each band is --
    and differs only in that it does not refuse: an ancilla carrying a mixed
    check is counted in both bands instead of being rejected, so the count stays
    defined exactly where the band sets are not.

    That is what makes the mixed frame representable. Its checks read both bases
    through one ancilla, so the bands it defines overlap, and a record that
    reports its two widths is stating a number that exists. A record whose
    checks each measure one basis gets the same pair from either reading,
    because one ancilla then measures one check, and this is where a record's
    stated widths are compared against the widths its checks define.

    Args:
        checks: The checks whose per-band ancilla counts are being read.

    Returns:
        The ``(x_count, z_count)`` pair, each the number of distinct ancillas in
        that band.
    """

    return (
        len({check.ancilla_qubit for check in checks if check.stabilizer.x_qubits}),
        len({check.ancilla_qubit for check in checks if check.stabilizer.z_qubits}),
    )


@dataclass(frozen=True)
class CodeCheck:
    """One stabilizer check with its ancilla and its CNOT coupling.

    Each entry of ``cnot_qubits`` is a ``(control, target)`` pair, and every pair
    couples the check's single ancilla to one data qubit of the stabilizer's
    support. Which way a pair points is fixed by the stabilizer factor it
    carries rather than left to the caller:

    * A **Z factor** on a data qubit is read by coupling that data qubit into the
      ancilla with the data qubit controlling, so it comes first:
      ``(data_qubit, ancilla)``. That is the Z-type gadget: the ancilla is
      prepared in ``|0>`` and read in the Z basis.
    * An **X factor** is read by conjugating the ancilla with ``H`` and coupling
      it out into that data qubit with the ancilla controlling:
      ``(ancilla, data_qubit)``. That is the X-type gadget: the same ``H``
      prepares the ancilla in ``|+>`` and reads it back in the Z basis.

    Both gadgets leave the ancilla's Z-basis readout equal to the check's
    eigenvalue, which is what makes the two symmetric here. So ``cnot_qubits`` is
    the Z-factor pairs in the stabilizer's own qubit order, then the X-factor
    pairs in the stabilizer's own qubit order, and the boundary between the two
    runs is what says which pairs the ``H`` gates wrap. The emitted gadget is the
    Z pairs, then ``H``, then the X pairs, then ``H``, then a Z-basis readout of
    the ancilla -- which leaves that readout equal to the check's eigenvalue.

    **One ancilla measures a mixed stabilizer.** A pure Z-type check has an
    empty X run and no ``H`` gates at all; a pure X-type check has an empty Z
    run and is the ``H``-wrapped gadget alone. Those are the two shapes this
    record described before it described the mixed one, and they are not two
    rules beside a third: they are the two ends of the one rule above, and they
    are emitted by it unchanged.

    **Why the X run needs the H gates and the Z run does not.** Reading a Z
    factor through the ancilla is a CNOT in one direction, and reading an X
    factor is the same CNOT conjugated by ``H`` on the ancilla, because
    ``H X H = Z``. A mixed stabilizer needs both directions through the *same*
    ancilla, so it needs the ``H`` pair around the X half only -- never a second
    ancilla, and never an ``H`` that would also conjugate the Z half and turn
    the check into a different operator.
    """

    index: int
    stabilizer: Pauli
    ancilla_qubit: int
    cnot_qubits: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("check index must be non-negative")
        if not isinstance(self.stabilizer, Pauli):
            raise TypeError("check stabilizer must be a Pauli operator")
        if self.ancilla_qubit < 0:
            raise ValueError("check ancilla qubit must be non-negative")
        if not self.cnot_qubits:
            raise ValueError("check must declare at least one CNOT")
        for control, target in self.cnot_qubits:
            if control < 0 or target < 0:
                raise ValueError("check CNOT qubits must be non-negative")
        if not self.stabilizer.support:
            raise ValueError(
                "check stabilizer must have a non-empty support, because an "
                "identity check measures nothing and its ancilla has nothing to "
                "read"
            )
        self._validate_z_run()
        self._validate_x_run()

    def _validate_z_run(self) -> None:
        """Require the leading run to be the Z factors, each into the ancilla."""

        z_qubits = self.stabilizer.z_qubits
        run = self.cnot_qubits[: len(z_qubits)]
        for _, target in run:
            if target != self.ancilla_qubit:
                raise ValueError(
                    "check CNOT targets an X-factor pair, which must come after "
                    "the Z-factor pairs, or couples a qubit the stabilizer does "
                    "not carry a Z factor on"
                )
        controls = tuple(sorted(control for control, _ in run))
        if self.ancilla_qubit in controls:
            raise ValueError(
                "check CNOT controls must be data qubits, not the ancilla, "
                "because a Z-factor pair draws a data qubit into the ancilla"
            )
        if controls != z_qubits:
            raise ValueError(
                "check CNOT controls must match the stabilizer's Z-factor support"
            )

    def _validate_x_run(self) -> None:
        """Require the trailing run to be the X factors, each out of the ancilla."""

        x_qubits = self.stabilizer.x_qubits
        run = self.cnot_qubits[len(self.stabilizer.z_qubits) :]
        for control, _ in run:
            if control != self.ancilla_qubit:
                raise ValueError(
                    "check CNOT controls must be the declared ancilla, because an "
                    "X-factor pair is controlled by the declared ancilla and its "
                    "target data qubits are the stabilizer's X-factor support"
                )
        targets = tuple(sorted(target for _, target in run))
        if targets != x_qubits:
            raise ValueError(
                "check CNOT targets must match the stabilizer's X-factor support"
            )


@runtime_checkable
class StabilizerCode(Protocol):
    """A code that declares its qubit layout, checks, and logical observables.

    The ancilla count is stated twice on purpose: once as a total, and once per
    basis. Upstream's code record carries the same three accessors as pure
    virtuals -- ``get_num_ancilla_qubits``, ``get_num_ancilla_x_qubits`` and
    ``get_num_ancilla_z_qubits`` -- and the split is what a syndrome-extraction
    round and a detector band are laid out against, so a total alone does not say
    which ancillas a round has to visit. :func:`ancilla_bands` derives the two
    bands from ``checks``, and the records in this module read their counts from
    it, so a record cannot report a split its own checks contradict.

    ``num_x_stabilizers`` and ``num_z_stabilizers`` are the two further counts
    upstream's record declares beside those three, as the pure virtuals
    ``get_num_x_stabilizers`` and ``get_num_z_stabilizers``. Each of the three
    code classes upstream ships answers the same number to a stabilizer count as
    to the band count of the same basis, because one ancilla measures one
    stabilizer there exactly as it does here; the two are therefore one quantity
    in this repository rather than two, read once through
    :func:`ancilla_bands`, and these two accessors are aliases of the two band
    accessors rather than a second derivation. A caller reading upstream's
    accessor list finds the same five counts here, and a record that stated the
    same quantity a second time could state a different answer to it.
    """

    @property
    def distance(self) -> int: ...

    @property
    def num_data_qubits(self) -> int: ...

    @property
    def num_ancilla_qubits(self) -> int: ...

    @property
    def num_ancilla_x_qubits(self) -> int: ...

    @property
    def num_ancilla_z_qubits(self) -> int: ...

    @property
    def num_x_stabilizers(self) -> int: ...

    @property
    def num_z_stabilizers(self) -> int: ...

    @property
    def data_qubits(self) -> tuple[int, ...]: ...

    @property
    def ancilla_qubits(self) -> tuple[int, ...]: ...

    @property
    def checks(self) -> tuple[CodeCheck, ...]: ...

    @property
    def stabilizers(self) -> tuple[Pauli, ...]: ...

    @property
    def logical_observables(self) -> tuple[Pauli, ...]: ...


_retain_module(
    "flagquantum.qec.codes",
    ancilla_bands,
    ancilla_band_counts,
    CodeCheck,
    StabilizerCode,
)
