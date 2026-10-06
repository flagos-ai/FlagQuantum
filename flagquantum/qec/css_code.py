"""A code record whose parity-check matrices are the source of truth.

The code records in :mod:`flagquantum.qec.codes` are declared: a class states its
qubit layout and builds its checks from it. This module is the other direction of
the same seam. A caller who has the four Calderbank-Shor-Steane matrices and no
class declares a :class:`CssCode`, and every count and layout a code record owes
-- the data qubits, the ancilla total and its two bands, the checks, the
stabilizers, the logical observables -- is read off those matrices rather than
asserted beside them.

Upstream divides the same two things between two headers, and this module follows
that division rather than inventing one: ``code.h`` declares the code record, and
``code_matrices.h`` declares ``css_code_matrices`` as the input interface of
detector error model construction. :class:`~flagquantum.qec.CssCodeMatrices`
therefore stays with the construction that reads it, and this record is built on
top of it.

Two differences from a declared record are deliberate and are stated here rather
than discovered by a caller.

The distance is stated, not derived. Upstream's code record declares no distance
accessor at all: a concrete class reads one out of the options it was built with,
and its own factory refuses a construction that omits it. Computing a quantum
code's distance is a minimum-weight-codeword problem and is exponential in
general, so a record that derived it would refuse exactly the large matrices this
route exists to serve. What the record does instead is refuse every distance its
own matrices contradict, at a cost that does not grow with a search: a stated
logical operator of weight ``w`` proves the code distance is at most ``w``, so a
record claiming more than the lightest operator it states is refused rather than
trusted. That is one direction of proof and not two, and a record that understates
its distance is therefore accepted, because the direction that was dropped is the
one that cannot mislead a caller into believing a code protects more than it does.

The record is not registered by name. A name is how a caller who knows which code
it wants avoids knowing which class declares it, and the identity of a CSS record
is the matrix the caller hands over, which a name adds nothing to.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch

from .codes import CodeCheck, ancilla_bands
from .dem_construction import CssCodeMatrices
from .pauli import Pauli


def _row_masks(block: torch.Tensor, *, what: str) -> tuple[int, ...]:
    """Read a matrix block as one integer bitmask per row.

    Bit ``q`` of a row mask is column ``q``, so the two operations this module
    needs -- the overlap parity of two rows and elimination over GF(2) -- are one
    ``&`` and one ``^`` rather than a loop over columns.
    """

    masks: list[int] = []
    for index, row in enumerate(block.tolist()):
        mask = 0
        for column, entry in enumerate(row):
            if entry:
                mask |= 1 << column
        if not mask:
            raise ValueError(
                f"{what} {index} is empty, so it states the identity operator: a "
                "check and a logical observable must each act on at least one data "
                "qubit"
            )
        masks.append(mask)
    return tuple(masks)


def _support(mask: int) -> tuple[int, ...]:
    """Return the ascending qubit indices a row mask sets."""

    support: list[int] = []
    current = mask
    while current:
        low = current & -current
        support.append(low.bit_length() - 1)
        current ^= low
    return tuple(support)


def _row_basis(rows: tuple[int, ...]) -> dict[int, int]:
    """Reduce ``rows`` over GF(2) into a basis keyed by each row's pivot column."""

    basis: dict[int, int] = {}
    for row in rows:
        current = row
        for pivot in sorted(basis, reverse=True):
            if (current >> pivot) & 1:
                current ^= basis[pivot]
        if current:
            basis[current.bit_length() - 1] = current
    return basis


def _in_span(row: int, basis: dict[int, int]) -> bool:
    """Whether ``row`` is a product of the basis vectors, i.e. reduces to zero."""

    current = row
    for pivot in sorted(basis, reverse=True):
        if (current >> pivot) & 1:
            current ^= basis[pivot]
    return current == 0


@dataclass(frozen=True, eq=False)
class CssCode:
    """A Calderbank-Shor-Steane code declared by its parity-check matrices.

    Column ``q`` of every block of ``matrices`` is data qubit ``q``, so the data
    qubits are ``0..n-1`` and the checks take the ancillas that follow them, in
    the order the two check blocks state their rows: the Z-type checks in the
    order of ``hz``, then the X-type checks in the order of ``hx``. That is the
    order the declared records already use, and it is why a record rebuilt from
    another record's matrices describes the same experiment with its detectors
    permuted by the check order rather than a different experiment: a detector's
    identity is the check and the round it belongs to, and a matrix carries no
    qubit labels for a rebuild to preserve.

    The record checks the algebra the four blocks do not.
    :class:`~flagquantum.qec.CssCodeMatrices` holds the blocks to a common width
    and to binary entries, which says the matrices are well formed and says
    nothing about whether they describe a code. This record reads each stated
    logical operator against the opposite-basis checks and refuses one that fails
    either half of the definition -- a Z-type operator is a logical operator when
    it commutes with every X-type check and is not a product of the Z-type checks,
    and the X-type case is the mirror -- so matrices that do not describe a code
    are refused at construction rather than decoded against a model of nothing.
    The stated logical operators of one basis must also be independent, because
    each one becomes one observable and two rows spanning one logical operator
    would state two observables for it.

    The pairing between the two logical families is not checked. This package's
    records state one readout basis by convention, so an X-type logical list is
    usually absent and a pairing check would be vacuous exactly where a caller
    would expect it to bite.

    `~flagquantum.qec.CssCodeMatrices` stores every block as a tensor, because
    it normalizes an omitted one to an empty block, so the three optional fields
    are the tensors they hold once a record carries them.

    The distance is passed in and is refused when the record's own matrices
    contradict it, as the module docstring states; it is not derived.

    Args:
        matrices: The four sparse binary blocks a CSS code is read by.
        distance: The code distance, which the caller states for the same reason
            upstream's own factory requires a ``distance`` option.

    Raises:
        TypeError: ``matrices`` is not a `~flagquantum.qec.CssCodeMatrices`, or
            ``distance`` is not an integer.
        ValueError: ``distance`` is not positive, or it exceeds the weight of the
            lightest logical operator the matrices state; the matrices state no
            check or no logical observable; a check row is empty; a logical
            operator fails either half of its definition, naming the row and the
            reason; or two stated logical operators of one basis are dependent.

    Examples:
        >>> import torch
        >>> from flagquantum.qec import CssCode, CssCodeMatrices
        >>> record = CssCode(
        ...     matrices=CssCodeMatrices(
        ...         hz=torch.tensor([[1, 1, 0], [0, 1, 1]]),
        ...         lz=torch.tensor([[1, 1, 1]]),
        ...     ),
        ...     distance=3,
        ... )
        >>> record.num_data_qubits, record.num_ancilla_qubits, record.distance
        (3, 2, 3)
    """

    matrices: CssCodeMatrices
    distance: int

    def __post_init__(self) -> None:
        if not isinstance(self.matrices, CssCodeMatrices):
            raise TypeError(
                "a CSS code record is built from a CssCodeMatrices, not from "
                f"{type(self.matrices).__name__}"
            )
        if isinstance(self.distance, bool) or not isinstance(self.distance, Integral):
            raise TypeError("CSS-code distance must be an integer")
        if self.distance < 1:
            raise ValueError("CSS-code distance must be at least one")
        object.__setattr__(self, "distance", int(self.distance))
        self._require_a_code()

    def _require_a_code(self) -> None:
        z_checks = _row_masks(self.matrices.hz, what="Z-type check")
        x_checks = _row_masks(
            self.matrices.hx,  # type: ignore[arg-type]
            what="X-type check",
        )
        if not z_checks and not x_checks:
            raise ValueError(
                "the matrices state no check, so they do not describe a code: a "
                "CSS code needs at least one stabilizer generator"
            )
        z_logicals = _row_masks(
            self.matrices.lz,  # type: ignore[arg-type]
            what="Z-type logical observable",
        )
        x_logicals = _row_masks(
            self.matrices.lx,  # type: ignore[arg-type]
            what="X-type logical observable",
        )
        if not z_logicals and not x_logicals:
            raise ValueError(
                "the matrices state no logical observable, so they do not describe "
                "an error-correcting code: a distance describes how many faults a "
                "logical operator survives"
            )

        self._require_logical(
            z_logicals, x_checks, _row_basis(z_checks), basis="Z", checks="X-type"
        )
        self._require_logical(
            x_logicals, z_checks, _row_basis(x_checks), basis="X", checks="Z-type"
        )
        self._require_distance_fits(z_logicals, x_logicals)

    @staticmethod
    def _require_logical(
        logicals: tuple[int, ...],
        opposite_checks: tuple[int, ...],
        own_check_basis: dict[int, int],
        *,
        basis: str,
        checks: str,
    ) -> None:
        """Hold one basis's stated logical operators to both halves of the definition.

        A ``basis``-type operator is a logical operator when every ``checks``
        operator meets it on an even number of qubits -- the two Paulis then
        commute -- and when it is not a product of the ``basis``-type checks
        themselves. The independence of the stated rows follows from the same
        reading: each row becomes one observable, so two rows spanning one logical
        operator would state that operator twice.
        """

        for index, logical in enumerate(logicals):
            for check_index, check in enumerate(opposite_checks):
                if (logical & check).bit_count() % 2:
                    raise ValueError(
                        f"{basis}-type logical observable {index} overlaps {checks} "
                        f"check {check_index} on an odd number of qubits, so the two "
                        "anticommute and the stated operator is not a logical "
                        "operator of this code"
                    )
            if _in_span(logical, own_check_basis):
                raise ValueError(
                    f"{basis}-type logical observable {index} is a product of the "
                    f"code's own {basis}-type checks, so it is a stabilizer rather "
                    "than a logical operator"
                )
        if len(_row_basis(logicals)) != len(logicals):
            raise ValueError(
                f"the {basis}-type logical observables are not independent: they "
                f"span {len(_row_basis(logicals))} logical operator(s) over "
                f"{len(logicals)} stated row(s), and each row becomes one observable"
            )

    def _require_distance_fits(
        self, z_logicals: tuple[int, ...], x_logicals: tuple[int, ...]
    ) -> None:
        lightest = min(mask.bit_count() for mask in z_logicals + x_logicals)
        if self.distance > lightest:
            raise ValueError(
                f"the stated distance {self.distance} exceeds the weight {lightest} "
                "of the lightest logical operator these matrices state, and a stated "
                "logical operator proves the distance is at most its own weight: "
                "state a logical operator of that weight, or a smaller distance"
            )

    @property
    def num_data_qubits(self) -> int:
        return self.matrices.num_qubits

    @property
    def num_ancilla_qubits(self) -> int:
        return len(self.checks)

    @property
    def num_ancilla_x_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[0])

    @property
    def num_ancilla_z_qubits(self) -> int:
        return len(ancilla_bands(self.checks)[1])

    @property
    def num_x_stabilizers(self) -> int:
        return self.num_ancilla_x_qubits

    @property
    def num_z_stabilizers(self) -> int:
        return self.num_ancilla_z_qubits

    @property
    def data_qubits(self) -> tuple[int, ...]:
        return tuple(range(self.num_data_qubits))

    @property
    def ancilla_qubits(self) -> tuple[int, ...]:
        first = self.num_data_qubits
        return tuple(first + index for index in range(self.num_ancilla_qubits))

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        checks: list[CodeCheck] = []
        for index, mask in enumerate(_row_masks(self.matrices.hz, what="Z-type check")):
            support = _support(mask)
            ancilla = self.num_data_qubits + index
            checks.append(
                CodeCheck(
                    index=index,
                    stabilizer=Pauli(z_qubits=support),
                    ancilla_qubit=ancilla,
                    cnot_qubits=tuple((qubit, ancilla) for qubit in support),
                )
            )
        offset = len(checks)
        x_checks = _row_masks(
            self.matrices.hx,  # type: ignore[arg-type]
            what="X-type check",
        )
        for index, mask in enumerate(x_checks):
            support = _support(mask)
            ancilla = self.num_data_qubits + offset + index
            checks.append(
                CodeCheck(
                    index=offset + index,
                    stabilizer=Pauli(x_qubits=support),
                    ancilla_qubit=ancilla,
                    cnot_qubits=tuple((ancilla, qubit) for qubit in support),
                )
            )
        return tuple(checks)

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        z_masks = _row_masks(
            self.matrices.lz,  # type: ignore[arg-type]
            what="Z-type logical observable",
        )
        x_masks = _row_masks(
            self.matrices.lx,  # type: ignore[arg-type]
            what="X-type logical observable",
        )
        return tuple(
            [Pauli(z_qubits=_support(mask)) for mask in z_masks]
            + [Pauli(x_qubits=_support(mask)) for mask in x_masks]
        )

    def __eq__(self, other: object) -> bool:
        """Compare two records by the matrices and the distance they were built from.

        The comparison is written out rather than generated, for the reason
        `~flagquantum.qec.DemSample` states: a generated one would ask ``bool()``
        of a tensor, and a generated hash would not agree with a content
        comparison, so instances are deliberately unhashable.
        """

        if not isinstance(other, CssCode):
            return NotImplemented
        if self.distance != other.distance:
            return False
        return all(
            bool(getattr(self.matrices, name).equal(getattr(other.matrices, name)))
            for name in ("hz", "hx", "lz", "lx")
        )


__all__ = ("CssCode",)
