"""Certification of a code's logical operators in a chosen readout frame.

A code declares its distance, its wires, its checks, and its logical
observables. The declared observables are read out by measuring every data wire
in the Z basis, which is the frame a memory experiment is normally stated in. A
memory experiment stated in another frame measures a *product* of the same
code's logical operators instead, and such a product is only a logical operator
if it is one the code actually protects.

This module decides that question. It certifies a candidate operator against the
code's own checks, and it derives the partner of a declared observable when a
caller needs to read a code out in the other basis. It does nothing else: it
builds no circuit, executes nothing, and holds no state.
"""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

from .gf2 import in_span
from .pauli import Pauli

if TYPE_CHECKING:
    from .codes import StabilizerCode


def _symplectic_mask(pauli: Pauli, positions: dict[int, int], width: int) -> int:
    """Return ``pauli`` as one X bit and one Z bit per data wire.

    The two halves are packed into one integer so that a whole stabilizer group is
    a set of GF(2) rows: bit ``p`` is the X factor on the ``p``-th data wire and
    bit ``width + p`` its Z factor. A wire listed in both carries ``Y``, which is
    what makes this the correct mask for a **mixed** check -- a check that is a
    product of the two families is neither an X-type row nor a Z-type row, and a
    mask that recorded only its support would conflate it with an operator of a
    different Pauli type.
    """

    bits = 0
    for wire in pauli.x_wires:
        bits |= 1 << positions[wire]
    for wire in pauli.z_wires:
        bits |= 1 << (width + positions[wire])
    return bits


def _z_part_mask(pauli: Pauli, positions: dict[int, int]) -> int:
    """Return only ``pauli``'s Z support, as one bit per data wire.

    An X-type candidate commutes with an operator exactly when the two agree on an
    even number of wires the operator carries a ``Z`` factor on, so a Z factor is
    the whole of what constrains it and an X factor constrains it not at all. This
    is the mask a commutation constraint is stated with, and it is why a mixed
    check constrains an X-type candidate as much as a pure Z-type check does.
    """

    bits = 0
    for wire in pauli.z_wires:
        bits |= 1 << positions[wire]
    return bits


def _solve(rows: list[int], rhs: list[int]) -> int | None:
    """Solve ``A x = b`` over GF(2), or return ``None`` when it is inconsistent.

    Rows are bit vectors over ``len(rows[0])``-ish width and ``rhs`` holds one bit
    per row. The returned vector is the unique solution whose values at the
    columns the elimination left free are zero, so the answer depends only on the
    row order and not on the order the elimination happened to visit columns in.
    """

    table: dict[int, int] = {}
    for row, target in zip(rows, rhs, strict=True):
        value = (row << 1) | target
        while value >> 1:
            pivot = (value >> 1).bit_length() - 1
            if pivot not in table:
                table[pivot] = value
                break
            value ^= table[pivot]
        else:
            if value & 1:
                return None
    solution = 0
    for pivot in sorted(table):
        value = table[pivot]
        if bin((value >> 1) & solution).count("1") % 2 != value & 1:
            solution |= 1 << pivot
    return solution


def certify_logical_product(code: StabilizerCode, product: Pauli) -> Pauli:
    """Return ``product`` once it is certified as a logical operator of ``code``.

    A candidate is a logical operator when it commutes with every one of the
    code's checks and is not itself a product of them. The first condition says
    it preserves the code space, so its outcome can be read out at all; the
    second says it is not a stabilizer, whose outcome is fixed at ``+1`` and
    therefore carries no information. Both are refused rather than repaired,
    because a candidate that fails either is not the operator the caller named.

    Only a pure X-type or pure Z-type candidate is accepted. Each wire of the
    product is measured in one basis, and a wire carrying both an ``X`` and a
    ``Z`` factor would need a third basis. Identity is refused for the same
    reason an identity observable is: it is measured by nothing.

    The **checks** are under no such restriction, and the second condition is
    asked of all of them together. A check of a code whose stabilizers mix the two
    factors is a product of both, so being a product of checks is a question about
    the group rather than about one family of it, and it is answered over the full
    X-and-Z description of every check. A candidate that is a product of a code's
    mixed checks is a stabilizer even though it matches no single family's rows.
    """

    if not isinstance(product, Pauli):
        raise TypeError("logical product must be a Pauli operator")
    if product.is_identity:
        raise ValueError("logical product must not be the identity")
    if product.x_wires and product.z_wires:
        raise ValueError(
            "logical product must be a pure X-type or pure Z-type operator, "
            "because each wire of the product is read out in a single basis "
            "and a Y factor would need a second rotation"
        )
    undeclared = sorted(set(product.support) - set(code.data_wires))
    if undeclared:
        raise ValueError(
            f"logical product wires {tuple(undeclared)} are not declared data "
            "wires of the code"
        )
    for check in code.checks:
        if not product.commutes_with(check.stabilizer):
            raise ValueError(
                "logical product must commute with every check of the code, and "
                f"it anticommutes with check {check.index}"
            )
    positions = {wire: index for index, wire in enumerate(code.data_wires)}
    width = len(code.data_wires)
    # Every check constrains the candidate, of whichever type it is. An operator
    # that is a product of checks is the identity element of the stabilizer group
    # only as a symplectic vector, so the span is asked of the full X-and-Z
    # description: a candidate that agrees with a mixed check's X half but not its
    # Z half is not a stabilizer, and a support-only test could not tell.
    rows = [
        _symplectic_mask(check.stabilizer, positions, width) for check in code.checks
    ]
    if in_span(_symplectic_mask(product, positions, width), rows):
        raise ValueError(
            "logical product must not lie in the code's stabilizer span, because "
            "a stabilizer outcome is fixed and the terminal readout would report "
            "a constant rather than the operator's eigenvalue"
        )
    return product


def derive_anticommuting_logical_product(code: StabilizerCode, index: int = 0) -> Pauli:
    """Return the pure X-type logical operator paired with one declared observable.

    A code's declared observables are read out in the Z basis. Reading the code
    out in the X basis instead means measuring the operator that anticommutes with
    one of them, and a code need not declare that operator: the surface patches
    declare only their Z-type logical observable. This function derives it.

    The derivation is a GF(2) linear system over the code's data wires. An X-type
    operator commutes with an operator exactly when it overlaps that operator's
    ``Z`` factor on an even number of wires, so a check constrains the candidate
    through its ``Z`` factor alone and a check carrying none constrains it not at
    all. A pure X-type check is therefore free, a pure Z-type check is a full
    constraint, and a **mixed** check is a constraint of exactly the size of its
    ``Z`` half -- which is the reason the rows are stated per check rather than
    per family. The constraint matrix has one row per check, each read on that
    check's ``Z`` factor, plus one row per declared Z-type observable, and the
    right-hand side is one at the target observable's row and zero at every other.

    Raises:
        ValueError: If ``index`` names no declared Z-type observable, or if the
            code has no such operator. The latter is not a defect in the code: a
            repetition code of even length has an all-Z operator that is a
            product of its checks, so nothing in its code space anticommutes with
            the operator it declares, and no X-basis readout of it exists.
    """

    if isinstance(index, bool) or not isinstance(index, Integral):
        raise TypeError("observable index must be an integer")
    readable = tuple(
        observable for observable in code.logical_observables if not observable.x_wires
    )
    if index < 0 or index >= len(readable):
        raise ValueError(
            f"the code declares no Z-type observable at index {index}, and only a "
            "Z-type observable has an X-type partner to derive"
        )
    positions = {wire: position for position, wire in enumerate(code.data_wires)}
    # Every check's Z factor is a constraint, a mixed check's included: an X-type
    # candidate anticommutes with a check exactly when it overlaps that check's Z
    # support oddly. A check carrying no Z factor therefore contributes a row of
    # zeros, which constrains nothing and is dropped by the elimination itself
    # rather than by a type test here.
    z_checks = [_z_part_mask(check.stabilizer, positions) for check in code.checks]
    z_observables = [_z_part_mask(observable, positions) for observable in readable]
    rhs = [0] * (len(z_checks) + len(z_observables))
    rhs[len(z_checks) + index] = 1
    solution = _solve([*z_checks, *z_observables], rhs)
    if solution is None:
        raise ValueError(
            f"the code declares no X-type logical operator that anticommutes with "
            f"observable {index}, so the code cannot be read out in the X basis"
        )
    wires = tuple(
        wire
        for position, wire in enumerate(code.data_wires)
        if solution >> position & 1
    )
    return certify_logical_product(code, Pauli(x_wires=wires))


__all__ = ("certify_logical_product", "derive_anticommuting_logical_product")
