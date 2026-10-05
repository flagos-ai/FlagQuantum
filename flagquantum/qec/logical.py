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


def _mask(pauli: Pauli, positions: dict[int, int]) -> int:
    """Return ``pauli`` as one bit per data wire, in ``positions`` order."""

    bits = 0
    for wire in pauli.support:
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
    wanted_x_type = bool(product.x_wires)
    rows = [
        _mask(check.stabilizer, positions)
        for check in code.checks
        if bool(check.stabilizer.x_wires) == wanted_x_type
    ]
    if in_span(_mask(product, positions), rows):
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
    operator commutes with every X-type check automatically, so the constraints
    that matter are commutation with the Z-type checks and the anticommutation
    with exactly one declared Z-type observable -- the one at ``index``. The
    constraint matrix therefore has one row per Z-type check and per declared
    Z-type observable, all read over the data wires, and the right-hand side is
    one at the target row and zero at every other.

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
    z_checks = [
        _mask(check.stabilizer, positions)
        for check in code.checks
        if not check.stabilizer.x_wires
    ]
    z_observables = [_mask(observable, positions) for observable in readable]
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
