"""The bivariate-bicycle family, and the logical basis read out of its matrices.

A bivariate-bicycle code is written down on a group of qubits rather than on a
lattice of checks. The qubits are the ``x_period * y_period`` elements of
``Z_x_period * Z_y_period``, two polynomials ``A`` and ``B`` over that group state
two blocks of that size, and the code's matrices are ``Hx = [A | B]`` and
``Hz = [B^T | A^T]``. Every check therefore spans at most six data qubits, the
data-qubit count is ``2 * x_period * y_period``, and the family's interest is that
the distance grows with the periods while the check weight does not: the smallest
published instance is ``[[72, 12, 6]]`` against the torus's weight-four checks, and
it carries twelve logical qubits on the seventy-two data qubits a distance-six
surface patch spends on one.

**The logical operators are read out of the matrices rather than named by the
rule.** This is the seam between this module and :mod:`flagquantum.qec.codes`.
That module holds the record a code arrives as and this one holds a rule, and the
rule differs from the two derived families beside it in one respect that decides
where the work goes: a triangular colour patch and a square-lattice torus can
*declare* their logical operators, because a side of a small triangle and a ring of
a small torus can be written down by hand. A pair of polynomials names no such
curve, so the Z-type logical operators are taken to be what the X-type checks leave
at zero, modulo the Z-type stabilizer span, and the X-type ones are the mirror of
that. That reading names no polynomial and needs none, so it is not stated here:
:mod:`flagquantum.qec.qldpc` owns it as a route a caller can also reach directly,
and this module calls those two helpers rather than carrying a second copy of them.

**The two bases are then changed into the one the record states.** A code's logical
operators pair up non-degenerately but the null-space reading above returns an
arbitrary basis of each family, and a record is handed the operators themselves
rather than the classes they lie in. The change of basis is a symplectic
elimination over the pairing matrix: two rows of it are swapped until the diagonal
holds a one, and the Z-type operator of every other row that pairs with it is
multiplied by it until it does not. What comes back is a family of ``k`` operators
each pairing with exactly one operator of the other family, which is the form
:class:`~flagquantum.qec.codes.CssCode` certifies and the form a memory experiment
reads out. That elimination is shared for the same reason, because a
bivariate-bicycle code is one of the codes the direct route states.

This module builds one record and searches for its distance. It does not decode,
protect a logical qubit, state a threshold, or choose a noise model.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

from .codes import _DISTANCE_SEARCH_WEIGHT, CssCode
from .qldpc import _logical_basis, _paired_basis, _rows

__all__ = ("bivariate_bicycle_code",)


def _monomial_exponents(term: Sequence[int], name: str) -> tuple[int, int]:
    """Return one term's two exponents, refusing anything that is not a pair."""

    if isinstance(term, (str, bytes)) or not isinstance(term, Sequence):
        raise TypeError(
            f"every term of {name} must be a pair of exponents, and {term!r} is not "
            "a sequence of two integers"
        )
    exponents = tuple(term)
    if len(exponents) != 2:
        raise ValueError(
            f"every term of {name} must name one monomial by its two exponents, and "
            f"{term!r} names {len(exponents)}"
        )
    for exponent in exponents:
        if isinstance(exponent, bool) or not isinstance(exponent, Integral):
            raise TypeError(
                f"every exponent of {name} must be an integer, and {exponent!r} is "
                "not one"
            )
    return int(exponents[0]), int(exponents[1])


def _polynomial_masks(
    x_period: int,
    y_period: int,
    terms: Sequence[Sequence[int]],
    name: str,
) -> list[int]:
    """Return multiplication by one polynomial, as one mask per group element.

    The group is indexed so that element ``(i, j)`` is bit ``i * y_period + j``, and
    the monomial ``x**a y**b`` carries that element to ``(i + a, j + b)`` with both
    coordinates wrapped by their own period. Multiplication by the whole polynomial
    is the exclusive-or of those displacements, one per term, which is what makes a
    repeated monomial cancel: two terms that name one monomial state it twice and
    the polynomial they state does not carry it at all.
    """

    width = x_period * y_period
    masks = [0] * width
    for term in terms:
        a, b = _monomial_exponents(term, name)
        for i in range(x_period):
            for j in range(y_period):
                source = i * y_period + j
                target = ((i + a) % x_period) * y_period + ((j + b) % y_period)
                masks[source] ^= 1 << target
    if not any(masks):
        raise ValueError(
            f"{name} names no monomial: a bivariate-bicycle code is stated by two "
            "polynomials over its group, and the zero polynomial states no check"
        )
    return masks


def _transpose(masks: list[int], width: int) -> list[int]:
    """Return the transpose of one block, as one mask per column."""

    transposed = [0] * width
    for row, mask in enumerate(masks):
        for column in range(width):
            if mask >> column & 1:
                transposed[column] |= 1 << row
    return transposed


def bivariate_bicycle_code(
    x_period: int,
    y_period: int,
    a_terms: Sequence[Sequence[int]],
    b_terms: Sequence[Sequence[int]],
    *,
    distance_search_weight: int = _DISTANCE_SEARCH_WEIGHT,
) -> CssCode:
    """Return the bivariate-bicycle code two polynomials name.

    The qubits are the ``x_period * y_period`` elements of ``Z_x_period *
    Z_y_period``, doubled because the code is Calderbank-Shor-Steane: the first
    ``x_period * y_period`` carry the X-type checks and the second half carries the
    Z-type ones. Each polynomial is a sum of monomials over that group, and the
    checks are its regular representation -- ``Hx = [A | B]`` and ``Hz = [B^T |
    A^T]`` -- so a term ``(a, b)`` states the monomial ``x**a y**b`` and an exponent
    is read modulo its own period, which is what makes ``-1`` the last element
    rather than a refusal.

    **The logical operators are derived, and the distance is searched for.** The
    record takes the two families' matrices and works both out from them, so the
    parameters above are the whole of what a caller states and no operator is
    transcribed from anywhere. Both are the reason this route costs what it costs:
    the search tries every wire subset in increasing weight up to
    ``distance_search_weight``, so a published instance of distance six over 72 data
    qubits takes about eighteen seconds to build while the small instances an
    experiment is demonstrated on are immediate. Measured on this implementation:
    0.00 s for the ``[[18, 4, 4]]`` parameters below and 18.5 s for the published
    ``[[72, 12, 6]]`` of ``x_period = y_period = 6`` with ``A = x**3 + y + y**2`` and
    ``B = y**3 + x + x**2``. The bound is therefore the caller's to accept, in the
    same sense the record means it: a bound the search does not reach is refused
    rather than reported as a distance.

    Args:
        x_period: The order of ``x`` in the group the qubits are indexed by, an
            integer of at least two.
        y_period: The order of ``y``, an integer of at least two. The code has
            ``2 * x_period * y_period`` data qubits and as many checks, one per
            group element and family.
        a_terms: The monomials of ``A``, each a pair of exponents. A monomial named
            twice cancels, so a pair of equal terms states no term.
        b_terms: The monomials of ``B``, in the same form.
        distance_search_weight: How far the distance search may look. The default is
            the record's own default, which admits every distance-three instance; a
            published instance of greater distance needs a bound the search reaches.

    Returns:
        The code record, with as many logical qubits as the two check families leave
        over the data qubits and both family distances searched for.

    Raises:
        TypeError: If a period is not an integer, a term is not a sequence of two
            integers, or an exponent is not an integer.
        ValueError: If a period is below two, a polynomial names no monomial, the
            bound is not at least one, or a family has no logical operator within
            ``distance_search_weight``.

    Examples:
        >>> code = bivariate_bicycle_code(
        ...     3,
        ...     3,
        ...     ((0, 0), (0, 1), (1, 0)),
        ...     ((0, 0), (0, 1), (2, 1)),
        ...     distance_search_weight=4,
        ... )
        >>> code.num_data_qubits, code.num_ancilla_qubits
        (18, 18)
        >>> code.distance, code.x_distance, code.z_distance
        (4, 4, 4)
        >>> len(code.logical_observables)
        8
    """

    for period, name in ((x_period, "x_period"), (y_period, "y_period")):
        if isinstance(period, bool) or not isinstance(period, Integral):
            raise TypeError(f"{name} must be an integer")
        if period < 2:
            raise ValueError(
                f"{name} is {period}, and a bivariate-bicycle code is stated over "
                "two cyclic coordinates of order at least two: one of them trivial "
                "states a one-dimensional bicycle rather than this family"
            )
    if isinstance(distance_search_weight, bool) or not isinstance(
        distance_search_weight, Integral
    ):
        raise TypeError("distance search weight must be an integer")

    size_x = int(x_period)
    size_y = int(y_period)
    block = _polynomial_masks(size_x, size_y, a_terms, "a_terms")
    other = _polynomial_masks(size_x, size_y, b_terms, "b_terms")
    width = size_x * size_y
    block_t = _transpose(block, width)
    other_t = _transpose(other, width)
    x_checks = [block[index] | other[index] << width for index in range(width)]
    z_checks = [other_t[index] | block_t[index] << width for index in range(width)]
    total = 2 * width
    z_logicals = _logical_basis(x_checks, z_checks, total)
    x_logicals = _logical_basis(z_checks, x_checks, total)
    z_logicals, x_logicals = _paired_basis(z_logicals, x_logicals)
    return CssCode(
        hz=_rows(z_checks, total),
        hx=_rows(x_checks, total),
        lz=_rows(z_logicals, total),
        lx=_rows(x_logicals, total),
        distance_search_weight=int(distance_search_weight),
    )
