"""The punctured Reed-Muller family, and the convention its checks are stated in.

A Reed-Muller code is written down on the Boolean cube rather than on a lattice:
``RM(degree, m)`` is the span of the monomials of degree at most ``degree``
evaluated at all ``2**m`` points of ``F_2**m``. This module states the pair of such
codes two degrees apart, punctured at the point where every variable is one, and
returns the Calderbank-Shor-Steane code whose two check families are the duals of
the two punctured blocks. It is the first family in this package that is neither a
lattice patch nor a pair of polynomials over a group, and its smallest instance is
fifteen qubits: the family's interest is that its length is a power of two minus
one while its checks stay low in weight, so it reaches a code of eleven logical
qubits at length thirty-one where a lattice patch of that length carries one.

**Which block states which check family is a convention, and the convention is what
decides which logical family is the light one.** The two readings of "the punctured
quantum Reed-Muller code" differ by swapping the X-type and Z-type blocks and by
nothing else, and both were measured here before this rule was written. With the
higher-degree block's dual stating the X-type checks and the lower-degree block's
dual stating the Z-type checks, the least X-type logical operator has weight seven
and the least Z-type one has weight three, so the code's distance is three; with the
two blocks exchanged the two numbers exchange with them and the code's distance is
again three. This module states the first reading, which is the one
``contracts/cudaq-parity-matrix.toml`` names, and a caller who wants the mirror has
it directly: that pair of blocks through :func:`~flagquantum.qec.qldpc.qldpc_code`
is the same code with the two families labelled the other way round.

**The search bound is the family's cost, and it bounds this route to two
instances.** A record's distance is computed rather than declared, so the least
weight of each family is found by trying wire subsets in increasing order, and
proving that the lighter family's least operator has weight seven means exhausting
every subset of weight up to six. That is affordable at the two smallest sizes and
not above them, and the module refuses above them by name rather than by waiting:
measured on this tree, ``m = 4`` builds in 0.005 s and ``m = 5`` in 0.527 s, while
the same search at ``m = 6`` would have to look through the 75611760 subsets of
weight up to six alone and at ``m = 7`` through 5434287328, against 942648 at
``m = 5``. The rule above is stated for every
``m`` at or above four; what is bounded is the route this module offers, and it is
bounded in the open, exactly as :func:`~flagquantum.qec.codes.triangular_colour_code`
bounds the colour patches it builds.

The rule's domain starts at four rather than three because three is not a code: the
two punctured blocks of a three-cube are one and zero degrees apart, their duals
have three and six rows over seven points, and more check rows than points cannot
commute -- ten of those eighteen pairs anticommute, so the pair states no stabilizer
group and there is no code for this module to return.

This module builds one record and searches for its distance. It does not decode,
protect a logical qubit, state a threshold, or choose a noise model.
"""

from __future__ import annotations

import itertools
from numbers import Integral

from .codes import CssCode
from .gf2 import nullspace
from .qldpc import _logical_basis, _paired_basis, _rows

__all__ = ("reed_muller_code",)

#: The least cube width whose punctured pair states a stabilizer code.
_SMALLEST_VARIABLES = 4

#: The largest cube width whose distance search this route pays for.
_LARGEST_VARIABLES = 5

#: The least bound that reaches either family's lightest logical operator.
_SEARCH_BOUND = 7


def _monomial_rows(degree: int, variables: int) -> list[int]:
    """Return the cube's monomials of degree at most ``degree``, as bit masks.

    The points of the cube are the integers ``0`` up to ``2**variables - 1``, read
    as the coordinates they name, and a monomial is the set of points at which every
    variable it names is one. Bit ``point`` of a mask is that monomial's value at
    that point, so a mask is one row of the block's generator matrix and a linear
    combination of rows is symmetric difference, which is what the callers below do
    with them.
    """

    rows: list[int] = []
    for size in range(degree + 1):
        for support in itertools.combinations(range(variables), size):
            mask = 0
            for point in range(1 << variables):
                if all(point >> bit & 1 for bit in support):
                    mask |= 1 << point
            rows.append(mask)
    return rows


def _punctured(rows: list[int], width: int) -> list[int]:
    """Return the rows with the point naming every variable dropped.

    The point dropped is ``width`` itself, which is the cube point at which every
    variable is one, and the points below it keep their order because it is the last
    of them: a mask built by :func:`_monomial_rows` carries no bit above ``width``,
    so the puncture is a mask and nothing moves down. A reader who expects the
    general "drop a column and close the gap" spelling will expect a shift here;
    there is none because there is nothing above the dropped point to move.

    **This helper states the puncture rather than depending on one being implied.**
    Every caller hands its answer to :func:`~flagquantum.qec.gf2.nullspace`, whose
    contract is that a row is read within the ``width`` it is given, so the same mask
    is applied again one call deeper and this one changes nothing today: mutating it
    to return its argument unchanged was measured to leave the whole suite green, and
    the audit records that as an unobservable change. It stays because the puncture
    is the family's name -- ``reed_muller_code`` returns the *punctured* quantum
    Reed-Muller code -- and because a reader should not have to know that a shared
    helper masks to learn where the point goes. The block this route states is then
    the punctured block on its own terms, correct whether or not anything masks it
    again.
    """

    return [row & ((1 << width) - 1) for row in rows]


def reed_muller_code(m: int, *, distance_search_weight: int = _SEARCH_BOUND) -> CssCode:
    """Return the punctured quantum Reed-Muller code of the requested cube width.

    The code is stated on ``2**m - 1`` data qubits, one per point of the cube except
    the point naming every variable, and its two check families are the duals of the
    punctured blocks ``RM(m - 2, m)`` and ``RM(m - 3, m)``. The higher-degree block's
    dual states the X-type checks and the lower-degree block's dual states the Z-type
    ones, which is the convention the module docstring measures; the two logical
    families are then derived from those matrices rather than declared, by the same
    quotient :func:`~flagquantum.qec.qldpc.qldpc_code` computes.

    Args:
        m: The cube width, an integer of four or five.
        distance_search_weight: How far the distance search may look. It defaults to
            the least bound this family needs, because the lighter family's least
            logical operator has weight seven and a smaller bound refuses the record
            instead of reporting the code's distance.

    Returns:
        The code record, with the two family distances seven and three and with the
        code's distance the smaller of them.

    Raises:
        TypeError: If ``m`` is not an integer.
        ValueError: If ``m`` is below four, where the punctured pair states no code,
            or above five, where this route's distance search is no longer paid for.
            A ``distance_search_weight`` the search cannot reach is refused by
            :class:`~flagquantum.qec.codes.CssCode` when the record is built.

    Examples:
        >>> code = reed_muller_code(4)
        >>> code.num_data_qubits, code.num_ancilla_qubits, code.distance
        (15, 14, 3)
        >>> code.x_distance, code.z_distance
        (7, 3)
    """

    if isinstance(m, bool) or not isinstance(m, Integral):
        raise TypeError("Reed-Muller cube width must be an integer")
    variables = int(m)
    if variables < _SMALLEST_VARIABLES:
        raise ValueError(
            f"a punctured Reed-Muller pair on a cube of width {variables} states no "
            "code: the two blocks are one and zero degrees apart, so their duals "
            "carry more rows than the cube has points and the two check families "
            "cannot commute -- at width three, ten of the eighteen pairs of rows "
            "anticommute -- and this rule's domain begins at width four, whose "
            "fifteen points are the smallest instance"
        )
    if variables > _LARGEST_VARIABLES:
        raise ValueError(
            f"this route does not build the punctured Reed-Muller code of width "
            f"{variables}: a record's distance is computed rather than declared, and "
            "the lighter family's least logical operator has weight seven, so the "
            "search has to exhaust every subset of weight up to six -- 942648 "
            "subsets over the thirty-one points of width five, 75611760 over the "
            "sixty-three of width six, and 5434287328 over the hundred and "
            "twenty-seven of width seven -- and only the two smallest widths are "
            "paid for by this route"
        )

    width = (1 << variables) - 1
    higher = _punctured(_monomial_rows(variables - 2, variables), width)
    lower = _punctured(_monomial_rows(variables - 3, variables), width)
    x_checks = nullspace(higher, width)
    z_checks = nullspace(lower, width)
    z_logicals = _logical_basis(x_checks, z_checks, width)
    x_logicals = _logical_basis(z_checks, x_checks, width)
    z_logicals, x_logicals = _paired_basis(z_logicals, x_logicals)
    return CssCode(
        hz=_rows(z_checks, width),
        hx=_rows(x_checks, width),
        lz=_rows(z_logicals, width),
        lx=_rows(x_logicals, width),
        distance_search_weight=int(distance_search_weight),
    )
