"""Toric code definitions."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

from ._compat import _retain_module

if TYPE_CHECKING:
    from .css import CssCode


def toric_code(linear_size: int) -> CssCode:
    """Return the square-lattice toric code of the requested linear size.

    A toric code is written down on a torus rather than on a patch. The lattice is
    a ``linear_size``-by-``linear_size`` grid whose opposite sides are identified,
    every vertex carries one X-type stabilizer over the four edges meeting it, and
    every face carries one Z-type stabilizer over the four edges around it. The
    identification is what the family is: a periodic lattice has no boundary, so
    no face is cut down by one and every check has weight four at every size.

    **The patch is a torus, so the code carries two logical qubits rather than
    one.** A logical operator is a cycle of edges that wraps one of the two
    directions without bounding a face, and each direction carries one of each
    family. The Z-type family below is therefore two operators -- the ring of
    horizontal edges in row zero and the ring of vertical edges in column zero --
    and the X-type family is the matching pair of cuts, which cross the first ring
    and the second ring respectively. This is the first record in this package
    whose matrices leave ``k = 2``: ``css_code_matrices`` reports four logical
    operators for it, and a memory circuit built from it declares two observables.
    ``hz`` and ``hx`` are not one matrix here, because a star spans the edges at a
    vertex and a face spans the edges around a face; the two families are
    transposes of each other in the lattice sense and are computed from the same
    edge numbering rather than kept as two tables in step.

    **The linear size is the distance, and the distance is proved rather than
    stated.** The record handed back is the matrix route's
    :class:`~flagquantum.qec.CssCode`, so the distance it states is searched for
    over the matrices derived here: the least weight of a logical operator of each
    family is found -- the two rings and the two cuts, each of weight
    ``linear_size`` -- and the smaller of the two becomes the record's ``distance``.
    The search is given the linear size as its bound. That bound is the whole cost
    of this function, because the search tries every qubit subset up to the bound:
    measured, the record builds in 0.00 s at size two, three and four,
    0.34 s at five and 23.1 s at six, on 2 * size ** 2 data qubits. It is
    therefore a way to write down the small tori an experiment is built from and
    not a way to write down a large one, and a caller who wants a larger lattice
    wants a decoder that reads its distance from the lattice instead -- which is
    the work of its own row rather than this route's.

    Args:
        linear_size: The number of vertices along one side of the torus, an
            integer of at least two. The patch has ``2 * linear_size ** 2`` data
            qubits, ``2 * linear_size ** 2`` checks and two logical qubits.

    Returns:
        The code record, with two logical qubits and both family distances equal
        to ``linear_size``.

    Raises:
        TypeError: If ``linear_size`` is not an integer.
        ValueError: If ``linear_size`` is below two, or if no logical operator of
            a family reaches the requested weight.

    Examples:
        >>> torus = toric_code(2)
        >>> torus.num_data_qubits, torus.num_ancilla_qubits, torus.distance
        (8, 8, 2)
        >>> torus.stabilizers[0] == Pauli(z_qubits=(0, 1, 4, 6))
        True
    """

    if isinstance(linear_size, bool) or not isinstance(linear_size, Integral):
        raise TypeError("toric linear size must be an integer")
    if linear_size < 2:
        raise ValueError(
            f"the torus of linear size {linear_size} has no face and therefore no "
            "check, so it is not a code"
        )
    size = int(linear_size)
    width = 2 * size * size

    def edge(side: int, row: int, column: int) -> int:
        """The qubit of one edge, with both torus coordinates wrapped."""

        return side * size * size + (row % size) + size * (column % size)

    def matrix_row(support: frozenset[int]) -> tuple[int, ...]:
        return tuple(1 if wire in support else 0 for wire in range(width))

    # A star is the four edges at a vertex: the two horizontal edges to its right
    # and the two vertical edges below it. A face is the four edges around a face:
    # the two horizontal edges of its top row and the two vertical edges of its
    # left column. Both are read off the one edge numbering above, so wrapping is
    # the modulus in that function and not a special case here.
    stars = tuple(
        matrix_row(
            frozenset(
                (
                    edge(0, row, column),
                    edge(0, row, column + 1),
                    edge(1, row, column),
                    edge(1, row + 1, column),
                )
            )
        )
        for row in range(size)
        for column in range(size)
    )
    faces = tuple(
        matrix_row(
            frozenset(
                (
                    edge(0, row - 1, column),
                    edge(0, row, column),
                    edge(1, row, column - 1),
                    edge(1, row, column),
                )
            )
        )
        for row in range(size)
        for column in range(size)
    )
    row_ring = frozenset(edge(0, 0, column) for column in range(size))
    column_ring = frozenset(edge(1, row, 0) for row in range(size))
    column_cut = frozenset(edge(0, row, 0) for row in range(size))
    row_cut = frozenset(edge(1, 0, column) for column in range(size))
    # The record is built on this module's protocol, so it is imported here rather
    # than at module level, where it would close a cycle between the two modules.
    from .css import CssCode

    return CssCode(
        hz=faces,
        hx=stars,
        lz=(matrix_row(row_ring), matrix_row(column_ring)),
        lx=(matrix_row(column_cut), matrix_row(row_cut)),
        distance_search_weight=size,
    )


_retain_module("flagquantum.qec.codes", toric_code)
