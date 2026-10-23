"""Colour code definitions."""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

from ._compat import _retain_module

if TYPE_CHECKING:
    from .css import CssCode


def triangular_colour_code(distance: int) -> CssCode:
    """Return the triangular 6.6.6 colour code of the requested distance.

    A colour code is written down on a trivalent lattice whose faces are
    three-colourable, and every face carries one X-type and one Z-type stabilizer
    over the data qubits at its vertices. That is what makes the family self-dual,
    which is why the two check blocks below are one matrix, and it is also why the
    family is not a matching code: a face in the bulk has six vertices, so one data
    fault inside it can flip more detectors than an edge can join.

    **The patch is a triangle and its side is the distance.** A point ``(row,
    column)`` with ``0 <= column <= row <= bound`` is either a data vertex or a face
    centre, and the face centres are the points where ``column % 3 == 2 - row % 3``.
    Every other point is a data vertex. A face is the six in-patch points one step
    diagonally or orthogonally away from it, so a face in the bulk has weight six
    and a face on the triangle's edge has weight four, where the boundary cuts the
    hexagon away. The declared logical operator is the data vertices on the
    ``column == 0`` side, which is the patch's shortest logical string; it commutes
    with every face and overlaps itself on an odd number of vertices, so it is a
    logical operator of both families rather than of one.

    **The requested distance is proved rather than stated.** The record handed back
    is the matrix route's :class:`~flagquantum.qec.CssCode`, so the distance it
    states is searched for over the matrices derived here: the least weight of a
    logical operator of each family is found and the smaller of the two becomes the
    record's ``distance``. The bound that search is given is the requested distance
    itself, since a smaller one would refuse the code whose distance is the number
    asked for, and that bound is the whole cost of this function: the search tries
    every qubit subset up to the distance, so the family is affordable at the small
    distances an experiment is built from and is not a way to write down a large
    patch.

    **At distance three this is the Steane code, relabelled.** The smallest patch
    has seven data vertices over three faces, and its three weight-four checks are
    the Steane code's three weight-four checks under a permutation of the data
    qubits -- the identity is not that permutation, which
    ``tests/qec/test_colour_code.py`` exhibits rather than asserts. The two records
    are therefore one code in two descriptions at distance three rather than two
    seven-qubit codes, and their qubit numbers are not comparable position by
    position. Nothing here reads that equivalence, because the distance is proved
    from this patch's own matrices, and the family grows away from it: at distance
    five the patch has nineteen data vertices and checks of weight four and six,
    which no Steane-style description reaches.

    Args:
        distance: The code distance, an odd integer of at least three.

    Returns:
        The code record, with one logical qubit and both family distances equal to
        ``distance``.

    Raises:
        TypeError: If ``distance`` is not an integer.
        ValueError: If ``distance`` is even or below three, or if no logical
            operator of a family reaches the requested weight.

    Examples:
        >>> colour = triangular_colour_code(3)
        >>> colour.num_data_qubits, colour.num_ancilla_qubits, colour.distance
        (7, 6, 3)
        >>> colour.stabilizers[0] == Pauli(z_qubits=(0, 1, 2, 3))
        True
    """

    if isinstance(distance, bool) or not isinstance(distance, Integral):
        raise TypeError("colour-code distance must be an integer")
    if distance % 2 == 0:
        raise ValueError(
            f"the triangular colour patch of distance {distance} has no whole side: "
            "the side spans (distance - 1) lattice periods of three rows, so the "
            "distance must be odd"
        )
    if distance < 3:
        raise ValueError(
            f"the triangular colour patch of distance {distance} has no face and "
            "therefore no check, so it is not a code"
        )

    bound = 3 * (distance - 1) // 2
    points = tuple(
        (row, column) for row in range(bound + 1) for column in range(row + 1)
    )
    # A face centre is the point whose column sits two places back from its row in
    # the three-colouring, in the row coordinates the patch is written in.
    faces = tuple(point for point in points if point[1] % 3 == 2 - point[0] % 3)
    vertices = tuple(point for point in points if point[1] % 3 != 2 - point[0] % 3)
    wire_of = {vertex: index for index, vertex in enumerate(vertices)}

    offsets = ((-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1))
    supports = tuple(
        tuple(
            sorted(
                wire_of[(row + down, column + across)]
                for down, across in offsets
                if (row + down, column + across) in wire_of
            )
        )
        for row, column in faces
    )
    side = tuple(sorted(wire_of[vertex] for vertex in vertices if vertex[1] == 0))

    width = len(vertices)
    checks = tuple(
        tuple(1 if wire in support else 0 for wire in range(width))
        for support in supports
    )
    logical = (tuple(1 if wire in side else 0 for wire in range(width)),)
    # The record is built on this module's protocol, so it is imported here rather
    # than at module level, where it would close a cycle between the two modules.
    from .css import CssCode

    return CssCode(
        hz=checks,
        hx=checks,
        lz=logical,
        lx=logical,
        distance_search_weight=int(distance),
    )


_retain_module("flagquantum.qec.codes", triangular_colour_code)
