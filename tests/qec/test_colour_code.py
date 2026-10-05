"""Unit coverage for the triangular colour code as a record.

``test_css_code.py`` pins the algebra of a code a caller writes down as
matrices, and ``test_steane_code.py`` pins one declared family. This file pins
the family the two routes meet on: a colour code, whose matrices are derived from
a triangular patch rather than written by the caller and which is not the code
any of the declared families is.

The family is worth a file of its own because three of its properties are absent
from every other record this package declares. It is **self-dual**, so one matrix
carries both check families. Its checks are **not all the same weight**: a face
in the bulk of the patch has six vertices and a face on the triangle's edge has
four, so a single data fault can light more detectors than any edge can join.
And it is the only family here whose **lattice is derived rather than tabulated**,
which is what lets the number of data qubits, the number of faces and the length
of the shortest logical string be checked against closed forms instead of against
a table that was copied from the same place as the code.

What this file proves
---------------------

1. ``test_the_patch_carries_the_published_number_of_data_qubits_and_faces``: the
   record's data-qubit count is ``(3*d**2 + 1) / 4`` and its face count is
   ``3*(d**2 - 1) / 8``, at three distances. Both are closed forms the lattice
   derivation does not use, and their difference leaves exactly one logical qubit.
2. ``test_the_two_check_families_are_one_matrix``: the Z-type and X-type checks
   have the same supports in the same order, every support has weight four or
   six, and the bulk weight appears as soon as the patch has a face away from its
   edge. This is the self-duality the family is named for, stated as a property of
   the record rather than of the lattice it came from.
3. ``test_the_face_incidence_graph_admits_one_three_colouring``: reading only the
   record's check supports, every data qubit lies in at least one check, exactly
   three lie in one, no qubit lies in more than three, and the graph joining two
   faces that share a qubit has exactly six proper three-colourings -- one
   colouring and its relabellings -- each splitting the faces into equal classes.
   Three-colourability is the defining property of the family, and it is the one
   the parity-check matrices do not state.
4. ``test_every_single_fault_has_its_own_syndrome_at_distance_three``: the three
   checks of the distance-three patch map its seven data qubits onto the seven
   non-zero three-bit syndromes, each exactly once. That is the perfect-code
   property, and it is asserted as a set identity against the syndrome space
   rather than as a recorded spectrum.
5. ``test_the_record_proves_the_distance_it_is_asked_for``: the reported
   ``distance``, ``x_distance`` and ``z_distance`` all equal the requested
   distance, and the shortest logical string has that many vertices.
6. ``test_a_heavier_logical_representative_does_not_raise_the_distance``: the
   same patch with its declared logical operator multiplied by one of its own
   checks -- a representative of weight eleven instead of five -- still reports
   five. Without this the distance could be read off the operator the caller
   wrote down rather than searched for.
7. ``test_the_family_refuses_a_distance_it_cannot_build``: an even distance and a
   distance below three are refused for two different reasons, and the reason is
   checkable in the message.
8. ``test_the_smallest_patch_is_the_steane_code_up_to_a_relabelling``: the
   distance-three patch has seven data vertices over three faces, and its check
   set is the Steane code's check set under a permutation of the data wires, which
   the test exhibits rather than asserts the existence of. This is the fact a
   reader is most likely to get wrong, so it is stated as a candidate-set identity
   instead of left implicit: the patch is a second description of one code at
   distance three and not a second seven-qubit code.
9. ``test_the_public_namespace_publishes_the_family``: ``flagquantum.qec``
   resolves the family to the same callable and its ``__all__`` names it, so the
   record is part of the surface a caller reads rather than only an attribute
   that happens to be bound.

What this file does not prove
-----------------------------

It makes no decoding, noise, threshold or logical-suppression claim, and it
demonstrates nothing about transversal gates or magic-state cultivation, which
are what colour codes are used for and which this package does not compute. It
also does not compare the derived lattice against another framework's patch
vertex by vertex: the closed forms in point 1 are what stand in for that, and
they pin the size and the check count rather than the numbering.

The distance at seven is the largest this file builds, and building it is the
whole cost of the family: the record searches wire subsets up to the requested
distance, which measured 3.3 s at distance seven and 0.01 s at five. Distances
three and five carry every claim that is not about the closed forms.
"""

from __future__ import annotations

import itertools
from collections import Counter
from functools import lru_cache

import pytest

from flagquantum.qec import (
    CssCode,
    SteaneCode,
    css_code_matrices,
    triangular_colour_code,
)

pytestmark = pytest.mark.unit

# The distances this file builds. Seven is included so the closed forms are not
# checked at the two smallest patches only, and it is the last one the record's
# distance search reaches in a useful time.
_DISTANCES = (3, 5, 7)


@lru_cache(maxsize=None)
def _colour(distance: int) -> CssCode:
    """Return the patch once per distance, because the distance search is the cost."""

    return triangular_colour_code(distance)


def _z_supports(code: CssCode) -> tuple[tuple[int, ...], ...]:
    """Return the supports of the Z-type checks, in the record's own order."""

    return tuple(
        tuple(check.stabilizer.z_wires)
        for check in code.checks
        if check.stabilizer.z_wires
    )


def _x_supports(code: CssCode) -> tuple[tuple[int, ...], ...]:
    """Return the supports of the X-type checks, in the record's own order."""

    return tuple(
        tuple(check.stabilizer.x_wires)
        for check in code.checks
        if check.stabilizer.x_wires
    )


def _proper_three_colourings(
    supports: tuple[tuple[int, ...], ...],
) -> list[tuple[int, ...]]:
    """Return every colouring of the faces in which neighbours differ.

    Two faces are neighbours when they share a data qubit. The search is exact
    rather than greedy, so a patch the greedy order happens to colour is not
    mistaken for a patch that has a colouring at all.
    """

    neighbours = {
        index: {
            other
            for other in range(len(supports))
            if other != index and set(supports[index]) & set(supports[other])
        }
        for index in range(len(supports))
    }
    found: list[tuple[int, ...]] = []
    colour: dict[int, int] = {}

    def extend(index: int) -> None:
        if index == len(supports):
            found.append(tuple(colour[face] for face in range(len(supports))))
            return
        taken = {colour[other] for other in neighbours[index] if other in colour}
        for candidate in range(3):
            if candidate not in taken:
                colour[index] = candidate
                extend(index + 1)
                del colour[index]

    extend(0)
    return found


@pytest.mark.parametrize("distance", _DISTANCES)
def test_the_patch_carries_the_published_number_of_data_qubits_and_faces(
    distance: int,
) -> None:
    """Check the sizes against closed forms the lattice derivation never uses.

    ``(3*d**2 + 1) / 4`` and ``3*(d**2 - 1) / 8`` are the published counts for this
    patch. Both are arithmetic on the distance, while the record's counts come
    from enumerating and classifying lattice points, so agreement is evidence
    about the enumeration rather than one formula checked against itself. The
    third assertion is the logical-qubit count: the two check families span all
    but one of the data qubits.
    """

    code = _colour(distance)
    faces = _z_supports(code)

    assert code.num_data_qubits == (3 * distance * distance + 1) // 4
    assert len(faces) == 3 * (distance * distance - 1) // 8
    assert code.num_data_qubits - 2 * len(faces) == 1
    assert code.num_ancilla_qubits == 2 * len(faces)
    # The declared logical operator is the triangle's shortest side, so its
    # support has one vertex per unit of distance.
    assert {len(observable.support) for observable in code.logical_observables} == {
        distance
    }


@pytest.mark.parametrize("distance", _DISTANCES)
def test_the_two_check_families_are_one_matrix(distance: int) -> None:
    """The family is self-dual: one set of faces carries both check families.

    Equality is asserted on the supports in order, not on the counts, because two
    families of equal size could still be different matrices. The weight spectrum
    is the other half: a face away from the triangle's edge touches six qubits and
    a face on the edge touches four, so the bulk weight is absent from the
    distance-three patch, which has no interior face, and present from five up.
    """

    code = _colour(distance)
    weights = Counter(len(support) for support in _z_supports(code))

    assert _z_supports(code) == _x_supports(code)
    assert set(weights) <= {4, 6}
    assert weights[4] > 0
    assert weights[6] > 0 if distance > 3 else weights[6] == 0


@pytest.mark.parametrize("distance", _DISTANCES)
def test_the_face_incidence_graph_admits_one_three_colouring(distance: int) -> None:
    """Three-colourability is the family's defining property and is not in the matrix.

    The parity-check matrices say which qubits a face touches. They do not say
    that the faces can be coloured with three colours so that neighbours differ,
    which is the property a colour code is built from and the reason transversal
    gates exist for the family. The incidence counts are asserted first, because a
    patch with a hanging qubit or a four-valent vertex would still admit a
    colouring and would not be this lattice.
    """

    code = _colour(distance)
    faces = _z_supports(code)
    incidence = Counter()
    for support in faces:
        for wire in support:
            incidence[wire] += 1

    assert set(incidence) == set(code.data_wires)
    assert max(incidence.values()) == 3
    # The three corners of the triangle are the only once-checked qubits.
    assert sum(1 for count in incidence.values() if count == 1) == 3

    colourings = _proper_three_colourings(faces)
    per_class = (distance * distance - 1) // 8

    # One colouring and its six relabellings, and every one of them splits the
    # faces equally, so the equal split is a property of the graph rather than of
    # the colouring the search happened to find first.
    assert len(colourings) == 6
    assert {tuple(sorted(Counter(colouring).values())) for colouring in colourings} == {
        (per_class, per_class, per_class)
    }


def test_every_single_fault_has_its_own_syndrome_at_distance_three() -> None:
    """The distance-three patch is perfect, so its syndromes are the whole space.

    Three checks leave seven non-zero syndromes and the patch has seven data
    qubits, so a single fault is identifiable exactly when the seven syndromes are
    those seven vectors. Asserting the set identity says that; asserting a weight
    spectrum would say the same thing in the echo of the formula that produced it.
    """

    code = _colour(3)
    faces = _z_supports(code)
    syndromes = {
        wire: tuple(1 if wire in support else 0 for support in faces)
        for wire in code.data_wires
    }

    assert all(any(syndrome) for syndrome in syndromes.values())
    assert set(syndromes.values()) == {
        candidate
        for candidate in itertools.product((0, 1), repeat=len(faces))
        if any(candidate)
    }


@pytest.mark.parametrize("distance", (3, 5))
def test_the_record_proves_the_distance_it_is_asked_for(distance: int) -> None:
    """Both families reach the requested distance and no shorter operator exists."""

    code = _colour(distance)

    assert code.distance == distance
    assert code.x_distance == distance
    assert code.z_distance == distance


def test_a_heavier_logical_representative_does_not_raise_the_distance() -> None:
    """A logical operator multiplied by a check is still a logical operator.

    The distance-five patch declares the five-vertex side of its triangle. Here
    the same patch is written with that operator multiplied by a bulk check it
    overlaps in nothing, which is a weight-eleven representative of the same
    logical operator class. A record that read its distance off the declared
    operator would report eleven; the search reports five.
    """

    declared = triangular_colour_code(5)
    matrices = css_code_matrices(declared)
    bulk = next(
        tuple(check.stabilizer.z_wires)
        for check in declared.checks
        if len(check.stabilizer.z_wires) == 6
    )
    side = tuple(
        observable.z_wires
        for observable in declared.logical_observables
        if observable.z_wires
    )[0]
    lifted = tuple(sorted(set(side) ^ set(bulk)))

    assert len(lifted) > len(side)
    heavier = CssCode(
        hz=matrices.hz.tolist(),
        hx=matrices.hx.tolist(),
        lz=[[1 if wire in lifted else 0 for wire in range(declared.num_data_qubits)]],
        lx=matrices.lx.tolist(),
        distance_search_weight=5,
    )

    assert heavier.distance == declared.distance == 5


def test_the_family_refuses_a_distance_it_cannot_build() -> None:
    """The two physical reasons are separate and each names itself.

    An even distance has no whole side, because the side spans ``distance - 1``
    lattice periods of three rows. A distance below three has no face and
    therefore no check. Both are asserted with their own text, so collapsing the
    two refusals into one would fail here rather than pass quietly.
    """

    with pytest.raises(ValueError, match="no whole side"):
        triangular_colour_code(4)
    with pytest.raises(ValueError, match="no face"):
        triangular_colour_code(1)

    for invalid in (True, 2.0, "3", None):
        with pytest.raises(TypeError, match="must be an integer"):
            triangular_colour_code(invalid)  # type: ignore[arg-type]


def test_the_smallest_patch_is_the_steane_code_up_to_a_relabelling() -> None:
    """The seven-vertex patch is the Steane code under a permutation of its wires.

    Both records carry three weight-four checks over seven data wires, so the two
    check sets have the same weight multiset, and a search over the seven-wire
    permutations exhibits one that maps the patch's check set onto the Steane
    code's. The identity is not such a permutation, so this is a relabelling and
    not an equality of matrices: the record's own wire numbering is the lattice's,
    and a caller must not compare the two check lists position by position.

    Nothing downstream reads this equivalence -- the distance is proved from the
    patch's own matrices -- so the test exists to keep a future change to the
    lattice from silently turning distance three into a different code.
    """

    patch = _colour(3)
    steane = SteaneCode()
    target = {frozenset(check.stabilizer.z_wires) for check in steane.checks}
    source = {frozenset(check.stabilizer.z_wires) for check in patch.checks}

    assert sorted(len(support) for support in source) == sorted(
        len(support) for support in target
    )

    relabelling = next(
        (
            permutation
            for permutation in itertools.permutations(range(patch.num_data_qubits))
            if {frozenset(permutation[wire] for wire in support) for support in source}
            == target
        ),
        None,
    )

    assert relabelling is not None
    assert relabelling != tuple(range(patch.num_data_qubits))


def test_the_public_namespace_publishes_the_family() -> None:
    """The package root reaches the family, and its ``__all__`` says so.

    A name bound in ``flagquantum.qec`` is importable whether or not the package
    lists it, so an import alone does not prove the family was published. The
    record is only part of the surface a caller reads from ``flagquantum.qec``
    if ``__all__`` names it, which is the same statement ``test_codes.py`` makes
    about the other records.
    """

    import flagquantum.qec as qec

    assert hasattr(qec, "triangular_colour_code")
    assert "triangular_colour_code" in qec.__all__
    assert qec.triangular_colour_code is triangular_colour_code
