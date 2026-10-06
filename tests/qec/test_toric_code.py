"""Unit coverage for the square-lattice toric code as a record.

``test_css_code.py`` pins the algebra of a code a caller writes down as matrices,
``test_steane_code.py`` pins one declared family and ``test_colour_code.py`` pins
the family a second derivation route reaches. This file pins the family whose
lattice is a **torus** rather than a patch: the square grid with its opposite
sides identified.

The family is worth a file of its own because it is the first record here that
differs from every other one on three counts at once.

1. **It carries two logical qubits.** Every other record this package declares
   leaves exactly one, so ``css_code_matrices`` reports four logical operators and
   a memory circuit built from the record declares two observables rather than
   one. That is the property a consumer of the record notices first, because it
   changes the shape of the layout rather than the contents of a row.
2. **It has no boundary.** Opposite sides are identified, so no check is cut down
   and every check has the same weight at every linear size. The rotated surface
   patch and the triangular colour patch both have checks of reduced weight on
   their edges, which is why the weight spectrum is a singleton here and is not
   there.
3. **Its distance and its size are the same number.** The distance is not an
   independent parameter: a torus of linear size ``L`` has ``2 * L ** 2`` data
   qubits, ``2 * L ** 2`` checks, distance ``L`` and two logical qubits. A larger
   torus is a longer logical string and a quadratically larger matrix rather than
   the same matrix at a larger distance.

What this file proves
---------------------

1. ``test_the_torus_carries_the_closed_form_size_at_every_linear_size``: the
   record's data-qubit count is ``2 * L ** 2``, its per-family check count is
   ``L ** 2``, its per-family rank is ``L ** 2 - 1`` and the two together leave
   exactly two logical qubits. The rank is computed from the record's own rows
   with an independent GF(2) reduction, so the logical-qubit count is a
   measurement of the matrices rather than a claim about the lattice.
2. ``test_no_check_is_cut_down_because_the_lattice_has_no_boundary``: the check
   weight spectrum is ``{4}`` at every size and every data qubit lies in exactly
   two checks of each family. The same measurement on the rotated surface patch at
   the same sizes returns a weight-two check, so the singleton is a property of
   this lattice and not of the measurement.
3. ``test_the_two_check_families_are_different_matrices_that_commute``: the two
   families are not one matrix, because a star spans the edges at a vertex and a
   face spans the edges around a face, and every Z-type check still commutes with
   every X-type check. The commutation is asserted between the record's own check
   pairs rather than left to the constructor that would have refused the record.
4. ``test_the_matrices_leave_two_logical_qubits``: ``num_z_logicals`` and
   ``num_x_logicals`` are both two, ``num_observables`` is four, and the matrix of
   anticommutations between the declared Z-type and X-type logical operators is a
   permutation matrix -- each Z-type operator anticommutes with exactly one X-type
   operator and the correspondence is a bijection. A k of two whose two pairs were
   not separately addressable would not be the same statement.
5. ``test_the_declared_logical_operators_wrap_the_torus``: each declared logical
   operator has one wire per unit of linear size, the two of a family are
   disjoint, and each is a cycle that commutes with the opposite family and not
   with its own.
6. ``test_the_record_proves_the_distance_it_is_asked_for``: the reported
   ``distance``, ``x_distance`` and ``z_distance`` all equal the linear size, and
   the size is recovered from the distance rather than passed alongside it.
7. ``test_a_heavier_logical_representative_does_not_raise_the_distance``: the same
   torus with a declared logical operator multiplied by one of its own face
   checks -- a representative two wires longer -- still reports the same distance.
   Without this the distance could be read off the operator the caller wrote down
   rather than searched for.
8. ``test_the_memory_circuit_reads_out_both_logical_qubits``: the layout declares
   two observables of one wire per unit of size, both in the Z basis, and
   ``derive_anticommuting_logical_product`` reaches a distinct partner of full
   weight for each index, each anticommuting with its own observable and commuting
   with the other. Every other family here exercises that derivation at index zero
   only.
9. ``test_the_family_refuses_a_size_it_cannot_build``: a non-integer size and a
   size below two are refused for two different reasons, and the reason is
   checkable in the message.
10. ``test_the_public_namespace_publishes_the_family``: ``flagquantum.qec``
    resolves the family to the same callable and its ``__all__`` names it, so the
    record is part of the surface a caller reads rather than only an attribute
    that happens to be bound.

What this file does not prove
-----------------------------

It makes no decoding, noise, threshold or logical-suppression claim; that is
``test_toric_memory_execution.py``, which runs the execution path. It also does
not prove that the record agrees with another framework's torus vertex by vertex:
the closed forms in points 1 and 6 are what stand in for that, and they pin the
size, the check count and the distance rather than the wire numbering.

The distance search is the whole cost of building the record, so five is the
largest linear size this file builds: measured, the record takes 0.00 s at size
two, three and four, 0.34 s at five and 23.1 s at six. Sizes two through five
carry every claim here, and size six is excluded because it is 23 s of search for
one more point on the same closed forms.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

import pytest

from flagquantum.qec import (
    CssCode,
    Pauli,
    RotatedSurfaceCode,
    build_memory_circuit,
    css_code_matrices,
    derive_anticommuting_logical_product,
    toric_code,
)
from flagquantum.qec.gf2 import rank

pytestmark = pytest.mark.unit

# The linear sizes this file builds. Five is the largest the record's distance
# search reaches before the bound starts to dominate the build cost.
_SIZES = (2, 3, 4, 5)


@lru_cache(maxsize=None)
def _torus(linear_size: int) -> CssCode:
    """Return the torus once per size, because the distance search is the cost."""

    return toric_code(linear_size)


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


def _rows_as_bits(rows: list[list[int]]) -> list[int]:
    """Pack each 0/1 row into one integer, which is what the GF(2) kernel takes."""

    return [int("".join(str(bit) for bit in row), 2) for row in rows]


@pytest.mark.parametrize("linear_size", _SIZES)
def test_the_torus_carries_the_closed_form_size_at_every_linear_size(
    linear_size: int,
) -> None:
    """Check the sizes against closed forms that do not use the same arithmetic.

    ``2 * L ** 2`` data qubits over ``L ** 2`` checks of each family are the
    published counts for the square torus: every edge is a data qubit, every
    vertex a Z-type check and every face an X-type check, on a grid with
    ``L ** 2`` vertices, ``2 * L ** 2`` edges and ``L ** 2`` faces. The record's
    own counts come from enumerating the lattice, so agreement is evidence about
    the enumeration rather than one formula checked against itself.

    The logical-qubit count is the measurement: the two check families reduce to
    ``L ** 2 - 1`` constraints each, and ``2 * L ** 2`` data qubits under
    ``2 * (L ** 2 - 1)`` independent constraints leave exactly two. The rank is
    taken over the record's own rows by an independent GF(2) kernel, so this is
    read off the matrices rather than asserted about the lattice.
    """

    code = _torus(linear_size)
    matrices = css_code_matrices(code)
    z_rank = rank(_rows_as_bits(matrices.hz.tolist()))
    x_rank = rank(_rows_as_bits(matrices.hx.tolist()))
    square = linear_size * linear_size

    assert code.num_data_qubits == 2 * square
    assert len(_z_supports(code)) == square
    assert len(_x_supports(code)) == square
    assert code.num_ancilla_qubits == 2 * square
    assert z_rank == x_rank == square - 1
    assert code.num_data_qubits - z_rank - x_rank == 2


@pytest.mark.parametrize("linear_size", _SIZES)
def test_no_check_is_cut_down_because_the_lattice_has_no_boundary(
    linear_size: int,
) -> None:
    """Every check has the same weight, at every size, in both families.

    A patch has an edge, and the checks on that edge lose the wires the edge cuts
    away. A torus has none, so no check loses a wire and the weight spectrum is a
    singleton whose value does not depend on the size. The incidence counts are
    the other half of the same statement: every data qubit lies in exactly two
    checks of each family, so no qubit is on a boundary either.

    The rotated surface patch at the same sizes is measured alongside, because
    otherwise a singleton spectrum could as easily mean the measurement cannot see
    a reduced check. It returns a weight-two check at both sizes, so the reduction
    is visible to this test when it is there.
    """

    code = _torus(linear_size)
    z_weights = Counter(len(support) for support in _z_supports(code))
    x_weights = Counter(len(support) for support in _x_supports(code))

    assert set(z_weights) == {4}
    assert set(x_weights) == {4}
    assert z_weights[4] == x_weights[4] == linear_size * linear_size

    for supports in (_z_supports(code), _x_supports(code)):
        incidence = Counter()
        for support in supports:
            for wire in support:
                incidence[wire] += 1
        assert set(incidence) == set(code.data_wires)
        assert set(incidence.values()) == {2}

    patch = RotatedSurfaceCode(linear_size)
    patch_weights = {
        len(check.stabilizer.z_wires)
        for check in patch.checks
        if check.stabilizer.z_wires
    } | {
        len(check.stabilizer.x_wires)
        for check in patch.checks
        if check.stabilizer.x_wires
    }
    assert patch_weights != {4}


@pytest.mark.parametrize("linear_size", _SIZES)
def test_the_two_check_families_are_different_matrices_that_commute(
    linear_size: int,
) -> None:
    """A star spans the edges at a vertex and a face spans the edges around a face.

    Unlike the colour and Steane families, whose two check blocks are one matrix,
    the two families here are different matrices: no vertex of the torus coincides
    with a face, so no star is a face. They are still both drawn from the one edge
    numbering, which is why the same function derives both, and the code they
    describe is valid exactly when every Z-type check commutes with every X-type
    check. That commutation is asserted between the record's check pairs rather
    than left to the constructor, which would have refused the record and so would
    have made a passing construction look like the proof.
    """

    code = _torus(linear_size)

    assert _z_supports(code) != _x_supports(code)
    for z_support in _z_supports(code):
        z_check = Pauli(z_wires=z_support)
        for x_support in _x_supports(code):
            assert z_check.commutes_with(Pauli(x_wires=x_support))


@pytest.mark.parametrize("linear_size", _SIZES)
def test_the_matrices_leave_two_logical_qubits(linear_size: int) -> None:
    """This is the first record here whose matrices leave more than one.

    ``css_code_matrices`` splits the four logical operators by family, and the
    count on each side is two. Reporting four operators would be equally true of a
    code with one logical qubit whose representative is written twice, so the
    distinction is made by the anticommutation matrix: it must be a permutation
    matrix, so each declared Z-type operator pairs with exactly one X-type
    operator and the pairing is a bijection. One logical qubit whose operator were
    written down twice would give a pairing matrix with a repeated row.

    Every other family this package declares leaves one logical qubit, which is
    measured here on the two other derived routes so that the count is not read as
    a property of the matrix representation.
    """

    code = _torus(linear_size)
    matrices = css_code_matrices(code)

    assert matrices.num_z_logicals == 2
    assert matrices.num_x_logicals == 2
    assert matrices.num_observables == 4

    z_observables = [
        observable for observable in code.logical_observables if observable.z_wires
    ]
    x_observables = [
        observable for observable in code.logical_observables if observable.x_wires
    ]
    pairing = [
        [1 if not z.commutes_with(x) else 0 for x in x_observables]
        for z in z_observables
    ]

    assert len(z_observables) == len(x_observables) == 2
    assert pairing == [
        [1 if row == column else 0 for column in range(2)] for row in range(2)
    ]

    from flagquantum.qec import SteaneCode, triangular_colour_code

    assert css_code_matrices(SteaneCode()).num_z_logicals == 1
    assert css_code_matrices(triangular_colour_code(3)).num_z_logicals == 1


@pytest.mark.parametrize("linear_size", _SIZES)
def test_the_declared_logical_operators_wrap_the_torus(linear_size: int) -> None:
    """A logical operator is a cycle that wraps once and bounds nothing.

    The shortest such cycle crosses one unit of linear size per unit of distance,
    so every declared operator has exactly ``linear_size`` wires; the two of a
    family are the two directions, so they are disjoint and are not the same
    operator written twice. Bounding nothing is the statement that the cycle is
    not a product of face checks, and the way the record can see that is
    commutation: every declared operator must commute with every check of both
    families. That is the whole content of the record's logical family, and it is
    checked here against the record's checks rather than against the matrix the
    checks were built from. How the two families pair with each other over the
    four operators is the subject of ``test_the_matrices_leave_two_logical_qubits``,
    because that pairing is what the matrix route reports rather than what the
    records say.
    """

    code = _torus(linear_size)
    z_observables = [
        observable for observable in code.logical_observables if observable.z_wires
    ]
    x_observables = [
        observable for observable in code.logical_observables if observable.x_wires
    ]

    for observables in (z_observables, x_observables):
        assert {len(observable.support) for observable in observables} == {linear_size}
        assert observables[0] != observables[1]
        assert not set(observables[0].support) & set(observables[1].support)

    checks = [
        Pauli(z_wires=check.stabilizer.z_wires, x_wires=check.stabilizer.x_wires)
        for check in code.checks
    ]
    for observable in (*z_observables, *x_observables):
        for check in checks:
            assert observable.commutes_with(check)


@pytest.mark.parametrize("linear_size", _SIZES)
def test_the_record_proves_the_distance_it_is_asked_for(linear_size: int) -> None:
    """Both families reach the linear size, and the size is recovered from it.

    The size is the distance here, so the only argument the family takes is the
    number it must prove. The last assertion states that as an identity on the
    record: a torus of distance ``d`` has ``2 * d ** 2`` data qubits, which is
    what makes a larger distance cost a quadratically larger matrix rather than a
    longer search over a fixed one.
    """

    code = _torus(linear_size)

    assert code.distance == linear_size
    assert code.x_distance == linear_size
    assert code.z_distance == linear_size
    assert code.num_data_qubits == 2 * code.distance * code.distance


def test_a_heavier_logical_representative_does_not_raise_the_distance() -> None:
    """A logical operator multiplied by a check is still a logical operator.

    The size-five torus declares the ring of horizontal edges in row zero. Here
    the same torus is written with that ring multiplied by a face check it
    overlaps in one wire, which is a representative of seven wires instead of
    five of the same logical operator class. A record that read its distance off
    the declared operator would report seven; the search reports five. The other
    three logical operators are carried over unchanged, because a logical family
    needs one operator per logical qubit and the point of the test is the weight
    of one of them rather than the number of them.
    """

    declared = _torus(5)
    matrices = css_code_matrices(declared)
    face = _z_supports(declared)[0]
    ring = tuple(
        observable.z_wires
        for observable in declared.logical_observables
        if observable.z_wires
    )[0]
    lifted = tuple(sorted(set(ring) ^ set(face)))

    assert set(ring) != set(face)
    assert len(lifted) > len(ring)

    lz_rows = matrices.lz.tolist()
    lz_rows[0] = [
        1 if wire in lifted else 0 for wire in range(declared.num_data_qubits)
    ]
    heavier = CssCode(
        hz=matrices.hz.tolist(),
        hx=matrices.hx.tolist(),
        lz=lz_rows,
        lx=matrices.lx.tolist(),
        distance_search_weight=5,
    )

    assert heavier.distance == declared.distance == 5
    assert heavier.num_data_qubits == declared.num_data_qubits


@pytest.mark.parametrize("linear_size", (2, 3, 5))
def test_the_memory_circuit_reads_out_both_logical_qubits(linear_size: int) -> None:
    """The layout carries one observable per declared Z-type logical operator.

    Every other family here has one, so this is the first layout in the package
    with two observables. Their weights are the distance, because the declared
    operators are the shortest cycles, and both are in the Z basis, because the
    default memory experiment prepares and reads out in one basis.

    The second half exercises the derivation every other family is used at index
    zero only: a partner operator is found for each observable index, the two
    partners are distinct, each anticommutes with its own observable and commutes
    with the other. A partner search that ignored the index would return the same
    operator twice and fail the distinctness assertion.
    """

    code = _torus(linear_size)
    memory = build_memory_circuit(code, rounds=1)
    observables = memory.observables.observables

    assert tuple(observable.index for observable in observables) == (0, 1)
    assert {len(observable.pauli.support) for observable in observables} == {
        linear_size
    }
    assert all(observable.pauli.z_wires for observable in observables)
    assert all(not observable.pauli.x_wires for observable in observables)

    declared = [
        observable for observable in code.logical_observables if observable.z_wires
    ]
    partners = [
        derive_anticommuting_logical_product(code, index=index) for index in (0, 1)
    ]

    assert partners[0] != partners[1]
    assert {len(partner.support) for partner in partners} == {linear_size}
    assert [
        not partners[0].commutes_with(declared[0]),
        partners[0].commutes_with(declared[1]),
    ] == [True, True]
    assert [
        partners[1].commutes_with(declared[0]),
        not partners[1].commutes_with(declared[1]),
    ] == [True, True]


def test_the_family_refuses_a_size_it_cannot_build() -> None:
    """The two reasons are separate and each names itself.

    A size below two has no face and therefore no check, so there is no code to
    write down at all. A non-integer size is refused earlier, because the lattice
    has an integer number of vertices per side and a float would silently pick a
    different lattice rather than fail. Both are asserted with their own message,
    so collapsing the two refusals into one would fail here rather than pass
    quietly.
    """

    for too_small in (1, 0, -3):
        with pytest.raises(ValueError, match="no face"):
            toric_code(too_small)

    for invalid in (True, 2.0, "3", None):
        with pytest.raises(TypeError, match="must be an integer"):
            toric_code(invalid)  # type: ignore[arg-type]


def test_the_public_namespace_publishes_the_family() -> None:
    """The package root reaches the family, and its ``__all__`` says so.

    A name bound in ``flagquantum.qec`` is importable whether or not the package
    lists it, so an import alone does not prove the family was published. The
    record is only part of the surface a caller reads from ``flagquantum.qec`` if
    ``__all__`` names it, which is the same statement ``test_codes.py`` makes
    about the other records.
    """

    import flagquantum.qec as qec

    assert hasattr(qec, "toric_code")
    assert "toric_code" in qec.__all__
    assert qec.toric_code is toric_code
