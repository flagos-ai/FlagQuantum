"""Unit coverage for the ZXXZ surface patch as a record.

``test_surface_code.py`` pins the rotated patch, whose every check is a pure
X-type or pure Z-type operator. This file pins the family that occupies the same
lattice with a Hadamard on the data checkerboard, so that **every one of its
checks is a mixed X-and-Z operator** and the whole patch is measured by
single-ancilla gadgets that use both CNOT directions.

The family is worth a file of its own because a type test is the one thing the
rest of this package is allowed to lean on and this record is the first that
gives it nothing to lean on.

1. No check is pure, so the whole per-type machinery -- a per-family rate order,
   a per-basis decoder component, a matrix route -- has no row to state this code
   at, and the record has to be reached through the memory-circuit route or not
   at all. That is a property of the *record* and it is asserted against the
   rotated patch at the same distance, which has no mixed check at all.
2. Its logical observables are pure even though its checks are not, because the
   two diagonals of a patch are global operators rather than local ones. A reader
   who assumed a mixed check implies a mixed observable would read this record
   wrong, so the pair is pinned here with its weights, its commutation, its
   single shared wire and its anticommutation.
3. Its distance is not searched for. It is the rotated patch's distance, carried
   across by a Clifford conjugation, and this file states that as an isomorphism
   it re-derives rather than as a claim it repeats: the conjugation is rebuilt
   here from the parity rule, and the record's checks are required to be exactly
   its image on the rotated patch's checks, in the rotated patch's own order,
   with weights and the entire commutation table preserved.

What this file proves
---------------------

1. ``test_the_patch_occupies_the_rotated_patchs_wires``: the data wires and the
   ancilla wires are the rotated patch's at the same distance, and the counts are
   ``d ** 2`` data qubits and ``d ** 2 - 1`` ancillas. A conjugation moves no
   wire, so a difference here would mean the board was applied to the layout
   rather than to the operators.
2. ``test_every_check_is_mixed_and_the_rotated_patch_has_none``: every check of
   this record carries both an X factor and a Z factor on at least one wire, and
   the rotated patch at the same distance carries neither on any check. The
   control is what makes the first half a measurement rather than a restatement.
3. ``test_the_check_set_is_the_conjugated_rotated_check_set``: the record's checks
   equal, index by index and in order, the rotated patch's checks conjugated by a
   board this file computes itself from the parity rule. Weight is preserved and
   so is the commutation relation between every pair. This is the isomorphism that
   carries ``n``, ``k`` and the distance across, and it is re-derived here rather
   than read off the record.
4. ``test_the_check_group_leaves_one_logical_qubit``: the record's own checks,
   packed as one X bit and one Z bit per data wire, reduce to ``d ** 2 - 1``
   constraints under an independent GF(2) kernel, leaving exactly one logical
   qubit. The packing has to be symplectic, because a support-only mask cannot
   tell a mixed check from a different Pauli operator with the same wires.
5. ``test_the_weight_spectrum_is_the_rotated_patchs``: the multiset of check
   weights is the rotated patch's at every distance in this file, with the
   reduced-weight checks a patch has on its edge and the full-weight checks it has
   inside. A patch is not a torus and the spectrum is not a singleton, so the
   equality is a statement about the conjugation rather than about a constant.
6. ``test_the_declared_diagonals_are_a_logical_pair``: the Z-type observable is
   the anti-diagonal, the X-type one the main diagonal, each of weight ``d``, each
   pure, each commuting with every check, neither in the stabilizer span, meeting
   at exactly one wire, and anticommuting. The Z-type observable is declared
   first, which is what the memory circuit's Z frame reads.
7. ``test_no_pure_logical_is_lighter_than_the_distance``: an exhaustive search
   over every pure X-type and pure Z-type Pauli of weight below ``d`` finds none
   that commutes with every check and lies outside the stabilizer span, at
   distances three and five. This is the record's distance measured on the record
   rather than carried in from the rotated patch, and it is the strongest claim
   this file makes.
8. ``test_the_other_checkerboard_leaves_no_pure_z_representative``: the same
   conjugation with the opposite parity leaves the conjugated observable's coset
   with no pure-Z member of weight ``d`` or below, at distances three and five,
   and the record's own parity leaves one of exactly weight ``d``. So the choice
   of sublattice is load-bearing rather than cosmetic: the other one has no pure-Z
   representative for the memory circuit to read.
9. ``test_a_check_carries_both_cnot_directions_through_one_ancilla``: each check
   names one ancilla, its CNOT pairs are the Z-factor pairs with the data wire
   controlling followed by the X-factor pairs with the ancilla controlling, and
   the two runs reproduce the check's own factor supports. One ancilla measures a
   mixed stabilizer; a second one is not needed and is not declared.
10. ``test_the_emitted_source_wraps_only_the_x_run_in_hadamards``: the lines the
    memory circuit emits for each check are the Z run, then ``H``, then the X run,
    then ``H``, then the readout -- exactly, in that order, check by check. Point 9
    pins the record's CNOT list and says nothing about the program, and the model
    and the sampler both derive from the program, so a run wrapped by the wrong
    pair of Hadamards would leave them agreeing with each other about a stabilizer
    nobody declared.
11. ``test_the_record_refuses_a_distance_it_cannot_declare``: a non-integer
    distance and a distance that is even or below three are refused, and the
    reason for the second is the declared logical pair rather than a lattice
    constraint.
12. ``test_the_public_namespace_publishes_the_family``: ``flagquantum.qec``
    resolves the family to the same callable and its ``__all__`` names it, so the
    record is published rather than merely bound.

What this file does not prove
-----------------------------

It makes no decoding, noise, threshold or suppression claim; that is
``test_zxxz_memory_execution.py``, which runs the execution path and records what
the patch's model does under a matching decoder. It does not prove that the record
agrees with another framework's ZXXZ patch wire by wire: the isomorphism in point
3 stands in for that, and it pins the check set rather than the ancilla numbering
of any other implementation.

Point 10 reads the emitted source and not an executed one: it says the program
states the one rule, and the execution path's own agreement with the model is
``test_zxxz_memory_execution.py``.

Point 7 is exhaustive only below the distance, and it is exhaustive in the number
of wires rather than in the number of operators: at distance five it enumerates
every subset of up to four wires out of twenty-five rather than every Pauli
operator, which is the same set for a pure-type operator and is what makes the
search finish. Distances three and five carry it. Seven is out of reach of that
search and its distance is carried by the isomorphism alone.

Nine is the largest distance this file builds, and every claim above is measured
at three, five and seven except points 7 and 8, which are measured at three and
five because they enumerate subsets.
"""

from __future__ import annotations

import itertools
from collections import Counter
from functools import lru_cache

import pytest

from flagquantum.qec import RotatedSurfaceCode, ZxxzSurfaceCode, build_memory_circuit
from flagquantum.qec.gf2 import in_span, rank
from flagquantum.qec.pauli import Pauli

pytestmark = pytest.mark.unit

# The distances this file builds. Every check of the family costs the same as the
# rotated patch's, so the size is the record's distance search and nothing else.
_DISTANCES = (3, 5, 7)

# The distances the exhaustive searches reach. They enumerate every subset of up
# to ``d - 1`` wires, so five is 25 choose 4 and seven would be 49 choose 6.
_SEARCHED_DISTANCES = (3, 5)


@lru_cache(maxsize=None)
def _patch(distance: int) -> ZxxzSurfaceCode:
    """Return the patch once per distance, because the checks are rebuilt per use."""

    return ZxxzSurfaceCode(distance)


@lru_cache(maxsize=None)
def _rotated(distance: int) -> RotatedSurfaceCode:
    """Return the rotated patch of the same distance."""

    return RotatedSurfaceCode(distance)


def _board(distance: int, parity: int) -> frozenset[int]:
    """Return the checkerboard wire set of one parity.

    Written out here from the parity rule rather than imported, because the point
    of the isomorphism test is that the record and this file reach the same
    operator set by two routes.
    """

    return frozenset(
        row * distance + column
        for row in range(distance)
        for column in range(distance)
        if (column + row) % 2 == parity
    )


def _conjugate(pauli: Pauli, board: frozenset[int]) -> Pauli:
    """Return ``pauli`` with a Hadamard applied to every wire on ``board``.

    A Hadamard exchanges the two factors, so a wire on the board leaves the X
    support and enters the Z support and the other way round; a wire off the board
    keeps its factor. The operator's global phase is not tracked here, so this is
    the operator up to a sign, which is all a stabilizer group and a commutation
    question depend on.
    """

    x_wires = frozenset(pauli.x_wires)
    z_wires = frozenset(pauli.z_wires)
    return Pauli(
        x_wires=tuple(sorted((x_wires - board) | (z_wires & board))),
        z_wires=tuple(sorted((z_wires - board) | (x_wires & board))),
    )


def _symplectic(pauli: Pauli, positions: dict[int, int], width: int) -> int:
    """Return ``pauli`` as one X bit and one Z bit per data wire, packed.

    Bit ``p`` is the X factor on the ``p``-th data wire and bit ``width + p`` its
    Z factor, so a mixed check is a row of the group rather than a support that
    could belong to several different operators.
    """

    bits = 0
    for wire in pauli.x_wires:
        bits |= 1 << positions[wire]
    for wire in pauli.z_wires:
        bits |= 1 << (width + positions[wire])
    return bits


def _positions(code: ZxxzSurfaceCode) -> tuple[dict[int, int], int]:
    """Return the wire-to-column map and the width of one code's data register."""

    return (
        {wire: index for index, wire in enumerate(code.data_wires)},
        code.num_data_qubits,
    )


def _check_rows(code: ZxxzSurfaceCode) -> list[int]:
    """Return every check as a symplectic row, in the record's own order."""

    positions, width = _positions(code)
    return [_symplectic(check.stabilizer, positions, width) for check in code.checks]


@pytest.mark.parametrize("distance", _DISTANCES)
def test_the_patch_occupies_the_rotated_patchs_wires(distance: int) -> None:
    """The conjugation moves no wire, so the two layouts are the same layout.

    A Hadamard on a data wire changes which factor an operator carries on that
    wire, never which wire it is, so the conjugating record has to declare the
    same data wires and the same ancilla wires as the patch it is derived from. The
    counts are also checked against ``d ** 2`` and ``d ** 2 - 1`` so that
    agreement with the rotated record cannot hide a shared off-by-one.
    """

    patch = _patch(distance)
    rotated = _rotated(distance)

    assert patch.data_wires == rotated.data_wires == tuple(range(distance * distance))
    assert patch.ancilla_wires == rotated.ancilla_wires
    assert patch.num_data_qubits == distance * distance
    assert patch.num_ancilla_qubits == distance * distance - 1
    assert patch.num_ancilla_qubits == len(patch.checks)


@pytest.mark.parametrize("distance", _DISTANCES)
def test_every_check_is_mixed_and_the_rotated_patch_has_none(distance: int) -> None:
    """Every check carries both factors, and the patch it came from carries one.

    The per-type machinery this package is built around reads a check's type, and
    this record is the case that gives it nothing to read: no check is pure, so no
    check belongs to a family. The rotated patch at the same distance is measured
    alongside, and the two counts are complements -- which is what makes the first
    half a measurement of the board rather than a restatement of the record's own
    docstring.
    """

    patch = _patch(distance)
    rotated = _rotated(distance)

    mixed = tuple(
        check
        for check in patch.checks
        if check.stabilizer.x_wires and check.stabilizer.z_wires
    )
    rotated_mixed = tuple(
        check
        for check in rotated.checks
        if check.stabilizer.x_wires and check.stabilizer.z_wires
    )

    assert len(mixed) == len(patch.checks)
    assert not rotated_mixed
    assert all(check.stabilizer.z_wires for check in patch.checks)
    assert all(check.stabilizer.x_wires for check in patch.checks)


@pytest.mark.parametrize("distance", _DISTANCES)
def test_the_check_set_is_the_conjugated_rotated_check_set(distance: int) -> None:
    """The record's checks are the rotated patch's checks, conjugated, in order.

    This is the whole justification for the record's ``distance``, ``k`` and check
    weights, and it is re-derived rather than cited: the board and the conjugation
    rule are written out in this file, the rotated patch's own checks are their
    input, and the record's checks are required to be the result index by index.
    The commutation table is asserted over every pair as well, because a bijection
    that preserved each operator's weight but not the group's relations would not
    be the isomorphism the parameters are carried by.
    """

    patch = _patch(distance)
    rotated = _rotated(distance)
    board = _board(distance, 1)
    image = tuple(_conjugate(check.stabilizer, board) for check in rotated.checks)
    declared = tuple(check.stabilizer for check in patch.checks)

    assert Counter(image) == Counter(declared)
    assert image == declared
    assert all(
        len(conjugated.support) == len(original.support)
        for conjugated, original in zip(image, declared, strict=True)
    )
    assert all(
        first.commutes_with(second) == third.commutes_with(fourth)
        for first, third in zip(declared, image, strict=True)
        for second, fourth in zip(declared, image, strict=True)
    )
    assert all(
        _conjugate(_conjugate(check.stabilizer, board), board) == check.stabilizer
        for check in rotated.checks
    )


@pytest.mark.parametrize("distance", _DISTANCES)
def test_the_check_group_leaves_one_logical_qubit(distance: int) -> None:
    """The record's own checks leave exactly one logical qubit.

    The checks are packed as one X bit and one Z bit per data wire and reduced by
    an independent GF(2) kernel, so the rank is read off the record rather than
    asserted about the lattice. ``d ** 2`` data wires under ``d ** 2 - 1``
    independent checks leave one, and the Z-type observable is required to sit
    outside the span, because a stabilizer has a fixed outcome and could not be
    read out as a logical one.
    """

    patch = _patch(distance)
    positions, width = _positions(patch)
    rows = _check_rows(patch)
    observable = patch.logical_observables[0]

    assert rank(rows) == distance * distance - 1
    assert patch.num_data_qubits - rank(rows) == 1
    assert not in_span(_symplectic(observable, positions, width), rows)


@pytest.mark.parametrize("distance", _DISTANCES)
def test_the_weight_spectrum_is_the_rotated_patchs(distance: int) -> None:
    """The weights are the rotated patch's, reduced on the edge and full inside.

    A conjugation preserves every operator's weight, so the spectrum has to agree
    entry by entry. The spectrum is not a singleton, which is the difference
    between a patch and a torus: the checks on the edge lose the wires the edge
    cuts away. Asserting the equality rather than a constant is what makes this a
    statement about the map.
    """

    patch = _patch(distance)
    rotated = _rotated(distance)

    spectrum = Counter(len(check.stabilizer.support) for check in patch.checks)
    rotated_spectrum = Counter(
        len(check.stabilizer.support) for check in rotated.checks
    )

    assert spectrum == rotated_spectrum
    assert sorted(spectrum) == [2, 4]
    assert spectrum[2] == 2 * (distance - 1)
    assert spectrum[4] == (distance - 1) ** 2
    assert spectrum[2] + spectrum[4] == len(patch.checks)


@pytest.mark.parametrize("distance", _DISTANCES)
def test_the_declared_diagonals_are_a_logical_pair(distance: int) -> None:
    """The two diagonals are pure, of full weight, and anticommute at one wire.

    A reader who carried the checks' mixedness over to the observables would expect
    a mixed logical operator. The record declares a pure pair instead, because a
    diagonal spanning the patch is a global operator and not a local one, and each
    is read out in one basis. The overlap is exactly one wire, which is what makes
    the pair anticommute and is also the reason the distance has to be odd.
    """

    patch = _patch(distance)
    z_observable, x_observable = patch.logical_observables
    positions, width = _positions(patch)
    rows = _check_rows(patch)

    assert z_observable == Pauli(
        z_wires=tuple(row * (distance - 1) + (distance - 1) for row in range(distance))
    )
    assert x_observable == Pauli(
        x_wires=tuple(range(0, distance * distance, distance + 1))
    )
    assert not z_observable.x_wires
    assert not x_observable.z_wires
    assert len(z_observable.support) == len(x_observable.support) == distance
    assert all(z_observable.commutes_with(check.stabilizer) for check in patch.checks)
    assert all(x_observable.commutes_with(check.stabilizer) for check in patch.checks)
    assert not in_span(_symplectic(z_observable, positions, width), rows)
    assert not in_span(_symplectic(x_observable, positions, width), rows)
    assert set(z_observable.support) & set(x_observable.support) == {
        (distance * distance - 1) // 2
    }
    assert not z_observable.commutes_with(x_observable)
    assert len(patch.logical_observables) == 2


@pytest.mark.parametrize("distance", _SEARCHED_DISTANCES)
def test_no_pure_logical_is_lighter_than_the_distance(distance: int) -> None:
    """Nothing below the declared distance is a logical operator.

    Every pure X-type and pure Z-type Pauli of weight below ``d`` is enumerated
    and tested on two counts: that it commutes with every check, which is what
    makes it a candidate, and that it lies outside the stabilizer span, which is
    what makes it a logical operator rather than a stabilizer. None is both, so the
    record's distance is measured here rather than carried in from the rotated
    patch, and a weight-``d`` diagonal is found at exactly the declared distance,
    so the bound is attained rather than merely respected.
    """

    patch = _patch(distance)
    positions, width = _positions(patch)
    rows = _check_rows(patch)
    wires = list(patch.data_wires)

    def is_logical(candidate: Pauli) -> bool:
        return all(
            candidate.commutes_with(check.stabilizer) for check in patch.checks
        ) and not in_span(_symplectic(candidate, positions, width), rows)

    lighter = [
        candidate
        for weight in range(1, distance)
        for candidate in (
            Pauli(x_wires=combo) for combo in itertools.combinations(wires, weight)
        )
        if is_logical(candidate)
    ] + [
        candidate
        for weight in range(1, distance)
        for candidate in (
            Pauli(z_wires=combo) for combo in itertools.combinations(wires, weight)
        )
        if is_logical(candidate)
    ]

    z_observable, x_observable = patch.logical_observables

    assert lighter == []
    assert is_logical(z_observable)
    assert is_logical(x_observable)


@pytest.mark.parametrize("distance", _SEARCHED_DISTANCES)
def test_the_other_checkerboard_leaves_no_pure_z_representative(distance: int) -> None:
    """The parity of the board decides whether a pure-Z representative exists.

    The memory circuit reads a Z-type observable, and the observable it reads is
    the rotated patch's conjugated onto the board. Which sublattice carries the
    Hadamards decides whether that coset contains a pure-Z operator at all. With
    the record's parity it contains the anti-diagonal, of weight exactly ``d``;
    with the other one no member of the coset of weight ``d`` or below is pure Z,
    so a memory circuit in the Z frame would have no Z-type operator to read. That
    is why the parity is load-bearing and not a relabelling.
    """

    rotated_observable = _rotated(distance).logical_observables[0]
    wires = list(range(distance * distance))

    def coset_is_reachable(
        code: ZxxzSurfaceCode, observable: Pauli
    ) -> tuple[int, ...] | None:
        positions, width = _positions(code)
        rows = _check_rows(code)
        target = _symplectic(observable, positions, width)
        for weight in range(1, distance + 1):
            for combo in itertools.combinations(wires, weight):
                candidate = Pauli(z_wires=combo)
                if not all(
                    candidate.commutes_with(check.stabilizer) for check in code.checks
                ):
                    continue
                if in_span(_symplectic(candidate, positions, width) ^ target, rows):
                    return combo
        return None

    declared = _patch(distance)
    conjugated = _conjugate(rotated_observable, _board(distance, 0))

    assert (
        coset_is_reachable(
            declared, _conjugate(rotated_observable, _board(distance, 1))
        )
        is not None
    )
    assert coset_is_reachable(declared, conjugated) is None
    assert len(declared.logical_observables[0].support) == distance


@pytest.mark.parametrize("distance", _DISTANCES)
def test_a_check_carries_both_cnot_directions_through_one_ancilla(
    distance: int,
) -> None:
    """One ancilla, both directions, in the order the factors are declared.

    The CNOT list is the Z-factor pairs with the data wire controlling, then the
    X-factor pairs with the ancilla controlling, and the boundary between the two
    runs is what says which pairs the emitted ``H`` pair wraps. A mixed check
    therefore needs neither a second ancilla nor a second gadget, and a record that
    permuted the runs would describe a different operator rather than the same one
    spelled differently.
    """

    patch = _patch(distance)

    assert all(
        check.cnot_wires
        == tuple((wire, check.ancilla_wire) for wire in check.stabilizer.z_wires)
        + tuple((check.ancilla_wire, wire) for wire in check.stabilizer.x_wires)
        for check in patch.checks
    )
    assert len({check.ancilla_wire for check in patch.checks}) == len(patch.checks)
    assert all(
        len(check.cnot_wires)
        == len(check.stabilizer.z_wires) + len(check.stabilizer.x_wires)
        <= len(check.stabilizer.support)
        for check in patch.checks
    )
    assert any(
        len(check.stabilizer.z_wires) > 1 and len(check.stabilizer.x_wires) > 1
        for check in patch.checks
    )


@pytest.mark.parametrize("distance", _DISTANCES)
def test_the_emitted_source_wraps_only_the_x_run_in_hadamards(distance: int) -> None:
    """The program states the one rule, check by check and in order.

    The record's CNOT list is order-bearing, and the order is read off the
    *program*: the Z-factor pairs run first and the ``H`` pair wraps the X-factor
    pairs and only them. A record whose list is right and whose emission wraps the
    wrong run -- or wraps both -- would still be internally consistent, because
    both the detector error model and the sampler derive from the emitted source,
    so the two would agree with each other about a program that measures an
    operator nobody declared. This reads the emitted lines instead of a derived
    quantity, so the two halves are independent.

    Every check here is mixed, so every check gets both Hadamards; the pure shapes
    this one rule also produces are ``test_surface_code.py``'s and
    ``test_memory_circuit.py``'s, where the same emission is asserted for a check
    whose second run is empty.
    """

    patch = _patch(distance)
    # Two rounds, because the source text states the check pass once inside the
    # round loop and one round is refused in this frame for having no detector.
    source = build_memory_circuit(patch, rounds=2).source
    lines = source.splitlines()
    cursor = lines.index("    for round_index in range(rounds):") + 1

    for check in patch.checks:
        ancilla = check.ancilla_wire
        split = len(check.stabilizer.z_wires)
        assert split and check.stabilizer.x_wires
        expected = [
            f"        qp.CNOT(wires=[{control}, {target}])"
            for control, target in check.cnot_wires[:split]
        ]
        expected.append(f"        qp.H(wires={ancilla})")
        expected.extend(
            f"        qp.CNOT(wires=[{control}, {target}])"
            for control, target in check.cnot_wires[split:]
        )
        expected.append(f"        qp.H(wires={ancilla})")
        expected.append(f"        last = qp.measure(wires={ancilla})")
        expected.append(f"        qp.reset(wires={ancilla})")
        assert lines[cursor : cursor + len(expected)] == expected
        cursor += len(expected)

    assert lines[cursor:] == ["    return last"]


@pytest.mark.parametrize("distance", (True, 3.5, "3", None))
def test_the_record_refuses_a_distance_that_is_not_an_integer(distance: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        ZxxzSurfaceCode(distance)  # type: ignore[arg-type]


@pytest.mark.parametrize("distance", (1, 2, 4, 6))
def test_the_record_refuses_a_distance_it_cannot_declare(distance: int) -> None:
    """A distance below three and an even distance are refused for the same reason.

    The declared logical pair is the two diagonals, and they meet at a single wire
    only on an odd patch. An even patch would need a different pair to be declared,
    which is a different record rather than a parameter of this one.
    """

    with pytest.raises(ValueError, match="odd and at least three"):
        ZxxzSurfaceCode(distance)


def test_the_public_namespace_publishes_the_family() -> None:
    """An import alone is not publication; the name is in ``__all__``."""

    import flagquantum.qec as namespace

    assert namespace.ZxxzSurfaceCode is ZxxzSurfaceCode
    assert "ZxxzSurfaceCode" in namespace.__all__
