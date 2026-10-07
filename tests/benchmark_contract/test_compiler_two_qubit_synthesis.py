"""Contract for the two-qubit KAK synthesis measurements W8-02 and W8-03 record.

The benchmark module answers one question: how much of the declared two-qubit
unitary group does native-gate legalization reach once it can decompose a unitary
over a supercontrolled entangler? The answer W8-02 recorded is that every basis
which publishes a z-rotation, a pi/2 x-rotation, and ``cx`` or ``cz`` reaches all
11 declared opcodes through an explicit matrix, up from none at all, at an
entangler cost of 1, 2, or 3. W8-03 extended the entangler table to every
supercontrolled spelling -- adding ``cy``, and the interaction rotations ``rzz``,
``rxx`` and ``ryy`` at an angle of pi/2 -- which is what opens a basis whose only
two-qubit gate is a rotation.

These tests hold that evidence in place. They fail if a basis stops being fully
reachable, if the entangler cost of any opcode drifts, if the synthesized program
stops being equal to its source up to a single global phase, or if the phase gap
is reported as zero.
"""

import pytest

from benchmarks.compiler_two_qubit_synthesis import (
    _PRE_TABLE_NAMED_OPCODES,
    DEFAULT_BASES,
    RANDOM_CASE_COUNT,
    SUPERCONTROLLED_ENTANGLERS,
    TWO_QUBIT_OPCODES,
    baseline_reach,
    reach,
    run_benchmark,
    two_qubit_instruction,
)
from flagquantum.compiler.two_qubit_synthesis import synthesize_two_qubit
from tests.benchmark_contract.qiskit_lane import require_certified_lane

pytestmark = pytest.mark.benchmark_contract

#: The entanglers whose own matrix is the basis, so a target equal to one of them
#: is emitted as a single instruction rather than rebuilt. The rotation family is
#: applied at pi/2 and is therefore never this case.
_PARAMETER_FREE = ("cx", "cz", "cy")

# Bases whose two-qubit group is now closed by matrix, and the count each one
# reached by name before the KAK synthesis existed. Both numbers are pinned so a
# regression that silently drops an opcode cannot pass as "still reachable".
_CLOSED_BASES = (
    "ibm-rz-sx-cx",
    "ibm-heron-cz",
    "rotational",
    "ion-trap-rz-rx-rzz",
)

#: The entangler cost of every declared arity-2 unitary opcode over an entangler
#: the basis has to synthesize, pinned per opcode rather than as a total so one
#: opcode drifting cannot hide in the sum. A basis that already publishes its own
#: entangler keeps that instruction instead, so its count is one; see
#: `_NATIVE_TWO_QUBIT`.
_ENTANGLER_COST = {
    "cphase": 2,
    "crx": 2,
    "cry": 2,
    "crz": 2,
    "cx": 1,
    "cy": 1,
    "cz": 1,
    "rxx": 2,
    "ryy": 2,
    "rzz": 2,
    "swap": 3,
}

#: The one arity-2 opcode each basis publishes natively. In every basis here it is
#: also the entangler the basis is built around, which is asserted rather than
#: assumed so a basis whose native two-qubit gate is not its entangler cannot be
#: added without revisiting the cost accounting.
_NATIVE_TWO_QUBIT = {
    "ibm-rz-sx-cx": "cx",
    "ibm-heron-cz": "cz",
    "rotational": "cz",
    "ion-trap-rz-rx-rzz": "rzz",
    "clifford-t": "cx",
}

#: The entangler count each closed basis emits over the whole group: the sum of
#: `_ENTANGLER_COST`, minus the difference for the opcode the basis keeps instead
#: of synthesizing. `ion-trap-rz-rx-rzz` is one lower because its own `rzz` is
#: kept as a single instruction where the decomposition would have paid two.
_TOTAL_ENTANGLERS = {
    "ibm-rz-sx-cx": 20,
    "ibm-heron-cz": 20,
    "rotational": 20,
    "ion-trap-rz-rx-rzz": 19,
}

#: The named-gate reach each basis had before W8-04: the native gate itself, plus
#: `swap` wherever `cx` is native, because `swap` was the only arity-2 opcode with
#: a hand-written rule. That rule is still in the table verbatim, so the replay
#: restricts the table to it and measures the before count from the same
#: descriptors rather than quoting it.
_HAND_WRITTEN_REACH = {
    "ibm-rz-sx-cx": 2,
    "ibm-heron-cz": 1,
    "rotational": 1,
    "ion-trap-rz-rx-rzz": 1,
    "clifford-t": 2,
}

#: What the equivalence table reaches on its own, with no synthesis. Reported
#: beside the reach so the two contributions can be told apart: a `cx` basis with
#: a z-rotation gets five names from the table and the remaining six from the
#: Euler synthesis, while `clifford-t` gets all four of its names from the table
#: alone because it publishes no z-rotation to synthesize against.
_IDENTITY_TABLE_REACH = {
    "ibm-rz-sx-cx": 5,
    "ibm-heron-cz": 1,
    "rotational": 1,
    "ion-trap-rz-rx-rzz": 1,
    "clifford-t": 4,
}


def _by_label(name: str):
    return next(basis for basis in DEFAULT_BASES if basis.label == name)


def _expected_entangler_counts(label: str) -> dict[str, int]:
    native = _NATIVE_TWO_QUBIT[label]
    return {
        opcode: 1 if opcode == native else _ENTANGLER_COST[opcode]
        for opcode in TWO_QUBIT_OPCODES
    }


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_benchmark()


def test_measurement_is_classified_as_a_local_microbenchmark(payload: dict) -> None:
    assert payload["schema"] == "flagquantum_compiler_two_qubit_synthesis_benchmark_v1"
    assert payload["artifact_classification"] == "local_compiler_microbenchmark"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert payload["reference_algorithm"] == "qiskit_two_qubit_basis_decomposer"
    assert payload["basis_count"] == len(DEFAULT_BASES)


def test_the_declared_two_qubit_group_is_the_whole_arity_two_group() -> None:
    # Non-vacuity: the reach below is only a statement about the group if the
    # group really is every declared two-qubit unitary, not a chosen subset.
    assert len(TWO_QUBIT_OPCODES) == 11
    assert TWO_QUBIT_OPCODES == (
        "cphase",
        "crx",
        "cry",
        "crz",
        "cx",
        "cy",
        "cz",
        "rxx",
        "ryy",
        "rzz",
        "swap",
    )
    assert dict(SUPERCONTROLLED_ENTANGLERS) == {
        "cx": None,
        "cz": None,
        "cy": None,
        "rzz": pytest.approx(1.5707963267948966),
        "rxx": pytest.approx(1.5707963267948966),
        "ryy": pytest.approx(1.5707963267948966),
    }
    # `cphase` is supercontrolled only at an angle of pi, where it is `cz` under
    # another name, so it adds a spelling rather than reach.
    assert "cphase" not in SUPERCONTROLLED_ENTANGLERS


def test_every_closed_basis_reaches_the_group_through_a_matrix() -> None:
    measured = {row["label"]: row for row in (reach(basis) for basis in DEFAULT_BASES)}
    assert set(measured) == set(_CLOSED_BASES) | {"clifford-t"}
    for label in _CLOSED_BASES:
        row = measured[label]
        native = _NATIVE_TWO_QUBIT[label]
        assert native == row["entangler_opcode"], label
        expected = _expected_entangler_counts(label)
        assert row["legalized_by_matrix_count"] == 11, row["unresolved_errors"]
        assert row["unresolved_opcode_count"] == 0
        assert row["entangler_count_by_opcode"] == expected
        assert sum(expected.values()) == _TOTAL_ENTANGLERS[label]
        assert row["total_entangler_count"] == _TOTAL_ENTANGLERS[label]
        assert row["entangler_opcode"] in SUPERCONTROLLED_ENTANGLERS
        # The count this change replaced, computed from the same descriptors.
        assert row["identity_table_reach_count"] == _IDENTITY_TABLE_REACH[label]


def test_a_named_two_qubit_gate_reaches_the_group_through_exact_identities() -> None:
    """A named gate gets there without a matrix, so no layer is bypassed.

    Turning a *named* two-qubit gate into the entangler basis through KAK would
    need that gate's matrix, and the matrix of a named gate belongs to
    `flagquantum.simulation`, which the Compiler layer must not import. The
    equivalence table is what closes the named path instead: a basis publishing
    `cx` or `cz` plus a z-rotation now reaches all eleven declared opcodes by
    name, where before this change it reached one or two.
    """

    measured = {row["label"]: row for row in (reach(basis) for basis in DEFAULT_BASES)}
    for basis in DEFAULT_BASES:
        row = measured[basis.label]
        before = _HAND_WRITTEN_REACH[basis.label]
        # Non-vacuity: the improvement is only a claim about the table if the
        # before count it is compared against is really smaller.
        assert before < 11, basis.label
        assert row["legalized_by_name_count"] > before, basis.label
    # A `cx` or `cz` basis with a z-rotation and a pi/2 pulse closes the group.
    for label in ("ibm-rz-sx-cx", "ibm-heron-cz", "rotational"):
        assert measured[label]["legalized_by_name_count"] == 11, label
    # `clifford-t` publishes no z-rotation, so only the exact identities that
    # need nothing else apply: `cz` and `cy` join `cx` and `swap`, and the other
    # seven names need an interaction rotation to build from.
    assert measured["clifford-t"]["legalized_by_name_count"] == 4
    # `ion-trap-rz-rx-rzz` publishes no `cx`, `cz` or `cy`, so the three
    # interaction rotations are the only names its `rzz` can carry.
    assert measured["ion-trap-rz-rx-rzz"]["legalized_by_name_count"] == 3


def test_a_basis_without_a_z_rotation_keeps_the_matrix_path_open_and_the_rest_shut() -> (
    None
):
    """`clifford-t` publishes `cx` but no z-rotation, so no local factor exists."""

    row = next(
        item
        for item in (reach(basis) for basis in DEFAULT_BASES)
        if item["label"] == "clifford-t"
    )
    # `cx` is native, so its own matrix is kept; the other ten have no path.
    assert row["legalized_by_matrix_count"] == 1
    assert row["unresolved_opcode_count"] == 10
    assert row["entangler_count_by_opcode"] == {"cx": 1}
    assert all(
        "no native contract" in error for error in row["unresolved_errors"].values()
    ), row["unresolved_errors"]


def test_the_baseline_replay_agrees_with_the_pinned_before_counts() -> None:
    for basis in DEFAULT_BASES:
        pre_table = baseline_reach(basis, opcodes=_PRE_TABLE_NAMED_OPCODES)
        assert (
            pre_table["legalized_opcode_count"] == _HAND_WRITTEN_REACH[basis.label]
        ), basis.label
        # And the table as it now stands, on its own, for both benchmarks to
        # report the same quantity.
        assert (
            baseline_reach(basis)["legalized_opcode_count"]
            == _IDENTITY_TABLE_REACH[basis.label]
        ), basis.label


def test_the_entangler_cost_is_pinned_per_opcode_and_entangler(payload: dict) -> None:
    rows = payload["entangler_cost"]
    assert len(rows) == len(TWO_QUBIT_OPCODES) * len(SUPERCONTROLLED_ENTANGLERS)
    seen = set()
    for row in rows:
        key = (row["opcode"], row["entangler"])
        assert key not in seen, key
        seen.add(key)
        assert row["entangler_count"] == _ENTANGLER_COST[row["opcode"]], key
        # One local factor at each end of each entangler, and nothing else, so a
        # leaf set narrower than the basis means a factor went missing.
        assert len(row["leaf_opcodes"]) == row["leaf_count"]
        assert set(row["leaf_opcodes"]) <= {"rz", "sx", row["entangler"]}
        assert row["leaf_opcodes"].count(row["entangler"]) == row["entangler_count"]
        if row["opcode"] == row["entangler"] and row["entangler"] in _PARAMETER_FREE:
            # `cx` over `cx`, `cz` over `cz` and `cy` over `cy` are already in
            # the entangler class, so this one is the entangler and nothing else.
            # The rotation entanglers are *not* in this case: they are applied at
            # pi/2, so a target rotation at another angle still has to be built.
            assert row["leaf_opcodes"] == [row["entangler"]]
            assert row["entangler_count"] == 1
        else:
            assert row["leaf_opcode_set"] == sorted(["rz", "sx", row["entangler"]]), key
            assert row["leaf_count"] >= row["entangler_count"]
    assert seen == {
        (opcode, entangler)
        for opcode in TWO_QUBIT_OPCODES
        for entangler in SUPERCONTROLLED_ENTANGLERS
    }


def test_the_rotation_entangler_is_applied_at_the_supercontrolled_angle() -> None:
    """The emitted instruction carries `pi/2`, and only for the rotation family.

    A parameter-free entangler must not acquire a parameter it does not have, and
    a rotation entangler must not be emitted without one.
    """

    for entangler, angle in SUPERCONTROLLED_ENTANGLERS.items():
        emitted = synthesize_two_qubit(
            two_qubit_instruction("swap", matrix=True).matrix,
            qubits=(0, 1),
            entangler=entangler,
            z_rotation="rz",
        )
        assert emitted is not None
        for leaf in emitted:
            if leaf.name != entangler:
                continue
            assert set(leaf.params) == (
                set() if angle is None else {"theta"}
            ), entangler
            if angle is not None:
                assert leaf.params["theta"] == pytest.approx(angle)
        assert sum(1 for leaf in emitted if leaf.name == entangler) >= 1


def test_every_entangler_costs_the_same_for_every_opcode(payload: dict) -> None:
    """`cx`, `cz`, `cy`, `rxx`, `ryy` and `rzz` are all locally equivalent to
    `cx` up to a local factor, so the cost may not depend on the spelling."""

    by_key = {
        (row["opcode"], row["entangler"]): row["entangler_count"]
        for row in payload["entangler_cost"]
    }
    for opcode in TWO_QUBIT_OPCODES:
        costs = {
            by_key[(opcode, entangler)] for entangler in SUPERCONTROLLED_ENTANGLERS
        }
        assert costs == {_ENTANGLER_COST[opcode]}, (opcode, costs)


def test_a_seeded_population_that_is_not_a_named_gate_is_reached_too(
    payload: dict,
) -> None:
    measured = {row["label"]: row for row in payload["random_reach"]}
    assert set(measured) == set(_HAND_WRITTEN_REACH)
    for label in _CLOSED_BASES:
        row = measured[label]
        assert row["case_count"] == RANDOM_CASE_COUNT
        assert row["reached_count"] == RANDOM_CASE_COUNT
        assert row["worst_reconstruction_gap"] < 1e-9
        assert row["entangler_count_distribution"] == {3: RANDOM_CASE_COUNT}, label
    blocked = measured["clifford-t"]
    assert blocked["reached_count"] == 0
    assert blocked["entangler_count_distribution"] == {}


def test_the_legalized_state_matches_up_to_exactly_one_global_phase(
    payload: dict,
) -> None:
    measured = {row["label"]: row for row in payload["phase"]}
    legalized = {label: row for label, row in measured.items() if row["legalized"]}
    assert set(legalized) == set(_CLOSED_BASES)
    for label, row in legalized.items():
        assert row["state_overlap_magnitude"] == pytest.approx(1.0, abs=1e-12), label
        assert row["matrix_instruction_count"] == 0, label
        # Every instruction in the mixed program needs rewriting: two matrix
        # two-qubit unitaries plus `h`, `u3`, and `sdg`.
        assert row["decomposition_count"] == 5, label
        # The entangler basis cannot carry the global phase of `swap`, so the raw
        # statevectors must differ. A change that starts preserving the phase
        # would make this fail rather than pass silently.
        assert abs(row["global_phase_radians"]) > 1e-3, label
        assert row["max_raw_state_difference"] > 0.1, label
    assert measured["clifford-t"]["legalized"] is False


def test_the_qiskit_anchor_agrees_on_the_entangler_count(payload: dict) -> None:
    """Qiskit's own basis decomposer, when this machine has Qiskit.

    The anchor is skipped rather than failed when Qiskit is absent: this module is
    a local benchmark and CI does not install Qiskit for it. Only the entangler
    count is compared. The two sides choose different Euler bases, so their raw
    instruction lists are different presentations of the same cost.

    The anchor spans the port's whole entangler table, including the three
    spellings Qiskit's own chooser does not list, so this is a check that each
    table entry really is supercontrolled rather than a check that the two tables
    agree.
    """

    anchor = payload["reference_anchor"]
    if not anchor["available"]:
        pytest.skip(f"Qiskit not importable: {anchor['reason']}")
    # Which lane took this reading. The entangler counts asserted below are the
    # stable half of the anchor; the per-row `reference_gate_count` beside them
    # moves with the installed library, so the field that separates the two has to
    # be present for the stable half to be read as stable. The lane also has to be
    # certified: the per-row half moves between lanes, and a lane nobody certified
    # is one where neither half carries a measurement this repository stands behind.
    require_certified_lane(anchor["qiskit_version"], recording="this anchor")
    expected = len(TWO_QUBIT_OPCODES) * len(SUPERCONTROLLED_ENTANGLERS)
    assert anchor["compared_case_count"] == expected
    assert anchor["mismatches"] == []
    assert anchor["entangler_count_agreement_count"] == expected
    assert anchor["reference_entanglers"] == list(SUPERCONTROLLED_ENTANGLERS)
    for row in anchor["rows"]:
        assert row["port_entangler_count"] == _ENTANGLER_COST[row["opcode"]], row


#: What each basis reaches on the local-split population, pinned per basis. These
#: are reach counts -- a route either serves a case or refuses it -- and they are
#: saturated (all 96, or none), so they are pinned as equalities.
_SPLIT_REACH = {
    # label: (comparable, shipped_unreachable, entangler_unreachable)
    "ibm-rz-sx-cx": (96, 0, 0),
    "ibm-heron-cz": (96, 0, 0),
    "rotational": (96, 0, 0),
    "ion-trap-rz-rx-rzz": (96, 0, 0),
    "rz-sx-rotation-only": (96, 0, 96),
}

#: How many cases each basis's *hand-written* route -- the same operator spoken as
#: its two declared factors -- refuses. ``clifford-t`` publishes no z-rotation, so a
#: one-wire matrix instruction has no route on it either. The basis is kept in the
#: benchmark precisely so that a population reaching almost nothing cannot be read
#: as agreement.
_SPLIT_HAND_UNREACHABLE = {
    "ibm-rz-sx-cx": 0,
    "ibm-heron-cz": 0,
    "rotational": 0,
    "ion-trap-rz-rx-rzz": 0,
    "clifford-t": 90,
}

#: The two length columns are **not** pinned as equalities, and the reason is
#: measured rather than cautious: the number of instructions the entangler route
#: emits is decided by the one-qubit Euler synthesis, whose short forms branch on
#: exact polar-angle comparisons, so a 1-ulp difference in the platform's `libm`
#: moves a case from a three-leaf form to the general five-leaf one. The
#: classification rule this repository already applies to such a column is a
#: reference plus a documented allowance, so that is what these are: the gap is
#: structural -- `synthesize_two_qubit` pays its full local padding on the identity
#: branch of a product, which is 10 leaves, while the split emits between one and
#: four -- so the floor and ceiling below sit in the middle of an order-of-magnitude
#: gap rather than on a boundary, and each is stated against the measured count.
_ENTANGLER_LONGER_FLOOR = 8  # measured 16 of 96
_ENTANGLER_SHORTER_CEILING = 8  # measured 1 of 96, the identity case

#: The split reproduces its input by arithmetic, so its residual is rounding of an
#: exact product. The entangler route is a KAK synthesis, which cannot record the
#: global phase it drops because `CircuitIR` has no field for one, so its residual
#: is pi. The gap between the two is the measurement, not the tolerance.
_SPLIT_RESIDUAL_BOUND = 1e-14
_ENTANGLER_PHASE_FLOOR = 3.14
_ENTANGLER_PHASE_CEILING = 3.15


def _split_by_label(payload: dict) -> dict[str, dict]:
    rows = list(payload["product_split"]) + [payload["product_split_rotation_only"]]
    return {row["label"]: row for row in rows}


def test_the_split_payload_is_pinned_per_basis(payload: dict) -> None:
    rows = _split_by_label(payload)
    assert set(rows) == set(_SPLIT_REACH) | {"clifford-t"}
    for label, expected in _SPLIT_REACH.items():
        row = rows[label]
        assert row["case_count"] == 96, label
        assert (
            row["comparable_case_count"],
            row["shipped_unreachable_case_count"],
            row["entangler_unreachable_case_count"],
        ) == expected, (label, row)
        if row["entangler"] is None:
            continue
        assert row["entangler_longer_than_shipped_count"] >= _ENTANGLER_LONGER_FLOOR, (
            label,
            row,
        )
        assert row["entangler_shorter_than_shipped_count"] <= (
            _ENTANGLER_SHORTER_CEILING
        ), (label, row)


def test_the_split_reaches_every_case_the_hand_written_route_reaches(
    payload: dict,
) -> None:
    """The claim, stated as reach and not as agreement between two routes.

    A basis whose shipped route reaches nothing must not be able to pass this by
    matching a hand-written route that also reaches nothing, which is why the two
    counts are pinned separately for every basis rather than compared to each
    other. The rotation-only basis is the sharp case: it publishes no entangler, so
    the route that existed before reaches none of the 96 cases there, and the split
    reaches every one of them.
    """

    rows = _split_by_label(payload)
    for label, hand_unreachable in _SPLIT_HAND_UNREACHABLE.items():
        assert rows[label]["hand_split_unreachable_case_count"] == hand_unreachable
    for label in _SPLIT_REACH:
        assert rows[label]["shipped_unreachable_case_count"] == 0, label

    # `clifford-t` reaches nothing either way, and says so rather than reporting a
    # comparison it cannot make.
    clifford = rows["clifford-t"]
    assert clifford["comparable_case_count"] == 0
    assert clifford["shipped_unreachable_case_count"] == 96

    # The rotation-only basis has no entangler to synthesize at all, so the
    # entangler arm reaches nothing and the count is a failure rather than a tie.
    rotation_only = rows["rz-sx-rotation-only"]
    assert rotation_only["entangler"] is None
    assert rotation_only["entangler_unreachable_case_count"] == 96


def test_the_split_drops_no_global_phase_and_the_entangler_route_drops_one(
    payload: dict,
) -> None:
    """Both halves of the comparison, so neither can pass by being zero.

    The split's phase column is the whole reason this pass exists in the form it
    does: its two factors are read off the product's own blocks with a positive real
    scaling, so no phase has to be recorded anywhere -- and `CircuitIR` has no field
    to record one in. The entangler route has the opposite property, and the
    benchmark reports it, which is what stops the first column from being read as
    "this input class is easy" rather than "this route is exact".
    """

    for row in _split_by_label(payload).values():
        if row["comparable_case_count"] == 0:
            continue
        assert row["worst_split_entry_residual"] < _SPLIT_RESIDUAL_BOUND, row["label"]
        assert row["worst_split_phase_radians"] < _SPLIT_RESIDUAL_BOUND, row["label"]
        if row["entangler"] is None:
            continue
        assert (
            _ENTANGLER_PHASE_FLOOR
            < row["worst_entangler_phase_radians"]
            < _ENTANGLER_PHASE_CEILING
        ), row["label"]
