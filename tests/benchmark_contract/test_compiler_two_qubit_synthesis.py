"""Contract for the two-qubit KAK synthesis measurement W8-02 records.

The benchmark module answers one question: how much of the declared two-qubit
unitary group does native-gate legalization reach once it can decompose a unitary
over a supercontrolled entangler? The recorded answer is that every basis which
publishes a z-rotation, a pi/2 x-rotation, and ``cx`` or ``cz`` now reaches all 11
declared opcodes through an explicit matrix, up from none at all, at an entangler
cost of 1, 2, or 3.

These tests hold that evidence in place. They fail if a basis stops being fully
reachable, if the entangler cost of any opcode drifts, if the synthesized program
stops being equal to its source up to a single global phase, or if the phase gap
is reported as zero.
"""

import pytest

from benchmarks.compiler_two_qubit_synthesis import (
    DEFAULT_BASES,
    RANDOM_CASE_COUNT,
    SUPERCONTROLLED_ENTANGLERS,
    TWO_QUBIT_OPCODES,
    baseline_reach,
    reach,
    run_benchmark,
)

pytestmark = pytest.mark.benchmark_contract

# Bases whose two-qubit group is now closed by matrix, and the count each one
# reached by name before the KAK synthesis existed. Both numbers are pinned so a
# regression that silently drops an opcode cannot pass as "still reachable".
_CLOSED_BASES = ("ibm-rz-sx-cx", "ibm-heron-cz", "rotational")

#: The entangler cost of every declared arity-2 unitary opcode, pinned per
#: opcode rather than as a total so one opcode drifting cannot hide in the sum.
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

#: The named-gate reach each basis had before W8-02: `cx` alone in a `cx` basis,
#: `cz` alone in a `cz` basis, and `swap` via the hand-written three-`cx` rule
#: wherever `cx` is native. A matrix-carrying instruction had no path at all.
_NAMED_BEFORE = {"ibm-rz-sx-cx": 2, "ibm-heron-cz": 1, "rotational": 1, "clifford-t": 2}


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
    assert SUPERCONTROLLED_ENTANGLERS == ("cx", "cz")


def test_every_closed_basis_reaches_the_group_through_a_matrix() -> None:
    measured = {row["label"]: row for row in (reach(basis) for basis in DEFAULT_BASES)}
    assert set(measured) == set(_CLOSED_BASES) | {"clifford-t"}
    for label in _CLOSED_BASES:
        row = measured[label]
        assert row["legalized_by_matrix_count"] == 11, row["unresolved_errors"]
        assert row["unresolved_opcode_count"] == 0
        assert row["entangler_count_by_opcode"] == _ENTANGLER_COST
        assert row["total_entangler_count"] == sum(_ENTANGLER_COST.values()) == 20
        assert row["entangler_opcode"] in SUPERCONTROLLED_ENTANGLERS
        # The count this change replaced, computed from the same descriptors.
        assert row["rewrite_only_legalized_by_name_count"] == _NAMED_BEFORE[label]


def test_a_named_two_qubit_gate_still_stops_at_the_hand_written_rules() -> None:
    """Only a matrix reaches KAK, and only from a caller that already has one.

    Turning a *named* two-qubit gate into the entangler basis needs its matrix,
    and the matrix of a named gate belongs to `flagquantum.simulation`. A
    compiler layer that imported it would take a dependency it must not have, so
    the named path keeps its previous reach and fails closed beyond it.
    """

    for basis in DEFAULT_BASES:
        row = reach(basis)
        assert row["legalized_by_name_count"] == _NAMED_BEFORE[basis.label]
    # `clifford-t` is the one basis where the named path reaches further than the
    # matrix path: `swap` has a hand-written three-`cx` rule that needs no matrix,
    # while `cx` is the only opcode whose own matrix is native there.
    assert reach(DEFAULT_BASES[0])["legalized_by_name_count"] < 11
    assert reach(DEFAULT_BASES[3])["legalized_by_name_count"] == 2


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
        assert (
            baseline_reach(basis)["legalized_opcode_count"]
            == _NAMED_BEFORE[basis.label]
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
        if row["opcode"] == row["entangler"]:
            # `cx` over `cx` and `cz` over `cz` are already in the entangler
            # class, so this one is the entangler and nothing else.
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


def test_the_two_entanglers_cost_the_same_for_every_opcode(payload: dict) -> None:
    """`cx` and `cz` are locally equivalent, so the cost may not depend on one."""

    by_key = {
        (row["opcode"], row["entangler"]): row["entangler_count"]
        for row in payload["entangler_cost"]
    }
    for opcode in TWO_QUBIT_OPCODES:
        assert by_key[(opcode, "cx")] == by_key[(opcode, "cz")], opcode


def test_a_seeded_population_that_is_not_a_named_gate_is_reached_too(
    payload: dict,
) -> None:
    measured = {row["label"]: row for row in payload["random_reach"]}
    assert set(measured) == set(_NAMED_BEFORE)
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
    """

    anchor = payload["reference_anchor"]
    if not anchor["available"]:
        pytest.skip(f"Qiskit not importable: {anchor['reason']}")
    expected = len(TWO_QUBIT_OPCODES) * len(SUPERCONTROLLED_ENTANGLERS)
    assert anchor["compared_case_count"] == expected
    assert anchor["mismatches"] == []
    assert anchor["entangler_count_agreement_count"] == expected
    for row in anchor["rows"]:
        assert row["port_entangler_count"] == _ENTANGLER_COST[row["opcode"]], row
