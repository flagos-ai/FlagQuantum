"""Contract for the one-qubit Euler synthesis measurement W8-01 records.

The benchmark module answers one question: how much of the declared single-qubit
unitary group does native-gate legalization reach once it can synthesize Euler
angles? The recorded answer is that the four bases which publish a z-rotation and
a pi/2 x-rotation now reach all 18 declared opcodes, up from 2, 3, or 7.

These tests hold that evidence in place. They fail if a basis stops being fully
reachable, if the synthesized program stops being equal to its source up to a
single global phase, or if the phase gap is reported as zero.
"""

import pytest

from benchmarks.compiler_one_qubit_synthesis import (
    DEFAULT_BASES,
    SINGLE_QUBIT_OPCODES,
    baseline_reach,
    reach,
    run_benchmark,
)

pytestmark = pytest.mark.benchmark_contract

# Bases whose one-qubit group is now closed, and the count each one reached
# before the synthesis existed. Both numbers are pinned so a regression that
# silently drops an opcode cannot pass as "still reachable".
_CLOSED_BASES = {
    "ibm-rz-sx-cx": 3,
    "ibm-heron-cz": 3,
    "rotational": 2,
    "deployment-cloud-simulator": 7,
}

#: What the equivalence table reaches on its own, with no synthesis, on every
#: basis including the one the synthesis cannot help. `clifford-t` reaches six
#: names there -- the four it publishes, plus `sdg` as `s s s` and `x` as
#: `h z h` with `z` as `s s` -- and its reach does not move, because it
#: publishes no z-rotation for the synthesis to work against.
_IDENTITY_TABLE_REACH = {
    "ibm-rz-sx-cx": 3,
    "ibm-heron-cz": 3,
    "rotational": 2,
    "deployment-cloud-simulator": 7,
    "clifford-t": 6,
}


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_benchmark()


def test_measurement_is_classified_as_a_local_microbenchmark(payload: dict) -> None:
    assert payload["schema"] == "flagquantum_compiler_one_qubit_synthesis_benchmark_v1"
    assert payload["artifact_classification"] == "local_compiler_microbenchmark"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert payload["reference_algorithm"] == "qiskit_one_qubit_euler_decomposer"
    assert payload["basis_count"] == len(DEFAULT_BASES)


def test_the_declared_single_qubit_group_is_the_whole_arity_one_group() -> None:
    # Non-vacuity: the reach below is only a statement about the group if the
    # group really is every declared single-qubit unitary, not a chosen subset.
    assert len(SINGLE_QUBIT_OPCODES) == 18
    assert tuple(sorted(SINGLE_QUBIT_OPCODES)) == SINGLE_QUBIT_OPCODES
    assert {"i", "x", "h", "s", "t", "sx", "rx", "ry", "rz", "u2", "u3"} <= set(
        SINGLE_QUBIT_OPCODES
    )


def test_every_basis_that_publishes_a_z_rotation_and_a_pulse_is_closed() -> None:
    measured = {row["label"]: row for row in (reach(basis) for basis in DEFAULT_BASES)}
    assert set(measured) == set(_CLOSED_BASES) | {"clifford-t"}
    for label, before in _CLOSED_BASES.items():
        row = measured[label]
        assert row["legalized_opcode_count"] == 18, row["unresolved_errors"]
        assert row["unresolved_opcode_count"] == 0
        assert row["replacement_lengths"]["u3"] == 5
        assert row["replacement_lengths"]["i"] == 0
        # The count this change replaced, computed from the same descriptors.
        assert row["identity_table_reach_count"] == before


def test_a_basis_without_a_z_rotation_gains_nothing_from_the_synthesis() -> None:
    """`clifford-t` publishes `h`, `s`, `t` and `cx`, and no z-rotation.

    The synthesis needs a z-rotation and this basis has none, so the reach does
    not move: the six opcodes that get through are the ones the basis names
    directly plus the three the equivalence table rewrites into them. Every
    refusal has to be explicit, and the reach has to equal the table's.
    """

    row = next(
        item
        for item in (reach(basis) for basis in DEFAULT_BASES)
        if item["label"] == "clifford-t"
    )
    assert row["legalized_opcode_count"] == 6
    assert row["identity_table_reach_count"] == 6
    assert row["unresolved_opcode_count"] == 12
    # The longest replacement is `ry` -> `sdg h rz h s` with `sdg` expanded,
    # which is four leaves once the two unsupported ones are dropped; nothing
    # here was synthesized from a matrix.
    assert row["max_replacement_length"] == 4
    assert all(
        "unsupported native gate" in error or "no verified decomposition" in error
        for error in row["unresolved_errors"].values()
    ), row["unresolved_errors"]


def test_the_baseline_replay_agrees_with_the_pinned_before_counts() -> None:
    for basis in DEFAULT_BASES:
        expected = _IDENTITY_TABLE_REACH[basis.label]
        assert baseline_reach(basis)["legalized_opcode_count"] == expected, basis.label


def test_the_legalized_state_matches_up_to_exactly_one_global_phase(
    payload: dict,
) -> None:
    measured = {row["label"]: row for row in payload["phase"]}
    legalized = {label: row for label, row in measured.items() if row["legalized"]}
    # Every basis that reached the whole group reaches it on a real entangled
    # program too, not just opcode by opcode.
    assert set(legalized) == set(_CLOSED_BASES)
    for label, row in legalized.items():
        assert row["state_overlap_magnitude"] == pytest.approx(1.0, abs=1e-12), label
        assert row["native_only"] == []
        # A z-rotation and a pi/2 pulse cannot carry `h`'s global phase, so the
        # raw statevectors must differ. A change that starts preserving the phase
        # would make this fail rather than pass silently.
        assert abs(row["global_phase_radians"]) > 1e-3, label
        assert row["max_raw_state_difference"] > 0.1, label
    assert measured["clifford-t"]["legalized"] is False


def test_the_two_ibm_bases_agree_on_the_global_phase(payload: dict) -> None:
    measured = {row["label"]: row for row in payload["phase"]}
    # `cx` and `cz` are both real and diagonal, so swapping the entangler changes
    # the program but not the phase the synthesis drops. Agreement here is
    # evidence that the dropped phase comes from the one-qubit synthesis and not
    # from the two-wire gate.
    assert measured["ibm-rz-sx-cx"]["max_raw_state_difference"] != pytest.approx(
        measured["ibm-heron-cz"]["max_raw_state_difference"], abs=1e-6
    )
    assert measured["ibm-rz-sx-cx"]["global_phase_radians"] == pytest.approx(
        measured["ibm-heron-cz"]["global_phase_radians"], abs=1e-12
    )


def test_the_qiskit_anchor_agrees_on_length_for_every_opcode(payload: dict) -> None:
    """Qiskit's own ZSX decomposer, when this machine has Qiskit.

    The anchor is skipped rather than failed when Qiskit is absent: this module is
    a local benchmark and CI does not install Qiskit for it. Where the two do
    differ, the sequence is the same length and the fixed rotation sits on the
    other side of the pulse, which is a global phase apart rather than an error --
    the unit and integration suites prove the unitaries agree.
    """

    anchor = payload["reference_anchor"]
    if not anchor["available"]:
        pytest.skip(f"Qiskit not importable: {anchor['reason']}")
    # Which lane took this reading. These counts were measured as stable across the
    # certified lanes, so this is the one anchor here whose numbers do not move with
    # the installed library; the field is asserted anyway, because "stable as far as
    # we measured" is still a reading of a lane rather than a property of the anchor.
    assert anchor["qiskit_version"], anchor
    assert anchor["decomposer"] == "OneQubitEulerDecomposer('ZSX')"
    assert anchor["compared_gate_count"] == len(SINGLE_QUBIT_OPCODES) - 1
    assert anchor["same_length_count"] == anchor["compared_gate_count"]
    # 15 of 17 are the identical sequence; `y` and `ry` differ only in which side
    # of the pulse the fixed z-rotation lands on.
    assert anchor["identical_gate_count"] == 15
    differing = {row["opcode"] for row in anchor["rows"] if not row["identical"]}
    assert differing == {"y", "ry"}


def test_the_synthesis_table_covers_every_opcode_but_the_z_rotation(
    payload: dict,
) -> None:
    rows = {row["opcode"]: row for row in payload["synthesis"]}
    assert set(rows) == set(SINGLE_QUBIT_OPCODES) - {"rz"}
    for opcode, row in rows.items():
        assert row["leaf_count"] == len(row["leaf_opcodes"]), opcode
        assert set(row["leaf_opcodes"]) <= {"rz", "sx"}, opcode
        assert row["leaf_count"] <= 5, opcode
    assert rows["i"]["leaf_count"] == 0
    assert rows["u3"]["leaf_count"] == 5
