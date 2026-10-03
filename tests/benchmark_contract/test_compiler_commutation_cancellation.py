"""Contract for the commutation measurements W9-01 and W9-02 record.

The benchmark module answers three questions at once. How much of the declared
arity-one and arity-two unitary group does the rule source in
``flagquantum.compiler.commutation`` actually decide? What does the cancellation
pass built on it remove that the pipeline could not remove before? And where does
a diagonal rotation sit inside a controlled gate without changing it?

The recorded answers are that the rule source decides **3602** of the **3826**
pairs that really do commute and never says "yes" to a pair that does not; that
the pass removes **495** instructions on a chain of 40 six-gate circuits over
**708** the legacy pipeline leaves, while the pipeline's reach on a rotation
placed on a wire the gate does not touch is unchanged at zero; and that the
rotation is tolerable on a control and on a diagonal gate's wire, and not on the
target of ``cx``, ``cy`` or ``ccx``, nor anywhere in ``swap`` or ``cswap``.

These tests hold that evidence in place. They fail if the rule source starts
answering a pair it should decline, if it stops answering one it did answer, if
the pass's reach on any population drifts, if the worst statevector difference
stops being reported, or if the gate counts are presented as agreeing with
Qiskit's when the two passes are measuring different sets.
"""

import pytest

from benchmarks.compiler_commutation_cancellation import (
    _CONTROLLED,
    _QISKIT_SCOPE,
    _REGISTER_WIDTH,
    _SWEEP_SEED,
    run_benchmark,
)

pytestmark = pytest.mark.benchmark_contract

#: The pair counts the rule source is pinned to. `pair_count` is a property of
#: the placement list and the declared opcode set, so it is a fixture rather than
#: a measurement: if it moves, the sweep below is measuring something else and the
#: other three numbers are not comparable with it.
_SWEEP = {
    "pair_count": 5776,
    "true_commuting_pair_count": 3826,
    "answered_pair_count": 3602,
    "declined_pair_count": 224,
    "rule_opcode_count": 28,
}

#: Per-population instruction counts after each pipeline. `removed_by_legacy` is
#: what `merge_self_inverse`, `merge_adjacent_rotations` and
#: `remove_identity_gates` find on their own; `removed_by_commutation` is the
#: delta this round adds, and it must be positive where the gap sits on a wire the
#: rule source proves and exactly zero where it does not.
_DELTA = {
    "gap_on_wire_0": {
        "circuit_count": 60,
        "source_instruction_count": 180,
        "legacy_instruction_count": 180,
        "optimized_instruction_count": 86,
        "removed_by_commutation": 94,
        "changed_circuit_count": 47,
    },
    "gap_on_wire_1": {
        "circuit_count": 60,
        "source_instruction_count": 180,
        "legacy_instruction_count": 180,
        "optimized_instruction_count": 132,
        "removed_by_commutation": 48,
        "changed_circuit_count": 24,
    },
    "gap_on_unrelated_qubit": {
        "circuit_count": 60,
        "source_instruction_count": 180,
        "legacy_instruction_count": 94,
        "optimized_instruction_count": 94,
        "removed_by_commutation": 0,
        "changed_circuit_count": 0,
    },
    "chained_control_gaps": {
        "circuit_count": 40,
        "source_instruction_count": 720,
        "legacy_instruction_count": 708,
        "optimized_instruction_count": 213,
        "removed_by_commutation": 495,
        "changed_circuit_count": 40,
    },
}

#: Where a diagonal single-qubit rotation commutes with each controlled gate,
#: read off `commute` position by position. This is the rule source's own answer
#: restated as a table, and it is what explains the `gap_on_wire_1` row above:
#: `cz` and the first two wires of `ccx` tolerate the rotation, `swap` never does.
_GAP_POSITIONS = {
    "cx": (True, False),
    "cy": (True, False),
    "cz": (True, True),
    "swap": (False, False),
    "ccx": (True, True, False),
    "cswap": (True, False, False),
}


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_benchmark()


def test_measurement_is_classified_as_a_local_microbenchmark(payload: dict) -> None:
    assert payload["schema"] == (
        "flagquantum_compiler_commutation_cancellation_benchmark_v1"
    )
    assert payload["artifact_classification"] == "local_compiler_microbenchmark"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert payload["seed"]["sweep"] == _SWEEP_SEED


def test_the_sweep_is_not_vacuous(payload: dict) -> None:
    """A sweep whose population is empty proves nothing, however clean it looks."""

    sweep = payload["rule_source_sweep"]
    assert sweep["pair_count"] > 1000
    assert sweep["true_commuting_pair_count"] > 1000
    # Both verdicts have to occur, or "no false yes" would be a statement about a
    # rule source that answers "no" to everything.
    assert 0 < sweep["answered_pair_count"] < sweep["true_commuting_pair_count"]
    assert sweep["same_wire_yes_count"] > 0


def test_the_rule_source_never_says_yes_to_a_pair_that_does_not_commute(
    payload: dict,
) -> None:
    """The one property that cannot be traded away for reach.

    A false "yes" lets the cancellation pass remove a gate that changes the
    program. Every entry below is an exact matrix comparison against the same
    runtime matrices the compiler's consumers execute with, so this is a
    correctness gate, not a coverage one.
    """

    sweep = payload["rule_source_sweep"]
    assert sweep["false_yes_count"] == 0, sweep["false_yes"]


def test_the_sweep_counts_are_the_ones_the_rule_source_currently_produces(
    payload: dict,
) -> None:
    """Pinned so a rule change cannot quietly move the decline boundary."""

    sweep = payload["rule_source_sweep"]
    for key, expected in _SWEEP.items():
        assert sweep[key] == expected, key
    assert sweep["answered_share"] == pytest.approx(
        _SWEEP["answered_pair_count"] / _SWEEP["true_commuting_pair_count"],
        abs=1e-12,
    )


def test_the_declines_are_reported_by_opcode_pair_rather_than_hidden(
    payload: dict,
) -> None:
    """An owned gap, not an oversight: the residue has to be enumerable."""

    rows = payload["rule_source_sweep"]["declined_by_opcode_pair"]
    assert rows
    assert sum(row["count"] for row in rows) == _SWEEP["declined_pair_count"]
    for row in rows:
        assert len(row["opcode_pair"]) == 2
        assert row["count"] > 0
    # Symmetric by construction: the sweep enumerates ordered pairs but keys the
    # decline table by the sorted pair, so no opcode pair may appear twice.
    keys = [tuple(row["opcode_pair"]) for row in rows]
    assert len(keys) == len(set(keys))


def test_every_population_reports_its_own_removal_count(payload: dict) -> None:
    rows = {row["label"]: row for row in payload["pipeline_delta"]}
    assert set(rows) == set(_DELTA)
    for label, expected in _DELTA.items():
        for key, value in expected.items():
            assert rows[label][key] == value, (label, key, rows[label][key])


def test_removing_every_instruction_preserved_the_program(payload: dict) -> None:
    """Every changed circuit was executed, and the difference is reported.

    Two of the four populations are exact: the removed gates are their own inverses
    and the gap between them provably commutes, so the statevector is bit-identical
    rather than merely close. The chained population composes rotations, which is
    where the only nonzero difference comes from -- it is the rotation merge's
    floating-point sum, not the removal.
    """

    rows = {row["label"]: row for row in payload["pipeline_delta"]}
    for label in ("gap_on_wire_0", "gap_on_wire_1", "gap_on_unrelated_qubit"):
        assert rows[label]["max_state_difference"] == 0.0, label
    chained = rows["chained_control_gaps"]
    assert chained["changed_circuit_count"] == chained["circuit_count"]
    assert 0.0 < chained["max_state_difference"] < 1e-12


def test_the_pass_removes_nothing_it_did_not_remove_before_on_a_disjoint_gap(
    payload: dict,
) -> None:
    """The control that makes the delta attributable to this pass.

    A rotation on a qubit the controlled gate does not touch is what
    `merge_self_inverse` already saw. The pass must leave that population exactly
    where the legacy pipeline left it, or the delta above would be measuring the
    legacy pipeline twice.
    """

    rows = {row["label"]: row for row in payload["pipeline_delta"]}
    disjoint = rows["gap_on_unrelated_qubit"]
    assert disjoint["removed_by_commutation"] == 0
    assert disjoint["changed_circuit_count"] == 0
    assert (
        disjoint["optimized_instruction_count"] == disjoint["legacy_instruction_count"]
    )
    # And the population is not degenerate: the legacy pipeline does remove gates
    # there, so "no change" is a statement about this pass and not about a fixture
    # where nothing happens at all.
    assert disjoint["removed_by_legacy"] > 0


def test_the_gap_position_table_is_the_rule_source_read_back(payload: dict) -> None:
    rows = {row["opcode"]: row for row in payload["gap_position_table"]}
    assert set(rows) == set(_GAP_POSITIONS)
    for opcode, expected in _GAP_POSITIONS.items():
        row = rows[opcode]
        assert tuple(row["rotation_commutes_on_wire"]) == expected, opcode
        assert row["arity"] == len(expected), opcode
        assert row["tolerated_wire_count"] == sum(expected), opcode


def test_the_gap_position_table_explains_the_wire_one_row(payload: dict) -> None:
    """Why `gap_on_wire_1` removes less than `gap_on_wire_0`, measured not asserted.

    Position 1 is a control of `ccx`, a wire of the diagonal `cz`, and a swapped
    wire of `cswap` and `swap`. Only the first two tolerate the rotation, so the
    pass must remove strictly less there -- if a future rule change made the two
    rows equal, the gate counts above would no longer mean what they claim.
    """

    rows = {row["label"]: row for row in payload["pipeline_delta"]}
    assert rows["gap_on_wire_1"]["removed_by_commutation"] < (
        rows["gap_on_wire_0"]["removed_by_commutation"]
    )
    tolerating = {
        row["opcode"]
        for row in payload["gap_position_table"]
        if row["rotation_commutes_on_wire"][1]
    }
    assert tolerating == {"cz", "ccx"}


def test_the_chain_removes_a_share_and_not_a_fixed_few(payload: dict) -> None:
    """The chain is where the pass compounds, so it is the one with a ratio.

    Asserting the ratio rather than only the count keeps the claim honest when the
    fixture changes: what matters is that six chained control gaps let the pass
    see past gates the legacy pipeline stopped at, not that 495 is a magic number.
    """

    chained = {row["label"]: row for row in payload["pipeline_delta"]}[
        "chained_control_gaps"
    ]
    assert chained["legacy_instruction_count"] > 0
    ratio = chained["legacy_instruction_count"] / chained["optimized_instruction_count"]
    assert ratio > 3.0, ratio
    assert chained["cancellation_group_count"] > 0
    assert chained["widest_commuting_block"] >= 1


def test_the_analysis_reports_a_block_shape_rather_than_only_a_verdict(
    payload: dict,
) -> None:
    rows = payload["pipeline_delta"]
    assert any(row["widest_commuting_block"] > 1 for row in rows)
    for row in rows:
        assert row["cancellable_position_count"] >= row["cancellation_group_count"]


def test_the_qiskit_anchor_compares_only_what_both_passes_attempt(
    payload: dict,
) -> None:
    """Qiskit's cancellation set is narrower, and the difference is declared.

    `CommutativeCancellation` cancels single-qubit rotations only, so `swap`,
    `ccx` and `cswap` pairs are outside what it attempts at all. Comparing the
    totals would dress a scope difference up as an algorithmic one, so the table
    is split: within `cx`, `cy` and `cz` the two remove the same number of gates,
    and the extra removals here are attributed to the opcodes only this pass
    handles.
    """

    anchor = payload["reference_anchor"]
    if not anchor["available"]:
        pytest.skip(f"Qiskit not importable: {anchor['reason']}")
    rows = {row["opcode"]: row for row in anchor["rows"]}
    assert set(rows) == set(_CONTROLLED)
    assert [row["opcode"] for row in anchor["rows"] if row["in_qiskit_scope"]] == list(
        _QISKIT_SCOPE
    )
    for opcode in _QISKIT_SCOPE:
        assert (
            rows[opcode]["port_removed_count"] == rows[opcode]["qiskit_removed_count"]
        ), opcode
        assert rows[opcode]["port_removed_count"] > 0, opcode
    assert anchor["in_scope_port_removed_count"] == (
        anchor["in_scope_qiskit_removed_count"]
    )
    # The out-of-scope removals are real and attributed; they are not claimed as a
    # win over Qiskit, because Qiskit was never asked the question.
    assert anchor["out_of_scope_port_removed_count"] > 0
    assert anchor["in_scope_port_removed_count"] > 0
    assert anchor["in_scope_opcodes"] == list(_QISKIT_SCOPE)
    assert _REGISTER_WIDTH >= 3
