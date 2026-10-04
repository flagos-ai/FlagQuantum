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

import contextlib
import random

import pytest

from benchmarks.compiler_commutation_cancellation import (
    _CONTROLLED,
    _QISKIT_SCOPE,
    _REGISTER_WIDTH,
    _SWEEP_SEED,
    run_benchmark,
)
from benchmarks.compiler_identity_elimination import (
    _CIRCUITS_PER_POPULATION as IDENTITY_CIRCUITS_PER_POPULATION,
)
from benchmarks.compiler_identity_elimination import (
    _POPULATIONS as IDENTITY_POPULATIONS,
)
from benchmarks.compiler_identity_elimination import (
    _substituted_rule,
    _superseded_rule,
)
from benchmarks.compiler_two_qubit_optimization import (
    _fold_patch,
    _rotation_merge_patch,
)
from flagquantum.compiler.pipeline import optimize

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
    # The two populations added for the rotation merge. `removed_by_legacy` is zero
    # in both, so everything the fixed point removes here is a commuting rule's work.
    "rotation_chain_on_controls": {
        "circuit_count": 40,
        "source_instruction_count": 680,
        "legacy_instruction_count": 680,
        "optimized_instruction_count": 554,
        "removed_by_commutation": 126,
        "changed_circuit_count": 40,
    },
    "rotation_chain_mixed_placement": {
        "circuit_count": 40,
        "source_instruction_count": 680,
        "legacy_instruction_count": 680,
        "optimized_instruction_count": 561,
        "removed_by_commutation": 119,
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

    Two of the four original populations are exact: the removed gates are their own
    inverses and the gap between them provably commutes, so the statevector is
    bit-identical rather than merely close. The chained population is the one with a
    nonzero difference, and the substitution measurement in `rotation_merge_reach`
    says it is not this rule's: the rotation merge removes nothing there. What the
    difference is, is an angle sum -- once the pairs cancel, two `rz`s end up
    adjacent and the adjacent merge adds their angles in floating point, where the
    source applied them one after the other.
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


#: The rotation-merge table's pinned reading, placement by placement. `merged` and
#: `commutes` are pinned together because the claim this round makes is that the
#: pass takes the whole of the reach the rule source proves on this shape and none
#: of the reach it does not; a row where the two disagree would be a wrong rewrite,
#: which is why the contract pins the pairs rather than only the totals.
_MERGED_PLACEMENTS = {
    "cx": (True, False),
    "cy": (True, False),
    "cz": (True, True),
    "swap": (False, False),
    "ccx": (True, True, False),
    "cswap": (True, False, False),
}

#: Where each rotation merges across each entangler. The diagonal opcodes reach
#: every placement the rule source proves; `rx` and `ry` are in a Pauli family with
#: exactly one entangler each (`cx` and `cy`), so they merge on that entangler's
#: wire of the same family and nowhere else. A table that only reported the total
#: would hide that boundary.
_MERGED_BY_OPCODE = {"rx": 2, "ry": 1, "rz": 7, "phase": 7, "u1": 7}


def test_the_rotation_merge_table_is_not_vacuous(payload: dict) -> None:
    """A table of refusals would look as clean as a table of merges."""

    merge = payload["rotation_merge_table"]
    assert merge["rotation_opcodes"] == ["rx", "ry", "rz", "phase", "u1"]
    assert merge["entangler_opcodes"] == list(_CONTROLLED)
    assert merge["row_count"] == len(merge["rotation_opcodes"]) * sum(
        len(_MERGED_PLACEMENTS[opcode]) for opcode in _CONTROLLED
    )
    assert merge["row_count"] == 70
    assert merge["merged_count"] == 24
    assert merge["merged_by_opcode"] == _MERGED_BY_OPCODE


def test_no_rotation_is_merged_without_a_commutation_proof(payload: dict) -> None:
    """The one direction that must never move: a merge the rule source refuses."""

    merge = payload["rotation_merge_table"]
    assert merge["merged_without_a_proof_count"] == 0
    for row in merge["rows"]:
        assert row["merged"] == row["commutes"], row
        if row["merged"]:
            assert row["optimized_instruction_count"] == (
                row["source_instruction_count"] - 1
            ), row


def test_the_rotation_merge_takes_the_whole_proven_reach_on_this_shape(
    payload: dict,
) -> None:
    """`proven_but_unmerged_count` is reach this round does not yet take.

    It is zero here, and pinning it at zero is the claim: on a three-instruction
    shape with one gap, every placement the rule source proves is merged. A later
    change that started declining one of these would move this count and fail here,
    which is the point of reporting it at all.
    """

    merge = payload["rotation_merge_table"]
    assert merge["proven_commuting_count"] == merge["merged_count"] == 24
    assert merge["proven_but_unmerged_count"] == 0


def test_the_merged_placements_are_the_commutation_proof_read_back(
    payload: dict,
) -> None:
    """The rows restate `commute` position by position, entangler by entangler.

    `_MERGED_PLACEMENTS` is `_GAP_POSITIONS` for the diagonal rotations, and the
    two tables agree because they are the same proof. Where they would differ is a
    row where the pass merged what the rule source declines, which is exactly what
    the previous test is for; this one checks that the diagonal rows really do
    cover the whole of the proven set rather than a chosen subset of it.
    """

    merge = payload["rotation_merge_table"]
    assert _MERGED_PLACEMENTS == _GAP_POSITIONS
    for row in merge["rows"]:
        if row["rotation_opcode"] not in {"rz", "phase", "u1"}:
            continue
        expected = _MERGED_PLACEMENTS[row["entangler_opcode"]][
            row["rotation_wire_index"]
        ]
        assert row["merged"] is expected, row
        assert row["commutes"] is expected, row


def test_every_merged_rotation_preserved_the_program(payload: dict) -> None:
    """Each merge is checked against an execution, not against a second opinion."""

    merge = payload["rotation_merge_table"]
    assert merge["worst_merged_state_difference"] < 1.0e-12
    merged_rows = [row for row in merge["rows"] if row["merged"]]
    assert merged_rows
    assert max(row["max_state_difference"] for row in merged_rows) == (
        merge["worst_merged_state_difference"]
    )
    # A refusal must not report a movement at all: the columns mean different
    # things and the table must not blur them.
    for row in merge["rows"]:
        if not row["merged"]:
            assert row["max_state_difference"] == 0.0, row


def test_the_pauli_family_boundary_is_where_the_table_says_it_is(payload: dict) -> None:
    """`rx` and `ry` merge far less often, and the reason is stated rather than hidden.

    Both are half turns in a Pauli algebra: `rx` commutes with `cx` on the target,
    `ry` with `cy`. Neither commutes with the diagonal entanglers on any wire, and
    the two diagonal-only entanglers `swap`/`cswap` commute with nothing here. So
    the diagonal rotations reach seven placements each and these two reach two and
    one. The totals are the finding; reporting only the sum would report the
    diagonal case and hide the other two.
    """

    merge = payload["rotation_merge_table"]
    by_opcode = merge["merged_by_opcode"]
    for diagonal in ("rz", "phase", "u1"):
        assert by_opcode[diagonal] == 7, diagonal
    assert by_opcode["rx"] == 2
    assert by_opcode["ry"] == 1
    rows = {
        (
            row["rotation_opcode"],
            row["entangler_opcode"],
            row["rotation_wire_index"],
        ): row
        for row in merge["rows"]
    }
    assert rows[("rx", "cx", 1)]["merged"]
    assert rows[("ry", "cy", 1)]["merged"]
    assert not rows[("rx", "cz", 0)]["merged"]
    assert not rows[("ry", "cx", 1)]["merged"]


def test_the_rotation_merge_reach_is_attributed_by_substitution(payload: dict) -> None:
    """`removed_by_commutation` sums both commuting rules, so it cannot attribute.

    The reach table substitutes `merge_commuting_rotations` with the identity and
    re-runs the whole fixed point, which is the only measurement here that separates
    the newer rule from the older one. The substitution has to be real for the zeros
    below to mean anything, so the two non-zero rows are part of this test rather
    than a separate one: a patch that silently failed to apply would show up as six
    zeros and this test would pass on four of them.
    """

    reach = {row["label"]: row for row in payload["rotation_merge_reach"]}
    assert set(reach) == {
        row["label"] for row in payload["pipeline_delta"]
    }, "the two tables have to describe the same populations"
    for label in ("gap_on_wire_0", "gap_on_wire_1", "gap_on_unrelated_qubit"):
        assert reach[label]["removed_by_rotation_merge"] == 0, label
    assert reach["chained_control_gaps"]["removed_by_rotation_merge"] == 0
    assert reach["rotation_chain_on_controls"]["removed_by_rotation_merge"] == 74
    assert reach["rotation_chain_mixed_placement"]["removed_by_rotation_merge"] == 95


def test_the_rotation_merge_reach_is_invisible_on_the_older_populations(
    payload: dict,
) -> None:
    """The finding, stated as a number rather than as prose.

    None of the four populations this benchmark had before this round shows any
    reach for the new rule, and that is not a defect in the rule: they carry one
    rotation each, or two identical entanglers next to each other, so the adjacent
    passes empty the gap before either commuting rule is asked. The two rotation
    chains were added because the older populations cannot see the rule at all.
    """

    reach = {row["label"]: row for row in payload["rotation_merge_reach"]}
    assert sum(row["removed_by_rotation_merge"] for row in reach.values()) == 169
    assert reach["rotation_chain_on_controls"]["shipped_instruction_count"] < (
        reach["rotation_chain_on_controls"]["without_rotation_merge_instruction_count"]
    )
    assert reach["rotation_chain_mixed_placement"]["shipped_instruction_count"] < (
        reach["rotation_chain_mixed_placement"][
            "without_rotation_merge_instruction_count"
        ]
    )
    for row in payload["rotation_merge_reach"]:
        assert row["max_state_difference"] < 1.0e-12, row


def test_the_rotation_chain_populations_are_not_vacuous(payload: dict) -> None:
    """The two new populations carry the shapes their names claim."""

    delta = {row["label"]: row for row in payload["pipeline_delta"]}
    for label in ("rotation_chain_on_controls", "rotation_chain_mixed_placement"):
        row = delta[label]
        assert row["circuit_count"] == 40, label
        assert row["source_instruction_count"] == 680, label
        assert row["removed_by_legacy"] == 0, label
        assert row["changed_circuit_count"] == 40, label
        assert row["rotation_group_count"] > 0, label
        assert row["rotation_position_count"] > 0, label
    # Measured, and deliberately not explained here: the mixed population merges
    # more rotations (95 against 74) even though only half of its entanglers put the
    # rotation wire in a proven-commuting position. A comment claiming the obvious
    # ordering would be wrong, so the two counts are pinned instead.
    assert delta["rotation_chain_on_controls"]["removed_by_commutation"] == 126
    assert delta["rotation_chain_mixed_placement"]["removed_by_commutation"] == 119
    assert delta["rotation_chain_on_controls"]["rotation_group_count"] == 83
    assert delta["rotation_chain_mixed_placement"]["rotation_group_count"] == 97


def test_the_anchor_reports_the_ports_own_column_without_qiskit(payload: dict) -> None:
    """The port's half of the anchor, which no optional dependency may hide.

    `CommutativeCancellation` merges a run of z-rotations in the same call, so
    ``rz(a) cx rz(b)`` on the control is a rewrite both implementations attempt. The
    port's column is a fact about this repository, so it is measured and pinned
    whether or not Qiskit is importable, and the same rows carry Qiskit's column when
    it is.

    This split exists because of a measured defect, not for tidiness. The whole anchor
    used to sit behind the import, so on the job that runs this contract -- which has
    no Qiskit -- every number in it was skipped, and the one Qiskit job in CI does not
    collect `tests/benchmark_contract/**` at all. A pinned total of the port's own
    column therefore shipped this round that no job in the repository could falsify;
    installing Qiskit and running the anchor by hand gave 14 where the pin said 15.
    Anything a reader can reproduce without a second framework is now outside the
    skip, and the skip covers exactly the column that genuinely needs Qiskit.
    """

    anchor = payload["reference_anchor"]
    rows = anchor["rotation_rows"]
    assert anchor["rotation_row_count"] == len(_QISKIT_SCOPE) * 2
    assert {(row["opcode"], row["wire_index"]) for row in rows} == {
        (opcode, wire_index) for opcode in _QISKIT_SCOPE for wire_index in (0, 1)
    }
    for row in rows:
        assert row["source_gate_count"] == 3, row
        merged = _MERGED_PLACEMENTS[row["opcode"]][row["wire_index"]]
        assert row["port_optimized_gate_count"] == (2 if merged else 3), row
    assert anchor["rotation_source_gate_count"] == 3 * anchor["rotation_row_count"]
    assert anchor["rotation_port_optimized_gate_count"] == sum(
        row["port_optimized_gate_count"] for row in rows
    )
    # The port merges on four of the six placements, which is the whole of what the
    # rule source proves about a `cx`, `cy` or `cz` on two wires: both placements of
    # `cz`, and the control of `cx` and of `cy`.
    assert anchor["rotation_port_optimized_gate_count"] == 14
    # The total above and the per-placement table are two statements of one fact, so
    # they are tied together by an equation rather than left free to drift apart: the
    # instructions the port removed are exactly the merged placements.
    assert anchor["rotation_source_gate_count"] - anchor[
        "rotation_port_optimized_gate_count"
    ] == sum(
        merged for opcode in _QISKIT_SCOPE for merged in _MERGED_PLACEMENTS[opcode]
    )
    assert {opcode for opcode in _QISKIT_SCOPE if any(_MERGED_PLACEMENTS[opcode])} == {
        "cx",
        "cy",
        "cz",
    }


def test_the_anchor_asks_qiskit_the_rotation_question_too(payload: dict) -> None:
    """Qiskit's half of the same rows, bounded rather than pinned.

    Qiskit's column is reported and bounded rather than pinned to a number, because a
    pinned number nobody can reproduce is exactly the kind of claim this repository
    forbids, and the CI job that has Qiskit does not collect this directory. What is
    asserted about it is only what a merge can do: it never lengthens the circuit and
    it never deletes it.
    """

    anchor = payload["reference_anchor"]
    if not anchor["available"]:
        pytest.skip(f"Qiskit not importable: {anchor['reason']}")
    rows = anchor["rotation_rows"]
    for row in rows:
        assert 0 < row["qiskit_optimized_gate_count"] <= row["source_gate_count"], row
    assert anchor["rotation_qiskit_optimized_gate_count"] == sum(
        row["qiskit_optimized_gate_count"] for row in rows
    )


#: The rotation merge's reach on the seven populations `compiler_identity_elimination`
#: publishes. Each entry is ``label -> ((superseded identity rule, shipped identity
#: rule) with the merge paused, ... with it active)``, and the two tables below are the
#: same measurement with the fold out and with the fold in. Holding the fold in both
#: positions is what keeps the two co-owners of these rows apart: the fold is the table
#: and the rotation merge is the second element of each pair against the first.
#:
#: The rows are re-measured rather than read from the identity benchmark's payload,
#: because that payload is the shipped pipeline in which both passes are active, and a
#: single number cannot say which of them moved it. This is the same substitution
#: instrument the two-qubit contract uses for the fold, pointed at the other pass.
_ROTATION_MERGE_IDENTITY_REACH = {
    "parameter_free_gates": ((167, 167), (167, 167)),
    "zero_angle_rotations": ((162, 162), (162, 162)),
    "full_turn_rotations": ((139, 150), (139, 150)),
    "zero_polar_u3": ((176, 176), (176, 176)),
    "two_wire_rotations": ((245, 245), (244, 244)),
    "mixed": ((406, 391), (404, 390)),
    "mixed_with_mid_circuit_measures": ((511, 498), (510, 497)),
}
#: The same seven rows with the fold as shipped, which is the pipeline those rows
#: actually describe and therefore the pair of numbers the identity benchmark's own
#: pins carry. Two of the three moved when the fold widened its membership from an
#: all-two-qubit run to a block that also draws in the single-qubit gates on the pair's
#: wires; the other five did not move at all, which is what makes this a re-measurement
#: of a co-owned row rather than a re-baselining of the table.
_ROTATION_MERGE_IDENTITY_REACH_WITH_THE_FOLD = {
    "parameter_free_gates": ((167, 167), (167, 167)),
    "zero_angle_rotations": ((162, 162), (162, 162)),
    "full_turn_rotations": ((139, 150), (139, 150)),
    "zero_polar_u3": ((176, 176), (176, 176)),
    "two_wire_rotations": ((201, 201), (200, 200)),
    "mixed": ((393, 383), (392, 382)),
    "mixed_with_mid_circuit_measures": ((499, 492), (498, 491)),
}
#: The rows this pass moves, which are exactly the three the fold also co-owns. That
#: the sets coincide is the point rather than a coincidence of naming: both passes
#: compose rotations, so both have reach in the populations built out of them.
_ROTATION_MERGE_IDENTITY_ROWS_MOVED = (
    "two_wire_rotations",
    "mixed",
    "mixed_with_mid_circuit_measures",
)
#: What this pass is responsible for, per table and per population, as ``paused minus
#: active`` on each column. It is pinned per table because the two tables are different
#: pipelines rather than two readings of one.
#:
#: On ``mixed`` the two tables disagree, and the disagreement is the measurement rather
#: than noise. With the fold as it shipped before this round, that row's superseded
#: column moved by two while its shipped column moved by one: the fold and this merge
#: were each taking one of the two placements, and the identity rule's substitution is
#: what let both be visible at once. Once the fold drew single-qubit members into its
#: blocks it took both, and this pass is left with one on each column -- the same one it
#: has everywhere else. The fold earned five more instructions on that row in the same
#: move, so the row is strictly shorter; the asymmetry going away is a consequence of
#: the fold reaching further, and it is recorded here instead of being smoothed into a
#: rule that would no longer describe either pipeline.
_ROTATION_MERGE_DELTA = {
    "fold_out": {
        "two_wire_rotations": (1, 1),
        "mixed": (2, 1),
        "mixed_with_mid_circuit_measures": (1, 1),
    },
    "fold_in": {
        "two_wire_rotations": (1, 1),
        "mixed": (1, 1),
        "mixed_with_mid_circuit_measures": (1, 1),
    },
}


def test_the_identity_rows_the_rotation_merge_moved_are_attributed() -> None:
    """The second co-owner of the round 18 identity rows, measured on its own.

    ``compiler_identity_elimination`` publishes seven populations measured through the
    whole pipeline, so every pass landed after it owns a share of those numbers. The
    fold's share is attributed by pausing the fold; this pass has no such test in its
    own module, so without this one its contribution to those rows would be inferred
    from arithmetic over two numbers, neither of which it owns.

    Pausing it moves exactly the three rows the fold also moves, by the same amount in
    both fold positions, one instruction off the optimized pipeline and never a
    lengthening. Pinning the whole four-cell table rather than the delta alone is
    deliberate: the delta is the claim, and the table is what keeps the claim auditable
    when either pass changes.
    """

    measured: dict[str, dict[tuple[str, bool], tuple[int, int]]] = {}
    for label, factory in IDENTITY_POPULATIONS:
        circuits = [
            factory(random.Random(seed))
            for seed in range(IDENTITY_CIRCUITS_PER_POPULATION)
        ]
        for fold_policy in ("off", "shipped"):
            for merge_paused in (True, False):
                with contextlib.ExitStack() as stack:
                    stack.enter_context(_fold_patch(fold_policy))
                    if merge_paused:
                        stack.enter_context(_rotation_merge_patch())
                    with _substituted_rule(_superseded_rule):
                        superseded = sum(len(optimize(circuit)) for circuit in circuits)
                    shipped = sum(len(optimize(circuit)) for circuit in circuits)
                measured.setdefault(label, {})[(fold_policy, merge_paused)] = (
                    superseded,
                    shipped,
                )

    for label, (paused, active) in _ROTATION_MERGE_IDENTITY_REACH.items():
        assert measured[label][("off", True)] == paused, label
        assert measured[label][("off", False)] == active, label
    for label, (paused, active) in _ROTATION_MERGE_IDENTITY_REACH_WITH_THE_FOLD.items():
        assert measured[label][("shipped", True)] == paused, label
        assert measured[label][("shipped", False)] == active, label

    moved = {
        label
        for label, (
            paused,
            active,
        ) in _ROTATION_MERGE_IDENTITY_REACH_WITH_THE_FOLD.items()
        if paused != active
    }
    assert moved == set(_ROTATION_MERGE_IDENTITY_ROWS_MOVED)
    # Under the fold out it is the same three rows, so the two passes are not silently
    # trading the reach between them.
    assert {
        label
        for label, (paused, active) in _ROTATION_MERGE_IDENTITY_REACH.items()
        if paused != active
    } == moved
    for name, table in (
        ("fold_out", _ROTATION_MERGE_IDENTITY_REACH),
        ("fold_in", _ROTATION_MERGE_IDENTITY_REACH_WITH_THE_FOLD),
    ):
        for label in moved:
            (paused_superseded, paused_shipped), (active_superseded, active_shipped) = (
                table[label]
            )
            assert active_superseded <= paused_superseded, (name, label)
            assert active_shipped <= paused_shipped, (name, label)
            assert (
                paused_superseded - active_superseded,
                paused_shipped - active_shipped,
            ) == _ROTATION_MERGE_DELTA[name][label], (name, label)
    # Non-vacuity: four of the seven rows are populations this pass cannot reach at
    # all, so the table is not three measurements of one behaviour.
    assert len(_ROTATION_MERGE_IDENTITY_REACH) - len(moved) == 4
