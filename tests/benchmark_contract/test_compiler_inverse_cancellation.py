"""Contract for the inverse-declaration measurements W9-09 records.

Before this round, ``compiler.optimize`` cancelled a gate that is its own inverse
but never a pair of *different* opcodes that invert each other, so ``s(0) sdg(0)``
survived as two instructions although it is the identity. The benchmark module
answers four questions about the pass that closes that gap. Where does its rule
come from? Is the declaration it reads actually a rule? What does the pass remove
that the pipeline could not remove before? And where does it stop?

The recorded answers are that the rule is the ``OperatorSchema.adjoint``
declaration ``Circuit.adjoint`` already consumes, held by **6** opcodes in **3**
mutually-declaring pairs and disjoint from the pipeline's own self-inverse set;
that all **31** declared unitaries invert exactly, with a worst residual of
**3.14e-16** and a smallest wrong-partner residual of **1.22**; that the pass
removes **840** instructions on seeded populations where the pipeline without it
removes **0**; and that on 60 pair/gap rows it crosses an empty or provably
commuting gap **12** times, declines **20** rows that were in fact removable, and
never crosses a gap that does not commute.

These tests hold that evidence in place. They fail if the declaration stops being
read from the schema, if a wrong partner starts looking like an inverse, if the
pass starts or stops removing on a population, if a removal is no longer checked
against an execution, or if the refused rows stop being split into "correctness"
and "deferred reach" and are reported as one number.
"""

import pytest

from benchmarks.compiler_inverse_cancellation import (
    _GAPS,
    _REGISTER_WIDTH,
    _SWEEP_SEED,
    run_benchmark,
)
from flagquantum.compiler.inverse_cancellation import inverse_pairs
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS, inverse_operator

pytestmark = pytest.mark.benchmark_contract

#: The declaration inventory. Every number here is a property of the schema module
#: and the pass's scope, so it is a fixture rather than a measurement: if one moves,
#: the sweep below is describing a different operator set and the residual and
#: pipeline numbers are not comparable with it.
_DECLARATION = {
    "unitary_opcode_count": 31,
    "declared_inverse_count": 31,
    "partner_opcode_count": 6,
    "parameter_free_self_inverse_count": 11,
    "angle_rule_opcode_count": 14,
    "in_pipeline_self_inverse_count": 10,
    "scope_opcode_count": 6,
    "scope_pair_count": 3,
}

#: The one opcode that is its own inverse, carries no parameter, and is still not in
#: the pipeline's ``_SELF_INVERSE`` set. Reported rather than papered over: the two
#: sets are not the same set, and the difference is stated instead of implied.
_PIPELINE_SELF_INVERSE_GAP = ["i"]

#: Per-population instruction counts. ``removed_by_legacy`` is what the pipeline
#: removes with this pass taken out of it, so it must be zero on every population
#: that isolates the new rule; ``removed_by_declared_inverse_pass`` is the delta.
_DELTA = {
    "adjacent_pair_members": {
        "circuit_count": 60,
        "source_instruction_count": 480,
        "legacy_instruction_count": 480,
        "optimized_instruction_count": 0,
        "removed_by_declared_inverse_pass": 480,
        "removed_by_legacy": 0,
        "changed_circuit_count": 60,
        "executed_circuit_count": 60,
    },
    "pair_across_another_qubit": {
        "circuit_count": 60,
        "source_instruction_count": 360,
        "legacy_instruction_count": 360,
        "optimized_instruction_count": 240,
        "removed_by_declared_inverse_pass": 120,
        "removed_by_legacy": 0,
        "changed_circuit_count": 60,
        "executed_circuit_count": 60,
    },
    "interleaved_pairs": {
        "circuit_count": 60,
        "source_instruction_count": 420,
        "legacy_instruction_count": 420,
        "optimized_instruction_count": 180,
        "removed_by_declared_inverse_pass": 240,
        "removed_by_legacy": 0,
        "changed_circuit_count": 60,
        "executed_circuit_count": 60,
    },
    "pair_across_a_declared_barrier": {
        "circuit_count": 60,
        "source_instruction_count": 360,
        "legacy_instruction_count": 360,
        "optimized_instruction_count": 360,
        "removed_by_declared_inverse_pass": 0,
        "removed_by_legacy": 0,
        "changed_circuit_count": 0,
        "executed_circuit_count": 0,
    },
}

#: The pair/gap census. ``non_commuting_gap_removed_count`` is the correctness gate
#: and ``declined_but_removable_count`` is the deferred reach; they are separate
#: because a refusal that had to happen and a refusal that merely did happen are
#: different findings, and a single "declined" total would let one hide the other.
_SHAPES = {
    "gap_count": 10,
    "row_count": 60,
    "empty_gap_row_count": 6,
    "empty_gap_removed_count": 12,
    "commuting_gap_row_count": 32,
    "commuting_gap_removed_count": 24,
    "commuting_gap_removable_in_fact_count": 20,
    "non_commuting_gap_row_count": 22,
    "non_commuting_gap_removed_count": 0,
    "non_commuting_gap_removable_in_fact_count": 0,
    "unknowable_gap_row_count": 6,
    "declined_but_removable_count": 20,
    "reached_but_not_removable_in_fact_count": 0,
}

#: Two opcode/gap rows read back one at a time, because the pairs are not
#: interchangeable: ``s`` is diagonal and ``sx`` is not, so the same gap is a
#: commuter for one and a barrier for the other. A single averaged row would hide
#: exactly the distinction the table exists to draw.
_SPOT_ROWS = {
    ("s", "rz_on_the_pairs_own_qubit"): (True, 0, True),
    ("s", "rx_on_the_pairs_own_qubit"): (False, 0, False),
    ("sx", "rz_on_the_pairs_own_qubit"): (False, 0, False),
    ("sx", "rx_on_the_pairs_own_qubit"): (True, 0, True),
    ("sx", "x_on_the_pairs_own_qubit"): (True, 0, True),
    ("sx", "cx_target_on_the_pair_qubit"): (True, 0, True),
    ("t", "a_member_of_the_other_pair"): (True, 0, True),
    ("s", "a_member_of_the_other_pair"): (False, 0, False),
}

#: The rows that were removed adjacent to no gap at all, in both placements.
_ADJACENT_GAPS = ("nothing_between", "a_gate_on_another_qubit")


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_benchmark()


def _rows(payload: dict) -> dict[tuple[str, str], dict]:
    return {(row["opcode"], row["gap"]): row for row in payload["shape_table"]["rows"]}


def test_measurement_is_classified_as_a_local_microbenchmark(payload: dict) -> None:
    assert payload["schema"] == (
        "flagquantum_compiler_inverse_cancellation_benchmark_v1"
    )
    assert payload["artifact_classification"] == "local_compiler_microbenchmark"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert payload["seed"]["sweep"] == _SWEEP_SEED
    assert payload["reference_algorithm"] == "qiskit_inverse_cancellation"


def test_the_declaration_inventory_is_the_one_the_schema_holds(payload: dict) -> None:
    """The inventory is a property of the schema, and it is pinned as one."""

    declaration = payload["declaration_table"]
    for key, expected in _DECLARATION.items():
        assert declaration[key] == expected, key
    assert declaration["disjoint_from_self_inverse_pass"] is True
    assert declaration["pipeline_self_inverse_gap"] == _PIPELINE_SELF_INVERSE_GAP
    assert (
        declaration["in_pipeline_self_inverse_count"]
        < declaration["parameter_free_self_inverse_count"]
    )


def test_every_declared_unitary_has_a_declared_inverse(payload: dict) -> None:
    """The declaration table has no silent hole in it, and no channel in it either.

    The four non-unitary channels live in ``OPERATOR_SCHEMAS`` and declare
    ``not_applicable``, which is why they are absent here rather than present with a
    ``None`` partner: the inverse of a dissipative channel is not a gate the pipeline
    could emit, so listing them would make the table look like it had covered them.
    """

    rows = payload["declaration_table"]["rows"]
    unitary = {opcode for opcode, schema in OPERATOR_SCHEMAS.items() if schema.unitary}
    assert {row["opcode"] for row in rows} == unitary
    assert len(rows) == _DECLARATION["unitary_opcode_count"]
    assert len(OPERATOR_SCHEMAS) > len(rows)
    for row in rows:
        schema = OPERATOR_SCHEMAS[row["opcode"]]
        assert schema.unitary is True, row["opcode"]
        assert row["arity"] == schema.arity, row["opcode"]
        assert row["declared_partner"] is not None, row["opcode"]
        assert row["parameter_count"] == len(schema.parameters), row["opcode"]


def test_the_pass_scope_is_derived_from_the_schema_and_not_declared_twice(
    payload: dict,
) -> None:
    """No second source of truth: the scope is recomputed from the schema here.

    ``inverse_pairs`` is the only thing the pass reads, and this recomputes the same
    relation straight from ``OPERATOR_SCHEMAS``. If the pass ever grew its own
    opcode list the two would stop being equal, which is the failure this test is
    for.
    """

    derived = {}
    for opcode, schema in sorted(OPERATOR_SCHEMAS.items()):
        if not schema.unitary or schema.parameters:
            continue
        declared = inverse_operator(schema, {})
        if declared is None or declared[0] == opcode:
            continue
        derived[opcode] = declared[0]
    assert dict(inverse_pairs()) == derived
    assert len(derived) == _DECLARATION["scope_opcode_count"]
    # Mutually declaring, so the count is even and the relation is an involution.
    for opcode, partner in derived.items():
        assert derived[partner] == opcode, opcode
    rows = payload["declaration_table"]["rows"]
    in_scope = {row["opcode"] for row in rows if row["in_this_pass_scope"]}
    assert in_scope == set(derived)


def test_the_self_inverse_pass_and_this_pass_do_not_overlap(payload: dict) -> None:
    """The two passes partition the parameter-free unitary opcodes.

    If they overlapped, the delta measured below would be double-counting whichever
    pass saw the pair first, and the pipeline's fixed point would depend on the
    order the two are called in rather than on the rules they hold.
    """

    rows = payload["declaration_table"]["rows"]
    scope = {row["opcode"] for row in rows if row["in_this_pass_scope"]}
    self_inverse = {row["opcode"] for row in rows if row["in_self_inverse_pass"]}
    assert scope & self_inverse == set()
    assert len(scope) + len(self_inverse) < _DECLARATION["unitary_opcode_count"]


def test_the_declaration_is_a_rule_and_not_an_average(payload: dict) -> None:
    """Every declared inverse annihilates its gate, over seeded draws.

    The matrices are the runtime's own, so this is an exact comparison and not a
    tolerance study. The point of the assertion is the size of the residual: a
    declaration that only held approximately would make this pass a rule source of
    last resort, which is a different and weaker thing than a rule source of proof.
    """

    proof = payload["identity_proof"]
    assert proof["checked_opcode_count"] == 16
    assert proof["draw_count_per_opcode"] == 20
    assert proof["max_residual"] < 1.0e-12
    for row in proof["residual_rows"]:
        assert row["residual"] < 1.0e-12, row["opcode"]


def test_the_same_measurement_rejects_a_wrong_partner(payload: dict) -> None:
    """The control that stops the residual above from being vacuous.

    A residual of zero means nothing unless the same comparison can produce a
    nonzero one. Every declared pair is therefore also multiplied against a
    parameter-free unitary that is *not* its inverse, and the smallest such residual
    is reported: if that number ever fell near the tolerance, the check above would
    have stopped discriminating.
    """

    proof = payload["identity_proof"]
    assert proof["control_opcode_count"] == _DECLARATION["scope_opcode_count"]
    assert proof["control_min_residual"] > 1.0
    assert proof["control_min_residual"] > proof["max_residual"] * 1.0e12
    for row in proof["control_rows"]:
        assert row["partner"] != row["opcode"], row["opcode"]
        assert row["residual"] > 1.0, row["opcode"]


def test_every_population_reports_its_own_removal_count(payload: dict) -> None:
    rows = {row["label"]: row for row in payload["pipeline_delta"]}
    assert set(rows) == set(_DELTA)
    for label, expected in _DELTA.items():
        for key, value in expected.items():
            assert rows[label][key] == value, (label, key, rows[label][key])


def test_the_delta_belongs_to_this_pass_and_not_to_the_pipeline(payload: dict) -> None:
    """The control that makes the delta attributable.

    The comparison pipeline is the real one with this pass removed and nothing else
    changed, so ``removed_by_legacy`` is a measurement of the rest of the pipeline
    and not an assumption about it. It is zero everywhere here, which is what makes
    the whole delta the new rule's.
    """

    rows = {row["label"]: row for row in payload["pipeline_delta"]}
    for label, row in rows.items():
        assert row["removed_by_legacy"] == 0, label
        assert row["source_instruction_count"] == row["legacy_instruction_count"], label
        assert row["removed_by_declared_inverse_pass"] == (
            row["legacy_instruction_count"] - row["optimized_instruction_count"]
        ), label
    total = sum(row["removed_by_declared_inverse_pass"] for row in rows.values())
    assert total == 840


def test_removing_every_instruction_preserved_the_program(payload: dict) -> None:
    """Every removed instruction was paid for with an execution.

    The three gate-only populations are exact, and the count of circuits actually
    executed is reported alongside the count of circuits that changed. A population
    that quietly stopped being executable would show up as a shrinking denominator
    rather than as a pass -- the barrier population is that case, honestly at zero,
    because a declared barrier makes the circuit unexecutable as a gate-only program
    and there is nothing there to check.
    """

    rows = {row["label"]: row for row in payload["pipeline_delta"]}
    for label in (
        "adjacent_pair_members",
        "pair_across_another_qubit",
        "interleaved_pairs",
    ):
        row = rows[label]
        assert row["changed_circuit_count"] == row["circuit_count"], label
        assert row["executed_circuit_count"] == row["circuit_count"], label
        assert row["max_state_difference"] == 0.0, label
    barrier = rows["pair_across_a_declared_barrier"]
    assert barrier["changed_circuit_count"] == 0
    assert barrier["executed_circuit_count"] == 0
    assert barrier["optimized_instruction_count"] == barrier["source_instruction_count"]


def test_the_shape_census_is_the_one_the_pass_currently_produces(payload: dict) -> None:
    shapes = payload["shape_table"]
    for key, expected in _SHAPES.items():
        assert shapes[key] == expected, key
    assert len(shapes["rows"]) == _SHAPES["row_count"]
    assert len(_GAPS) == _SHAPES["gap_count"]


def test_no_pair_was_removed_across_a_gap_that_does_not_commute(payload: dict) -> None:
    """The one property that cannot be traded away for reach.

    A pair crossed over a gap that does not commute would be a program change, not
    an optimization. ``gap_commutes_with_member`` is measured per row against the
    runtime matrices, so this is a correctness gate over the driven shapes rather
    than a restatement of the pass's own rule.
    """

    shapes = payload["shape_table"]
    assert shapes["non_commuting_gap_removed_count"] == 0
    assert shapes["reached_but_not_removable_in_fact_count"] == 0
    rows = _rows(payload)
    for (opcode, gap), row in rows.items():
        if row["gap_commutes_with_member"] is False:
            assert row["removed"] == 0, (opcode, gap)
            assert row["removable_in_fact"] is not True, (opcode, gap)


def test_every_refused_shape_is_split_into_correctness_and_deferred_reach(
    payload: dict,
) -> None:
    """The refused rows are two different findings and are counted separately.

    Rows whose gap does not commute were not removable in fact -- declining them was
    the only correct answer, and there are 22 of them. Rows whose gap commutes were
    frequently removable and were declined anyway: that is this pass's deferred
    reach, 20 rows, and it is reported as a number instead of as a promise. If a
    later round teaches the pass to reach across a proven gap, this is the row that
    has to move.
    """

    shapes = payload["shape_table"]
    assert shapes["non_commuting_gap_removable_in_fact_count"] == 0
    assert shapes["non_commuting_gap_row_count"] == 22
    assert shapes["declined_but_removable_count"] == 20
    assert shapes["declined_but_removable_count"] == (
        shapes["commuting_gap_removable_in_fact_count"]
    )
    rows = _rows(payload)
    declined = [row for row in rows.values() if row["removable_in_fact"] is True]
    assert len(declined) == shapes["declined_but_removable_count"]
    for row in declined:
        assert row["gap_commutes_with_member"] is True, (row["opcode"], row["gap"])
        assert row["removed"] == 0, (row["opcode"], row["gap"])


def test_a_shape_is_measured_both_ways_round(payload: dict) -> None:
    """One gap is a commuter for one opcode and a barrier for another.

    ``s`` is diagonal and ``sx`` is a half turn about the same axis as ``rx``, so
    ``rz`` commutes with the first and not the second while ``rx`` does the
    reverse. Reading these rows back one at a time is what stops the census above
    from being satisfiable by a table that labelled every row the same way.
    """

    rows = _rows(payload)
    for (opcode, gap), (commutes, removed, removable) in _SPOT_ROWS.items():
        row = rows[(opcode, gap)]
        assert row["gap_commutes_with_member"] is commutes, (opcode, gap)
        assert row["removed"] == removed, (opcode, gap)
        assert row["removable_in_fact"] is removable, (opcode, gap)
    assert rows[("s", "rz_on_the_pairs_own_qubit")]["gap_commutes_with_member"] != (
        rows[("sx", "rz_on_the_pairs_own_qubit")]["gap_commutes_with_member"]
    )


def test_a_gap_that_cannot_be_asked_about_is_not_counted_as_a_no(payload: dict) -> None:
    """A question that cannot be asked is not an answer of "no".

    The ``measure`` gap is not a declared gate, so neither column can be computed
    for it. Both are reported as ``None`` and the rows are counted apart from the
    refusals, because folding them into "declined" would let an unanswerable row
    inflate the correctness count.
    """

    shapes = payload["shape_table"]
    assert shapes["unknowable_gap_row_count"] == 6
    assert shapes["unknowable_gap_row_count"] == _DECLARATION["scope_opcode_count"]
    rows = _rows(payload)
    for opcode in sorted(inverse_pairs()):
        row = rows[(opcode, "measure_on_the_pairs_own_qubit")]
        assert row["gap_commutes_with_member"] is None, opcode
        assert row["removable_in_fact"] is None, opcode
        assert row["removed"] == 0, opcode
    assert shapes["unmeasurable_row_count"] == (
        shapes["unknowable_gap_row_count"]
        + sum(1 for row in rows.values() if row["gap"] in _ADJACENT_GAPS)
    )


def test_the_adjacent_gap_is_removed_for_every_pair_in_both_orders(
    payload: dict,
) -> None:
    rows = _rows(payload)
    for opcode in sorted(inverse_pairs()):
        for gap in _ADJACENT_GAPS:
            row = rows[(opcode, gap)]
            assert row["removed"] == 2, (opcode, gap)
            assert row["optimized_instruction_count"] == (
                row["source_instruction_count"] - 2
            ), (opcode, gap)
            assert row["gap_commutes_with_member"] is True, (opcode, gap)


def test_the_qiskit_anchor_compares_only_what_both_passes_attempt(
    payload: dict,
) -> None:
    """Qiskit was driven over the identical circuits, and the anchor can be absent.

    ``InverseCancellation`` takes the inverse pairs as an argument, exactly as this
    port does, so the two are answering the same question here and the per-row
    counts are directly comparable. The anchor is a cross-check rather than a
    dependency and the module has to run where Qiskit is not installed, so an
    unavailable anchor is a skip and not a failure -- but a ``qiskit_removed`` of
    zero on every row would mean the anchor was constructed wrong, so the totals are
    asserted nonzero as well.
    """

    shapes = payload["shape_table"]
    anchor = shapes["reference_anchor"]
    if not anchor["available"]:
        pytest.skip(f"Qiskit not importable: {anchor['reason']}")
    assert anchor["pass_name"] == "InverseCancellation"
    assert anchor["removed_count"] > 0
    assert anchor["source_gate_count"] == sum(
        row["source_instruction_count"] for row in shapes["rows"]
    )
    assert payload["anchor_disagreement_count"] == 0
    assert payload["anchor_agreement_count"] == _SHAPES["row_count"]
    assert "only over rows actually driven" in payload["anchor_agreement_scope"]
    for row in shapes["rows"]:
        assert row["removed"] == row["qiskit_removed"], (row["opcode"], row["gap"])
        assert row["agrees_with_qiskit"] is True, (row["opcode"], row["gap"])
    # And the agreement is not agreement about nothing: both ports remove on the
    # adjacent rows and both decline the unanswerable one.
    assert anchor["removed_by_shape"]["s|nothing_between"] == 2
    assert anchor["removed_by_shape"]["s|measure_on_the_pairs_own_qubit"] == 0
    assert _REGISTER_WIDTH >= 3
