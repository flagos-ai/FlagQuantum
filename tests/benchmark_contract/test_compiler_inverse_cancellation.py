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
**2.22e-16** and a smallest wrong-partner residual of **1.22**; that the pipeline
the branch merges into carries a same-wire one-qubit fold which reaches the
adjacent pairs on its own, so this pass's **marginal** compiler delta is **0** on
those populations and **+60** / **-60** on the two populations that frame a pair
inside the half-pi pulse basis, where it reaches **300** and **120** fewer native
gates than the fold alone although one of the two is a *longer* compiler program;
and that on 60 pair/gap rows it crosses an empty or provably commuting gap **12**
times, declines **20** rows that were in fact removable, and never crosses a gap
that does not commute.

These tests hold that evidence in place. They fail if the declaration stops being
read from the schema, if a wrong partner starts looking like an inverse, if the
pass starts or stops removing on a population, if a removal is no longer checked
against an execution, if the comparison pipeline stops being the shipped pipeline
minus this pass, if the native column stops being attributable, or if the refused
rows stop being split into "correctness" and "deferred reach" and are reported as
one number.
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

#: Per-population instruction counts. The comparison pipeline is the shipped
#: pipeline with this pass removed, so ``removed_by_rest_of_pipeline`` is the
#: one-qubit fold's own reach and ``removed_by_declared_inverse_pass`` is the
#: marginal delta this pass contributes on top of it -- zero on the four populations
#: the fold already reaches, and nonzero only on the two that frame a pair in the
#: half-pi pulse basis, in opposite directions. The native counts are the two
#: programs lowered into one declared basis (`rz` plus the half-pi pulse), which is
#: where the pass is visible even on the rows whose compiler delta is zero; the
#: barrier population's counts are ``None`` because a `measure` cannot be lowered
#: into that basis at all.
#:
#: The ``fold_first_*`` fields are the same populations through the opposite pass
#: order, and their shape is the honest result rather than a second copy: with the
#: fold ahead of this pass the marginal delta is zero everywhere and every count
#: equals that population's ``without_pass`` count, because the fold-first pipeline
#: *is* the comparison baseline. What the order changes is the native column, and it
#: changes it against the fold (690 and 736 against 600 and 540, 360 and 240 against
#: 60 and 120). Both directions are pinned, so a change to either order has to move a
#: recorded number instead of passing quietly.
_DELTA = {
    "adjacent_pair_members": {
        "circuit_count": 60,
        "source_instruction_count": 480,
        "without_pass_instruction_count": 0,
        "with_pass_instruction_count": 0,
        "removed_by_rest_of_pipeline": 480,
        "removed_by_declared_inverse_pass": 0,
        "fold_first_instruction_count": 0,
        "removed_by_declared_inverse_pass_fold_first": 0,
        "native_gate_count_without_pass": 0,
        "native_gate_count_with_pass": 0,
        "native_gate_count_fold_first": 0,
        "changed_circuit_count": 0,
        "executed_circuit_count": 60,
    },
    "pair_across_another_qubit": {
        "circuit_count": 60,
        "source_instruction_count": 360,
        "without_pass_instruction_count": 180,
        "with_pass_instruction_count": 180,
        "removed_by_rest_of_pipeline": 180,
        "removed_by_declared_inverse_pass": 0,
        "fold_first_instruction_count": 180,
        "removed_by_declared_inverse_pass_fold_first": 0,
        "native_gate_count_without_pass": 690,
        "native_gate_count_with_pass": 600,
        "native_gate_count_fold_first": 690,
        "changed_circuit_count": 0,
        "executed_circuit_count": 60,
    },
    "interleaved_pairs": {
        "circuit_count": 60,
        "source_instruction_count": 420,
        "without_pass_instruction_count": 180,
        "with_pass_instruction_count": 180,
        "removed_by_rest_of_pipeline": 240,
        "removed_by_declared_inverse_pass": 0,
        "fold_first_instruction_count": 180,
        "removed_by_declared_inverse_pass_fold_first": 0,
        "native_gate_count_without_pass": 736,
        "native_gate_count_with_pass": 540,
        "native_gate_count_fold_first": 736,
        "changed_circuit_count": 0,
        "executed_circuit_count": 60,
    },
    "pair_across_a_declared_barrier": {
        "circuit_count": 60,
        "source_instruction_count": 360,
        "without_pass_instruction_count": 300,
        "with_pass_instruction_count": 300,
        "removed_by_rest_of_pipeline": 60,
        "removed_by_declared_inverse_pass": 0,
        "fold_first_instruction_count": 300,
        "removed_by_declared_inverse_pass_fold_first": 0,
        "native_gate_count_without_pass": None,
        "native_gate_count_with_pass": None,
        "native_gate_count_fold_first": None,
        "changed_circuit_count": 0,
        "executed_circuit_count": 0,
    },
    "pair_behind_a_half_pi_pulse": {
        "circuit_count": 60,
        "source_instruction_count": 180,
        "without_pass_instruction_count": 120,
        "with_pass_instruction_count": 60,
        "removed_by_rest_of_pipeline": 60,
        "removed_by_declared_inverse_pass": 60,
        "fold_first_instruction_count": 120,
        "removed_by_declared_inverse_pass_fold_first": 0,
        "native_gate_count_without_pass": 360,
        "native_gate_count_with_pass": 60,
        "native_gate_count_fold_first": 360,
        "changed_circuit_count": 60,
        "executed_circuit_count": 60,
    },
    "pair_between_two_half_pi_pulses": {
        "circuit_count": 60,
        "source_instruction_count": 240,
        "without_pass_instruction_count": 60,
        "with_pass_instruction_count": 120,
        "removed_by_rest_of_pipeline": 180,
        "removed_by_declared_inverse_pass": -60,
        "fold_first_instruction_count": 60,
        "removed_by_declared_inverse_pass_fold_first": 0,
        "native_gate_count_without_pass": 240,
        "native_gate_count_with_pass": 120,
        "native_gate_count_fold_first": 240,
        "changed_circuit_count": 60,
        "executed_circuit_count": 60,
    },
}

#: The two pass orders summed over the populations the declared basis can express.
#: The compiler-instruction totals are equal -- the two orders differ by 60 on two
#: populations and in opposite directions -- and the native totals are not: 1320
#: against 2026. The barrier population is outside both sums, and
#: ``covered_population_count`` says so rather than letting the total read as a
#: figure for all six.
_ORDER_TOTALS = {
    "covered_population_count": 5,
    "population_count": 6,
    "native_gate_count_with_pass": 1320,
    "native_gate_count_fold_first": 2026,
    "instruction_count_with_pass": 540,
    "instruction_count_fold_first": 540,
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
    changed, and ``removed_by_rest_of_pipeline`` shows it is not the source circuit:
    the one-qubit fold the branch merged into reaches 480, 180, 240, 60, 60 and 180
    instructions on the six populations on its own. Every delta below is therefore
    measured against that pipeline and not against an unoptimized program, which is
    what makes a zero delta a finding -- the fold already reaches those shapes -- and
    a negative one a real outcome rather than an arithmetic slip.
    """

    rows = {row["label"]: row for row in payload["pipeline_delta"]}
    for label, row in rows.items():
        assert row["removed_by_rest_of_pipeline"] == (
            row["source_instruction_count"] - row["without_pass_instruction_count"]
        ), label
        assert row["removed_by_declared_inverse_pass"] == (
            row["without_pass_instruction_count"] - row["with_pass_instruction_count"]
        ), label
    assert sum(row["removed_by_rest_of_pipeline"] for row in rows.values()) > 0
    assert sum(row["removed_by_declared_inverse_pass"] for row in rows.values()) == 0
    assert rows["adjacent_pair_members"]["removed_by_declared_inverse_pass"] == 0
    assert (
        rows["pair_between_two_half_pi_pulses"]["removed_by_declared_inverse_pass"] < 0
    )


def test_the_native_column_is_attributable_to_this_pass(payload: dict) -> None:
    """The pass is visible in native gates on rows where it is invisible in order.

    An instruction count measured before lowering is not the number a user pays, so
    both programs are also lowered into one declared basis and counted. On the two
    populations whose compiler delta is zero the pass still changes which program the
    fold reaches, and the native counts show it: 690 to 600 and 736 to 540. A count
    the declared basis cannot express is reported as ``None`` and never as zero, and
    the pass must not make the native program longer on any gate-only population.
    """

    rows = {row["label"]: row for row in payload["pipeline_delta"]}
    assert rows["pair_across_a_declared_barrier"]["native_gate_count_without_pass"] is (
        None
    )
    assert rows["pair_across_a_declared_barrier"]["native_gate_count_with_pass"] is None
    for label, row in rows.items():
        without = row["native_gate_count_without_pass"]
        with_pass = row["native_gate_count_with_pass"]
        if without is None:
            assert with_pass is None, label
            assert row["executed_circuit_count"] == 0, label
            continue
        assert with_pass is not None, label
        # A population that still has instructions cannot lower to zero gates, so a
        # zero beside a non-empty program would mean the column was skipped.
        if row["with_pass_instruction_count"]:
            assert with_pass > 0, label
        assert with_pass <= without, label
    assert rows["pair_across_another_qubit"]["native_gate_count_without_pass"] == 690
    assert rows["pair_across_another_qubit"]["native_gate_count_with_pass"] == 600
    assert rows["pair_behind_a_half_pi_pulse"]["native_gate_count_without_pass"] == 360
    assert rows["pair_behind_a_half_pi_pulse"]["native_gate_count_with_pass"] == 60


def test_the_two_orders_of_the_two_passes_are_measured_and_named(
    payload: dict,
) -> None:
    """Both pass orders are run, and the one that ships is the one that lowers cheaper.

    ``pipeline._optimize_to_fixed_point`` calls this pass before
    ``merge_adjacent_rotations`` and the fold last; the fold-first pipeline is
    recorded under ``fold_first_*``. The fold-first order makes this pass contribute
    nothing, which is arithmetic rather than a surprise: the fold alone is the
    baseline these deltas are measured against, so whichever of the two runs first
    takes the program and the second finds nothing to do. That is why the equality
    below is asserted in that direction only -- ``fold_first`` must equal
    ``without_pass`` -- and the shipped order is *not* asserted to equal it.

    What the order changes is the native count, and the test asserts the direction
    rather than only the numbers: on every population the declared basis can express,
    the shipped order lowers to no more native gates than the fold-first one, and on
    four of the five it lowers to strictly fewer -- every population where the fold
    would have had to re-spell a declared pair on its own. The two exceptions are
    named as well: ``adjacent_pair_members`` is already empty in both orders, and the
    barrier population has no native count in either. So the order in the shipped
    pipeline is a measured choice; a future change that made fold-first cheaper would
    have to move these rows and say so.
    """

    rows = {row["label"]: row for row in payload["pipeline_delta"]}
    for label, row in rows.items():
        assert row["removed_by_declared_inverse_pass_fold_first"] == (
            row["without_pass_instruction_count"] - row["fold_first_instruction_count"]
        ), label
        # The fold-first pipeline is the comparison baseline, so its marginal delta
        # is zero by construction, and any nonzero value would mean the two pipelines
        # this module defines had drifted apart.
        assert row["removed_by_declared_inverse_pass_fold_first"] == 0, label
        assert (
            row["fold_first_instruction_count"] == row["without_pass_instruction_count"]
        ), label
        assert (
            row["native_gate_count_fold_first"] == row["native_gate_count_without_pass"]
        ), label
        if row["native_gate_count_with_pass"] is None:
            continue
        assert (
            row["native_gate_count_with_pass"] <= row["native_gate_count_fold_first"]
        ), label

    cheaper = [
        label
        for label, row in rows.items()
        if row["native_gate_count_with_pass"] is not None
        and row["native_gate_count_with_pass"] < row["native_gate_count_fold_first"]
    ]
    assert cheaper == [
        "pair_across_another_qubit",
        "interleaved_pairs",
        "pair_behind_a_half_pi_pulse",
        "pair_between_two_half_pi_pulses",
    ]
    assert rows["adjacent_pair_members"]["native_gate_count_fold_first"] == (
        rows["adjacent_pair_members"]["native_gate_count_with_pass"]
    )
    assert payload["pass_order_totals"] == _ORDER_TOTALS
    # The totals are not a claim about populations the basis cannot express.
    assert _ORDER_TOTALS["covered_population_count"] < _ORDER_TOTALS["population_count"]


def test_removing_every_instruction_preserved_the_program(payload: dict) -> None:
    """Every removed instruction was paid for with an execution.

    Five of the six populations are gate-only and every circuit in them is executed
    and compared against the source program; the barrier population is that case
    honestly at zero, because a declared barrier makes the circuit unexecutable as a
    gate-only program and there is nothing there to check. It is also why the
    comparison is against the pipeline without this pass rather than against the
    source: the barrier population still loses 60 instructions to the one-qubit fold,
    and a comparison against 360 would have attributed those to this pass. The
    difference is no longer exactly zero on the populations where the pass changes the
    program: the two programs are different instruction sequences, so the statevector
    contraction rounds differently, and 1e-16 is the size of that rounding rather than
    a tolerance chosen to make an inequality hold.
    """

    rows = {row["label"]: row for row in payload["pipeline_delta"]}
    gate_only = [label for label in rows if label != "pair_across_a_declared_barrier"]
    assert len(gate_only) == 5
    for label in gate_only:
        row = rows[label]
        assert row["executed_circuit_count"] == row["circuit_count"], label
        assert row["max_state_difference"] < 1.0e-12, label
    barrier = rows["pair_across_a_declared_barrier"]
    assert barrier["changed_circuit_count"] == 0
    assert barrier["executed_circuit_count"] == 0
    assert barrier["with_pass_instruction_count"] == (
        barrier["without_pass_instruction_count"]
    )
    assert barrier["with_pass_instruction_count"] < barrier["source_instruction_count"]
    # And the check is not vacuous: the populations where the pass changes the
    # program are the ones that carry the nonzero rounding.
    assert rows["pair_behind_a_half_pi_pulse"]["changed_circuit_count"] == 60
    assert rows["pair_behind_a_half_pi_pulse"]["max_state_difference"] > 0.0


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
            assert row["pass_instruction_count"] == (
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
