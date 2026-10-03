"""Contract for the zero-state reset measurements W9-11 records.

Before this round, ``compiler.optimize`` removed an ``i``, a zero angle, a
self-inverse pair and a pair of rotations that sum to zero, and never removed a
``reset`` in any position, so a leading ``reset(0)`` -- the identity, because a
``CircuitIR`` register starts in ``|0...0>`` -- survived the pipeline unchanged. The benchmark
module answers four questions about the pass that closes that gap. Where does its
rule come from? What does the pipeline gain? Where does the pass stop, and why? And
which of its two instruments can speak for which row?

The recorded answers are that the rule is a property of the IR rather than a table
-- the pass module holds exactly **1** string literal and imports neither the
operator schema nor any matrix source, and ``CircuitIR`` has **8** fields none of
which can carry another initial state; that on seven seeded populations the pass
removes **182** instructions the pipeline had left in place and **213** once its own
count is taken after the other passes, every one of them checked against an
execution; that on 24 shape rows and 30 resets it
removes **12**, declines **13** that a 64-trajectory average reduced state shows
were in fact removable, and removes **0** that were not; that it declines all three
non-bare forms of the instruction; and that both share and final-state comparisons
are driven with a control that moves a share by **0.5** where the tolerance is
**0.03**.

These tests hold that evidence in place. They fail if the rule stops being a
property of the IR and starts being a table, if the pass or the pipeline stops
removing on a population, if a removal is no longer checked against an execution,
if the deferred-reach count is folded into a single "declined" total, or if a
comparison is left without a control that could have failed.
"""

import pytest

from benchmarks.compiler_zero_state_reset import (
    _DRAWS,
    _REGISTER_WIDTH,
    _SHAPES,
    _SHARE_TOLERANCE,
    run_benchmark,
)
from flagquantum.core.ir import CircuitIR

pytestmark = pytest.mark.benchmark_contract

#: The rule. Every number here is a property of the pass module's text and of the IR's
#: field list, so it is a fixture rather than a measurement: if one moves, the pass is
#: resting on something else and its reach numbers are not comparable with these.
_RULE = {
    "ir_field_count": 8,
    "ir_carries_an_initial_state": False,
    "pass_module_string_literal_count": 1,
    "pass_module_string_literals": ["reset"],
    "pass_module_imports": ["__future__", "core.ir", "dataclasses"],
}

#: Per-population instruction counts. ``removed_by_the_pass_in_the_pipeline`` is the
#: delta over the pipeline without the pass. ``executed_circuit_count`` is asserted
#: equal to ``circuit_count`` on every population: a removal that no execution saw is
#: an arithmetic result, not evidence.
_DELTA = {
    "leading_resets": {
        "circuit_count": 30,
        "source_instruction_count": 129,
        "legacy_instruction_count": 123,
        "optimized_instruction_count": 60,
        "removed_by_the_pass_alone": 63,
        "removed_by_the_pass_in_the_pipeline": 63,
        "changed_circuit_count": 30,
        "executed_circuit_count": 30,
        "comparison": "final_state",
    },
    "reset_after_another_qubit_gate": {
        "circuit_count": 30,
        "source_instruction_count": 90,
        "legacy_instruction_count": 90,
        "optimized_instruction_count": 60,
        "removed_by_the_pass_alone": 30,
        "removed_by_the_pass_in_the_pipeline": 30,
        "changed_circuit_count": 30,
        "executed_circuit_count": 30,
        "comparison": "final_state",
    },
    "blocked_by_a_gate_on_its_own_qubit": {
        "circuit_count": 30,
        "source_instruction_count": 90,
        "legacy_instruction_count": 90,
        "optimized_instruction_count": 90,
        "removed_by_the_pass_alone": 0,
        "removed_by_the_pass_in_the_pipeline": 0,
        "changed_circuit_count": 0,
        "executed_circuit_count": 30,
        "comparison": "final_state",
    },
    "blocked_by_a_measure_on_its_own_qubit": {
        "circuit_count": 30,
        "source_instruction_count": 90,
        "legacy_instruction_count": 90,
        "optimized_instruction_count": 90,
        "removed_by_the_pass_alone": 0,
        "removed_by_the_pass_in_the_pipeline": 0,
        "changed_circuit_count": 0,
        "executed_circuit_count": 30,
        "comparison": "outcome_shares",
    },
    "reset_after_a_cancellable_pair": {
        "circuit_count": 30,
        "source_instruction_count": 120,
        "legacy_instruction_count": 60,
        "optimized_instruction_count": 30,
        "removed_by_the_pass_alone": 0,
        "removed_by_the_pass_in_the_pipeline": 30,
        "changed_circuit_count": 30,
        "executed_circuit_count": 30,
        "comparison": "outcome_shares",
    },
    "reset_before_a_measurement": {
        "circuit_count": 30,
        "source_instruction_count": 120,
        "legacy_instruction_count": 120,
        "optimized_instruction_count": 60,
        "removed_by_the_pass_alone": 60,
        "removed_by_the_pass_in_the_pipeline": 60,
        "changed_circuit_count": 30,
        "executed_circuit_count": 30,
        "comparison": "outcome_shares",
    },
    "mixed_dynamic_programs": {
        "circuit_count": 30,
        "source_instruction_count": 295,
        "legacy_instruction_count": 285,
        "optimized_instruction_count": 255,
        "removed_by_the_pass_alone": 29,
        "removed_by_the_pass_in_the_pipeline": 30,
        "changed_circuit_count": 19,
        "executed_circuit_count": 30,
        "comparison": "outcome_shares",
    },
}

#: The shape census. ``removed_but_not_removable_count`` is the correctness gate and
#: ``declined_but_removable_count`` is the deferred reach; they are separate because a
#: removal that should not have happened and a removal that merely did not happen are
#: different findings, and one total would let either hide behind the other.
_SHAPE_CENSUS = {
    "row_count": 24,
    "reset_count": 30,
    "removed_count": 12,
    "declined_but_removable_count": 13,
    "removed_but_not_removable_count": 0,
    "draws_per_reset": _DRAWS,
    "average_reduced_state_resolution": 1.0 / _DRAWS,
    "pipeline_only_row_count": 5,
}

#: Rows read back one at a time, because the shapes are not interchangeable. The two
#: measure rows are the pair that makes the instrument's choice matter: a measure on
#: the reset's *own* wire after nothing leaves the wire in ``|0>`` in every trajectory,
#: so the reset is removable in fact and the decline is deferred reach; a measure after
#: a Hadamard leaves it in ``|0>`` on half the trajectories, so the reset is genuinely
#: not removable. A single "a measure came before it" label would call both the same.
#:
#: (removed_by_the_pass, removable_in_fact, removed_by_the_pipeline) per shape.
_SPOT_ROWS = {
    "nothing_before_it": (1, 1, 1),
    "another_qubit_only": (1, 1, 1),
    "two_leading_resets": (2, 2, 2),
    "three_leading_resets": (3, 3, 3),
    "a_measure_on_another_qubit": (1, 1, 1),
    "a_hadamard_on_its_qubit": (0, 0, 0),
    "a_pauli_z_on_its_qubit": (0, 1, 0),
    "a_zero_angle_rotation": (0, 1, 2),
    "an_identity_gate": (0, 1, 2),
    "a_measure_on_its_qubit": (0, 1, 0),
    "a_measure_after_a_hadamard_on_its_qubit": (0, 0, 0),
    "two_resets_after_a_measure": (0, 2, 0),
    "a_self_inverse_pair": (0, 1, 3),
    "a_pair_of_rotations_that_sum_to_zero": (0, 1, 3),
    "a_controlled_not_from_a_zero_control": (0, 1, 0),
    "a_controlled_not_onto_the_wire": (0, 0, 0),
    "a_swap_with_a_zero_qubit": (0, 1, 0),
    "a_controlled_not_twice": (0, 1, 3),
    "an_entangled_partner": (0, 0, 0),
}

#: The shapes where the pipeline reaches further than the pass can alone, because
#: another pass empties the wire first. That reach belongs to the fixed-point loop and
#: is not evidence for this rule, so the count is reported rather than attributed.
_PIPELINE_ONLY_SHAPES = (
    "a_zero_angle_rotation",
    "an_identity_gate",
    "a_self_inverse_pair",
    "a_pair_of_rotations_that_sum_to_zero",
    "a_controlled_not_twice",
)

#: The instruction forms the pass declines even on a wire nothing has touched.
_FORMS = {
    "a_bare_reset": True,
    "a_conditional_reset": True,
    "a_two_qubit_reset": False,
    "a_parameterised_reset": False,
    "a_reset_with_a_caller_supplied_matrix": False,
}


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_benchmark()


def _rows(payload: dict) -> dict[str, dict]:
    return {row["shape"]: row for row in payload["shape_table"]["rows"]}


def _delta(payload: dict) -> dict[str, dict]:
    return {row["label"]: row for row in payload["pipeline_delta"]}


def test_measurement_is_classified_as_a_local_microbenchmark(payload: dict) -> None:
    assert payload["schema"] == "flagquantum_compiler_zero_state_reset_benchmark_v1"
    assert payload["artifact_classification"] == "local_compiler_microbenchmark"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False


def test_the_rule_is_a_property_of_the_ir_and_not_a_table(payload: dict) -> None:
    rule = payload["rule_contract"]

    for key, expected in _RULE.items():
        assert rule[key] == expected, key


def test_the_ir_cannot_carry_another_initial_state(payload: dict) -> None:
    """The day this fails, the pass's rule is wrong and not merely stale."""

    rule = payload["rule_contract"]

    assert rule["ir_carries_an_initial_state"] is False
    assert "initial_state" not in rule["ir_field_names"]
    assert len(rule["ir_field_names"]) == rule["ir_field_count"]


def test_the_pass_holds_exactly_one_gate_name_and_it_is_the_rule(payload: dict) -> None:
    rule = payload["rule_contract"]

    assert rule["pass_module_string_literals"] == ["reset"]
    assert rule["pass_module_string_literal_count"] == 1


def test_the_pass_reads_no_matrix_and_no_operator_schema(payload: dict) -> None:
    """A rule of proof reads the program; a rule of record reads a table."""

    imports = set(payload["rule_contract"]["pass_module_imports"])

    assert imports == {"__future__", "core.ir", "dataclasses"}
    assert not any("schema" in name or "matri" in name for name in imports)


def test_every_population_reports_its_own_removal_count(payload: dict) -> None:
    rows = _delta(payload)

    assert set(rows) == set(_DELTA)
    for label, expected in _DELTA.items():
        row = rows[label]
        assert row["circuit_count"] == expected["circuit_count"], label
        assert (
            row["source_instruction_count"] == expected["source_instruction_count"]
        ), label
        assert (
            row["legacy_instruction_count"] == expected["legacy_instruction_count"]
        ), label
        assert (
            row["optimized_instruction_count"]
            == expected["optimized_instruction_count"]
        ), label
        assert (
            row["removed_by_the_pass_alone"] == expected["removed_by_the_pass_alone"]
        ), label
        assert (
            row["removed_by_the_pass_in_the_pipeline"]
            == expected["removed_by_the_pass_in_the_pipeline"]
        ), label
        assert row["changed_circuit_count"] == expected["changed_circuit_count"], label


def test_every_population_was_executed_and_not_only_counted(payload: dict) -> None:
    """A removal nobody executed is an arithmetic result, not evidence."""

    for row in payload["pipeline_delta"]:
        assert row["executed_circuit_count"] == row["circuit_count"], row["label"]


def test_the_delta_belongs_to_this_pass_and_not_to_the_pipeline(payload: dict) -> None:
    """On the populations that isolate the new rule, the old pipeline removes nothing."""

    rows = _delta(payload)

    assert rows["leading_resets"]["legacy_instruction_count"] == 123
    assert rows["reset_after_another_qubit_gate"]["legacy_instruction_count"] == 90
    assert rows["reset_before_a_measurement"]["legacy_instruction_count"] == 120
    isolated = (
        rows["leading_resets"]["removed_by_the_pass_alone"]
        + rows["reset_after_another_qubit_gate"]["removed_by_the_pass_alone"]
        + rows["reset_before_a_measurement"]["removed_by_the_pass_alone"]
    )
    assert isolated > 150


def test_the_fixed_point_loop_reaches_further_than_the_pass_alone(
    payload: dict,
) -> None:
    """A reset behind a pair another pass cancels needs the loop, not just this rule."""

    rows = _delta(payload)
    row = rows["reset_after_a_cancellable_pair"]

    assert row["removed_by_the_pass_alone"] == 0
    assert row["removed_by_the_pass_in_the_pipeline"] == 30


def test_no_removal_exceeded_what_the_wire_measured(payload: dict) -> None:
    """The correctness gate: nothing was removed that was not removable in fact."""

    shapes = payload["shape_table"]

    assert shapes["removed_but_not_removable_count"] == 0
    assert all(row["removed_but_not_removable_count"] == 0 for row in shapes["rows"])


def test_deferred_reach_is_reported_and_not_folded_into_one_total(
    payload: dict,
) -> None:
    """A refusal that had to happen and one that merely did happen are different."""

    shapes = payload["shape_table"]

    assert shapes["declined_but_removable_count"] == 13
    assert shapes["removed_count"] == 12
    assert shapes["declined_but_removable_count"] != shapes["removed_count"]
    assert (
        sum(row["removable_in_fact_but_declined_count"] for row in shapes["rows"])
        == shapes["declined_but_removable_count"]
    )


def test_the_shape_census_is_the_one_the_pass_currently_produces(payload: dict) -> None:
    shapes = payload["shape_table"]

    for key, expected in _SHAPE_CENSUS.items():
        assert shapes[key] == expected, key
    assert shapes["row_count"] == len(_SHAPES)
    assert shapes["row_count"] == len(shapes["rows"])


def test_the_instrument_reports_its_own_resolution(payload: dict) -> None:
    """A trajectory average has a floor, and the floor is stated rather than implied."""

    shapes = payload["shape_table"]

    assert shapes["draws_per_reset"] == _DRAWS
    assert shapes["average_reduced_state_resolution"] == 1.0 / _DRAWS


def test_the_measure_rows_are_not_a_single_label(payload: dict) -> None:
    """The row that makes the instrument's choice matter, read back by hand."""

    rows = _rows(payload)
    trailing = rows["a_measure_on_its_qubit"]
    half = rows["a_measure_after_a_hadamard_on_its_qubit"]

    assert trailing["removable_in_fact_count"] == 1
    assert trailing["removable_in_fact_but_declined_count"] == 1
    assert half["removable_in_fact_count"] == 0
    assert half["removable_in_fact_but_declined_count"] == 0


@pytest.mark.parametrize("shape", sorted(_SPOT_ROWS))
def test_a_shape_is_measured_both_ways_round(payload: dict, shape: str) -> None:
    row = _rows(payload)[shape]
    expected_removed, expected_removable, expected_pipeline = _SPOT_ROWS[shape]

    assert row["removed_by_the_pass"] == expected_removed
    assert row["removable_in_fact_count"] == expected_removable
    assert row["removed_by_the_pipeline"] == expected_pipeline


def test_the_pipeline_only_rows_are_the_ones_that_were_named(payload: dict) -> None:
    """Reach the loop earned is counted, and counted apart from the rule's own."""

    rows = _rows(payload)
    predicted = {
        shape for shape in _SPOT_ROWS if _SPOT_ROWS[shape][2] > _SPOT_ROWS[shape][0]
    }

    assert predicted == set(_PIPELINE_ONLY_SHAPES)
    assert payload["shape_table"]["pipeline_only_row_count"] == len(
        _PIPELINE_ONLY_SHAPES
    )
    assert all(
        rows[shape]["removed_by_the_pipeline"] > rows[shape]["removed_by_the_pass"]
        for shape in _PIPELINE_ONLY_SHAPES
    )


def test_every_declined_form_is_declined_for_a_stated_reason(payload: dict) -> None:
    """Not an enumeration of the implementation: each refusal names a property."""

    rows = {row["form"]: row for row in payload["form_table"]["rows"]}

    assert set(rows) == set(_FORMS)
    for form, removed in _FORMS.items():
        assert rows[form]["removed"] is removed, form
    assert rows["a_two_qubit_reset"]["wire_count"] == 2
    assert rows["a_parameterised_reset"]["carries_parameters"] is True
    assert rows["a_reset_with_a_caller_supplied_matrix"]["carries_a_matrix"] is True
    assert payload["form_table"]["removed_count"] == 2


def test_a_reset_carrying_a_condition_is_removed_only_when_the_wire_is_clean(
    payload: dict,
) -> None:
    """A condition does not touch its wire, so it is not what decides the removal."""

    rows = {row["form"]: row for row in payload["form_table"]["rows"]}

    assert rows["a_conditional_reset"]["removed"] is True


def test_each_comparison_has_a_control_that_could_have_failed(payload: dict) -> None:
    """A tolerance is only meaningful next to something that exceeds it."""

    control = payload["execution_control"]

    assert control["tolerance"] == _SHARE_TOLERANCE
    assert control["deleting_a_reset_that_was_a_no_op"] < _SHARE_TOLERANCE
    assert (
        control["deleting_a_reset_after_a_gate_that_left_the_zero_state"]
        > 10 * _SHARE_TOLERANCE
    )


def test_no_population_moved_an_outcome_beyond_the_tolerance(payload: dict) -> None:
    for row in payload["pipeline_delta"]:
        if row["comparison"] == "outcome_shares":
            assert row["max_outcome_share_difference"] < _SHARE_TOLERANCE, row["label"]
            assert row["distinct_outcome_count"] >= 1, row["label"]


def test_no_population_moved_a_final_state_beyond_rounding(payload: dict) -> None:
    for row in payload["pipeline_delta"]:
        if row["comparison"] == "final_state":
            assert row["max_final_state_difference"] < 1.0e-9, row["label"]


def test_every_comparison_row_declares_which_instrument_spoke_for_it(
    payload: dict,
) -> None:
    """A reset reads a random draw, so which instrument ran is part of the record.

    The instrument is chosen by whether the population measures, and the two choices
    are not interchangeable: a final-state comparison of a measuring population would
    be comparing one trajectory against another trajectory, and an outcome-share
    comparison of a measure-free population would have one outcome with share 1.
    """

    for row in payload["pipeline_delta"]:
        assert row["comparison"] in {"final_state", "outcome_shares"}, row["label"]
        if row["comparison"] == "final_state":
            assert row["distinct_outcome_count"] == 0, row["label"]
        else:
            assert row["distinct_outcome_count"] >= 1, row["label"]


def test_the_qiskit_anchor_compares_only_what_both_passes_attempt(
    payload: dict,
) -> None:
    anchor = payload["shape_table"]["reference_anchor"]

    if not anchor["available"]:
        pytest.skip(f"Qiskit is not installed: {anchor['reason']}")
    assert anchor["pass_name"] == "RemoveResetInZeroState"
    assert payload["anchor_agreement_count"] == (
        payload["shape_table"]["row_count"] - payload["anchor_disagreement_count"]
    )
    assert payload["anchor_disagreement_count"] == 0
    assert "form table" in payload["anchor_agreement_scope"]


def test_the_whole_shape_table_was_actually_driven(payload: dict) -> None:
    """Non-vacuity: a table of rows that removed nothing would satisfy every gate."""

    shapes = payload["shape_table"]

    assert shapes["row_count"] == 24
    assert shapes["reset_count"] == 30
    assert shapes["removed_count"] >= 12
    assert shapes["declined_but_removable_count"] >= 13


def test_the_imported_module_under_test_is_the_ir_the_benchmark_reads(
    payload: dict,
) -> None:
    """The rule contract describes the field list of the type the pass receives."""

    import dataclasses

    fields = sorted(field.name for field in dataclasses.fields(CircuitIR))

    assert fields == payload["rule_contract"]["ir_field_names"]
    assert _REGISTER_WIDTH >= 3
