"""Contract for the diagonal-before-measurement measurements W9-10 records.

Before this round ``compiler.optimize`` removed a zero angle, an ``i``, a
self-inverse pair and a pair of rotations that sum to zero, and never asked whether
a gate commutes with the measurement that follows it. So ``h; rz(0.4); measure``
survived as three instructions even though the ``rz`` moves no amplitude of the
state the measurement reads. The benchmark module answers four questions about the
pass that closes that gap. Where does its rule come from? What does the pipeline
gain? Where does the pass stop, and why? And which rows can the Qiskit anchor be
asked about at all?

The recorded answers are that the rule is a property of the operator declaration
rather than a matrix read at run time -- the pass module holds exactly **8** string
literals and imports neither a matrix source nor the operator schema; that on seven
seeded populations the pass removes **316** instructions alone and **158** more once
its own count is taken after the other passes, every removal checked against an
exact outcome distribution; that on **196** shape rows it removes **26**, declines
**170**, and of the **81** it declined that were invisible in fact, **42** were
declined behind a condition and **39** because a later instruction stands on the
candidate's wire -- with **0** removals that were not invisible; and that the Qiskit
anchor disagrees on exactly **6** rows, all three of them the class-membership facts
its opcode list cannot express.

These tests hold that evidence in place. They fail if the rule stops being a property
of the declaration and starts being a matrix read, if the pass or the pipeline stops
removing on a population, if a removal is no longer checked against an execution, if
the two decline reasons are folded into one total, if a row stops asserting that the
program is one this instrument can speak for, or if the anchor is allowed to report a
disagreement that no published reason covers.
"""

import pytest

from benchmarks.compiler_diagonal_before_measure import (
    _SWEEP_SEED,
    run_benchmark,
)
from tests.benchmark_contract.qiskit_lane import require_certified_lane

pytestmark = pytest.mark.benchmark_contract

#: The rule. Every number here is a property of the pass module's text and of the
#: operator census, so it is a fixture rather than a measurement: if one moves, the
#: pass is resting on something else and its reach numbers are not comparable with
#: these. ``pass_module_line_count`` is deliberately absent -- it moves whenever the
#: file is edited and would pin prose.
_RULE = {
    "ir_field_count": 8,
    "pass_module_string_literal_count": 8,
    "pass_module_string_literals": [
        "condition_clauses",
        "conditions",
        "cphase",
        "crz",
        "cz",
        "is_dynamic",
        "measure",
        "rzz",
    ],
    "pass_module_imports": [
        "__future__",
        "core.ir",
        "dataclasses",
        "one_qubit_synthesis",
    ],
    "pass_module_imports_a_matrix_source": False,
    "pass_module_imports_the_operator_schema": False,
    "declared_two_wire_opcode_count": 4,
    "declared_two_wire_opcodes": ["cphase", "crz", "cz", "rzz"],
    "single_qubit_rule": (
        "the polar angle of the opcode's canonical Euler triple is exactly zero"
    ),
    "two_wire_rule": "membership in one four-opcode frozenset",
}

#: Per-population instruction counts. ``removed_by_the_pass_in_the_pipeline`` is the
#: delta over the shipped loop with this one pass swapped for the identity. These are
#: read under whatever the rest of the pipeline does, so a later pass landing on main
#: may move them and they are re-measured rather than assumed.
#:
#: The two populations that isolate this rule are ``two_wire_diagonal_before_two_
#: measurements`` -- the legacy loop removes nothing there at all -- and
#: ``a_gate_blocked_by_a_later_rotation``, the negative control, where the pass must
#: remove nothing and change no program. ``a_general_form_at_a_zero_polar_angle`` is
#: the population where a count delta of **0** co-exists with **12** changed programs:
#: the legacy loop already reaches those instructions by folding the two gates into
#: one, so the count ties while the programs differ, and ``changed_circuit_count`` is
#: what keeps that from reading as "no effect". It compares the shipped program with
#: the legacy one by opcode, wires and parameters rather than by length for that
#: reason. ``a_mixed_program_with_mid_circuit_measurements`` is the one population
#: whose ``executed_circuit_count`` is below its ``circuit_count`` and it says so.
_DELTA = {
    "terminal_diagonal_gates": {
        "circuit_count": 30,
        "source_instruction_count": 185,
        "legacy_instruction_count": 123,
        "optimized_instruction_count": 111,
        "removed_by_the_pass_alone": 30,
        "removed_by_the_pass_in_the_pipeline": 12,
        "changed_circuit_count": 25,
        "executed_circuit_count": 30,
    },
    "diagonal_gate_behind_a_hadamard": {
        "circuit_count": 30,
        "source_instruction_count": 90,
        "legacy_instruction_count": 61,
        "optimized_instruction_count": 60,
        "removed_by_the_pass_alone": 30,
        "removed_by_the_pass_in_the_pipeline": 1,
        "changed_circuit_count": 24,
        "executed_circuit_count": 30,
    },
    "a_general_form_at_a_zero_polar_angle": {
        "circuit_count": 30,
        "source_instruction_count": 90,
        "legacy_instruction_count": 60,
        "optimized_instruction_count": 60,
        "removed_by_the_pass_alone": 30,
        "removed_by_the_pass_in_the_pipeline": 0,
        "changed_circuit_count": 12,
        "executed_circuit_count": 30,
    },
    "two_wire_diagonal_before_two_measurements": {
        "circuit_count": 30,
        "source_instruction_count": 180,
        "legacy_instruction_count": 180,
        "optimized_instruction_count": 150,
        "removed_by_the_pass_alone": 30,
        "removed_by_the_pass_in_the_pipeline": 30,
        "changed_circuit_count": 30,
        "executed_circuit_count": 30,
    },
    "a_gate_blocked_by_a_later_rotation": {
        "circuit_count": 30,
        "source_instruction_count": 90,
        "legacy_instruction_count": 62,
        "optimized_instruction_count": 62,
        "removed_by_the_pass_alone": 0,
        "removed_by_the_pass_in_the_pipeline": 0,
        "changed_circuit_count": 0,
        "executed_circuit_count": 30,
    },
    "a_mixed_program": {
        "circuit_count": 30,
        "source_instruction_count": 296,
        "legacy_instruction_count": 227,
        "optimized_instruction_count": 173,
        "removed_by_the_pass_alone": 94,
        "removed_by_the_pass_in_the_pipeline": 54,
        "changed_circuit_count": 26,
        "executed_circuit_count": 30,
    },
    "a_mixed_program_with_mid_circuit_measurements": {
        "circuit_count": 30,
        "source_instruction_count": 351,
        "legacy_instruction_count": 288,
        "optimized_instruction_count": 227,
        "removed_by_the_pass_alone": 102,
        "removed_by_the_pass_in_the_pipeline": 61,
        "changed_circuit_count": 26,
        "executed_circuit_count": 9,
    },
}

#: The shape census. ``removed_but_not_removable_count`` is the correctness gate and
#: the two ``declined_but_removable_*`` columns are the deferred reach, counted apart
#: from each other because they are different limits and one total would let either
#: hide behind the other.
_SHAPE_CENSUS = {
    "row_count": 196,
    "candidate_count": 29,
    "gap_count": 8,
    "one_wire_row_count": 108,
    "two_wire_row_count": 88,
    "removed_count": 26,
    "removed_one_wire_row_count": 18,
    "removed_two_wire_row_count": 8,
    "declined_count": 170,
    "declined_but_removable_count": 81,
    "declined_but_removable_behind_a_condition_count": 42,
    "declined_but_removable_blocked_by_a_later_instruction_count": 39,
    "removed_but_not_removable_count": 0,
    "measured_row_count": 156,
    # The last two are read under the whole fixed-point loop and not under this pass
    # alone, so another pass landing on main may move them: the commutation pass that
    # merged as this round was prepared moved both by one row. They are re-measured
    # rather than kept. Every other number above is this pass's own.
    "pipeline_only_row_count": 75,
    "removed_in_the_pipeline_count": 101,
}

#: (removed_by_the_pass, candidate_is_diagonal, removable_in_fact,
#: removed_in_the_pipeline) per ``<candidate>|<gap>`` row. The rows are read back one
#: at a time because the shapes are not interchangeable.
#:
#: ``i|a_gate_on_its_wire`` and ``x|a_gate_on_its_wire`` are the pair that makes the
#: truth column worth having: the first is diagonal and invisible in fact, and the
#: second is not diagonal and is invisible *in this program* only, because a Hadamard
#: preparation leaves the wire in ``|+>`` where ``x`` is a global phase. The pass
#: declines both -- it is state-independent -- and the two declines are counted
#: differently on purpose.
#:
#: ``only_one_of_its_wires_is_measured`` is where the reach is largest and the reason
#: is not a condition: every one of the eleven two-wire candidates is invisible in
#: fact there, because only the untouched wire and the control's wire are read, and
#: the pass declines all eleven because it requires a measurement after every wire of
#: the candidate.
_SPOT_ROWS = {
    "z|nothing_between": (True, True, True, True),
    "i|nothing_between": (True, True, True, True),
    "cphase|nothing_between": (True, True, True, True),
    "h|nothing_between": (False, False, False, True),
    "rx|nothing_between": (False, False, True, False),
    "u3|nothing_between": (False, False, False, False),
    "z|a_gate_on_another_qubit": (True, True, True, True),
    "u3|a_gate_on_another_qubit": (False, False, False, False),
    "z|a_gate_on_its_wire": (False, True, False, True),
    "i|a_gate_on_its_wire": (False, True, True, True),
    "x|a_gate_on_its_wire": (False, False, True, True),
    "z|no_measurement": (False, True, None, True),
    "u3|no_measurement": (False, False, None, False),
    "z|the_measurement_is_conditional": (False, True, True, True),
    "z|the_candidate_is_conditional": (False, True, True, True),
    "cz|only_one_of_its_wires_is_measured": (False, True, True, False),
    "cx|only_one_of_its_wires_is_measured": (False, False, True, False),
    "cz|a_gate_between_the_two_measurements": (False, True, None, False),
}

#: The instruction forms the pass declines by construction, each with the count of
#: rows that form produced. Only the bare gate is removed.
_FORMS = {
    "a_bare_gate": True,
    "a_gate_with_a_caller_supplied_matrix": False,
    "a_conditional_gate": False,
    "a_condition_clause_gate": False,
    "a_dynamic_gate": False,
    "a_two_wire_gate_on_one_wire_of_measurement": False,
}

#: The anchor. It is asked about every row, and its answer is allowed to differ from
#: the pass's only where one of the three published class-membership facts explains
#: the difference. ``removed_count`` is the anchor's own reach over the same rows.
#:
#: The Qiskit release is deliberately not in here. It is the instrument the readings
#: were taken with, not one of them, and folding it into the same equality made this
#: anchor report that a newer certified instrument had produced a wrong measurement.
#: The instrument is checked separately below, against the certified lanes rather
#: than against whichever release one machine happened to have.
_ANCHOR = {
    "available": True,
    "pass_name": "RemoveDiagonalGatesBeforeMeasure",
    "row_count": 196,
    "removed_count": 20,
    "disagreement_count": 6,
}

#: The three opcodes the anchor cannot be asked about, because its list names gate
#: classes that these operators are not members of. Every disagreement the anchor
#: reports has to belong to one of these, and the count has to be the number of rows
#: these opcodes appear on where the pass removes and the anchor does not.
_ANCHOR_DISAGREEING_OPCODES = ("cphase", "i", "phase")


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_benchmark()


def _rows(payload: dict) -> dict[str, dict]:
    return {row["label"]: row for row in payload["shape_table"]["rows"]}


def _delta(payload: dict) -> dict[str, dict]:
    return {row["label"]: row for row in payload["pipeline_delta"]}


def test_the_benchmark_is_classified_as_a_local_microbenchmark(payload: dict) -> None:
    assert (
        payload["schema"] == "flagquantum_compiler_diagonal_before_measure_benchmark_v1"
    )
    assert payload["artifact_classification"] == "local_compiler_microbenchmark"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert payload["seed"] == {"sweep": _SWEEP_SEED}


def test_the_rule_is_a_property_of_the_declaration_and_not_a_table(
    payload: dict,
) -> None:
    rule = payload["rule_contract"]

    for key, expected in _RULE.items():
        assert rule[key] == expected, key


def test_the_pass_reads_no_matrix_and_no_operator_schema(payload: dict) -> None:
    """A rule of proof reads the program; a rule of record reads a table.

    This is the load-bearing scope constraint of the round: ``compiler/**`` may not
    import ``simulation/**``, so the pass cannot ask a matrix whether it is diagonal
    and has to rest on a declaration instead. The day this test fails, the pass is
    resting on something the architecture forbids it to read.
    """

    rule = payload["rule_contract"]
    imports = set(rule["pass_module_imports"])

    assert rule["pass_module_imports_a_matrix_source"] is False
    assert rule["pass_module_imports_the_operator_schema"] is False
    assert not any("matri" in name or "schema" in name for name in imports)


def test_the_two_wire_rule_is_a_declaration_and_it_is_the_measured_census(
    payload: dict,
) -> None:
    """The pass's four-opcode set is checked against the matrices, not restated.

    A per-opcode declaration is the only thing available: no ``OperatorSchema`` field
    records diagonality and no two-wire diagonality table exists in the repository.
    This census is where the declaration meets the matrix, so the declaration cannot
    drift from the operators it names without this failing.
    """

    rule = payload["rule_contract"]
    census = rule["classification_census"]

    assert census["disagreement_count"] == 0
    assert rule["declared_two_wire_opcode_count"] == len(
        rule["declared_two_wire_opcodes"]
    )
    assert sorted(rule["declared_two_wire_opcodes"]) == sorted(
        _RULE["declared_two_wire_opcodes"]
    )
    measured = {row["opcode"] for row in census["two_wire"] if row["measured_diagonal"]}
    assert measured == set(rule["declared_two_wire_opcodes"])


def test_every_single_qubit_opcode_is_checked_at_two_angles(payload: dict) -> None:
    """A generic angle is not enough: three opcodes are diagonal only at zero.

    ``rx``, ``ry`` and ``u3`` are one opcode with a parameter, so their diagonality is
    a property of the value and not of the name. A census that only measured the
    generic point would call them non-diagonal and the pass's reach at a zero polar
    angle would have no check behind it.
    """

    census = payload["rule_contract"]["classification_census"]
    zero_only = {
        row["opcode"]
        for row in census["single_qubit"]
        if row["measured_diagonal_at_a_zero_polar_angle"]
        and not row["measured_diagonal_at_the_generic_angles"]
    }

    assert zero_only == {"rx", "ry", "u3"}
    assert (
        sum(
            1
            for row in census["single_qubit"]
            if row["reported_diagonal_at_the_generic_angles"]
        )
        == 9
    )
    assert all(row["agrees"] for row in census["single_qubit"])
    assert all(row["agrees"] for row in census["two_wire"])


def test_every_population_reports_its_own_removal_count(payload: dict) -> None:
    rows = _delta(payload)

    assert set(rows) == set(_DELTA)
    for label, expected in _DELTA.items():
        row = rows[label]
        for key, value in expected.items():
            assert row[key] == value, (label, key)


def test_every_population_is_accounted_for_arithmetically(payload: dict) -> None:
    """The columns have to agree with each other, or one of them is mislabelled."""

    for row in payload["pipeline_delta"]:
        label = row["label"]
        assert (
            row["source_instruction_count"] - row["removed_by_the_legacy_pipeline"]
            == row["legacy_instruction_count"]
        ), label
        assert (
            row["legacy_instruction_count"] - row["removed_by_the_pass_in_the_pipeline"]
            == row["optimized_instruction_count"]
        ), label
        assert row["changed_circuit_count"] <= row["circuit_count"], label
        assert row["executed_circuit_count"] <= row["circuit_count"], label


def test_a_population_that_was_not_executed_says_so_and_the_rest_were(
    payload: dict,
) -> None:
    """A removal nobody executed is an arithmetic result, not evidence.

    Every population but one is executed in full. The exception is the one built to
    allow a measurement in the middle of the program, which the exact instrument
    cannot witness; it reports both counts rather than dropping the unwitnessed
    circuits from its arithmetic.
    """

    for row in payload["pipeline_delta"]:
        label = row["label"]
        if label == "a_mixed_program_with_mid_circuit_measurements":
            assert row["executed_circuit_count"] < row["circuit_count"], label
        else:
            assert row["executed_circuit_count"] == row["circuit_count"], label


def test_the_pass_is_measured_against_an_execution_on_every_executed_population(
    payload: dict,
) -> None:
    """The correctness gate: nothing was removed that moved the outcome distribution."""

    for row in payload["pipeline_delta"]:
        assert row["instrument"] == "exact_outcome_distribution", row["label"]
        assert (
            row["max_outcome_distribution_difference"]
            <= payload["execution_control"]["tolerance"]
        ), row["label"]


def test_the_instrument_has_a_control_that_moved(payload: dict) -> None:
    """A comparison that cannot fail is not evidence.

    Two controls delete an instruction that does move the distribution -- a
    non-diagonal one-qubit gate and an entangler -- and one deletes a gate that
    commutes with the measurement and must not move it. A tolerance with no control
    above it would pass on any program.
    """

    control = payload["execution_control"]

    assert control["deleting_a_diagonal_gate_that_commutes_with_the_measurement"] == 0.0
    assert (
        control["deleting_a_non_diagonal_gate_on_one_wire"] > control["control_floor"]
    )
    assert control["deleting_an_entangler"] > control["control_floor"]
    assert control["tolerance"] < control["control_floor"]


def test_the_shape_census_is_the_one_the_pass_currently_produces(payload: dict) -> None:
    shapes = payload["shape_table"]

    for key, expected in _SHAPE_CENSUS.items():
        assert shapes[key] == expected, key
    assert shapes["row_count"] == len(shapes["rows"])
    assert shapes["removed_count"] + shapes["declined_count"] == shapes["row_count"]
    assert (
        shapes["one_wire_row_count"] + shapes["two_wire_row_count"]
        == shapes["row_count"]
    )
    assert (
        shapes["removed_one_wire_row_count"] + shapes["removed_two_wire_row_count"]
        == shapes["removed_count"]
    )


def test_deferred_reach_is_split_by_reason_and_not_folded_into_one_total(
    payload: dict,
) -> None:
    """A refusal behind a condition and a refusal behind a later gate differ.

    Both were invisible in fact and both were declined; one total would let a change
    that moved every refusal from one reason to the other pass unnoticed.
    """

    shapes = payload["shape_table"]
    behind_a_condition = shapes["declined_but_removable_behind_a_condition_count"]
    blocked = shapes["declined_but_removable_blocked_by_a_later_instruction_count"]

    assert behind_a_condition + blocked == shapes["declined_but_removable_count"]
    assert behind_a_condition > 0
    assert blocked > 0
    assert behind_a_condition != blocked


def test_no_removal_exceeded_what_the_program_measured(payload: dict) -> None:
    shapes = payload["shape_table"]

    assert shapes["removed_but_not_removable_count"] == 0
    assert all(
        row["removable_in_fact"] is not False
        for row in shapes["rows"]
        if row["removed_by_the_pass"]
    )


def test_every_row_states_whether_the_pass_was_right_and_why_it_declined(
    payload: dict,
) -> None:
    """Every declined row carries the fact that it was invisible, or a reason it was not.

    A declined row with no truth column would be indistinguishable from a row the rule
    was right to decline, and ``no_measurement`` rows are the ones where the truth
    column must be absent rather than false: with no measurement there is no
    distribution to be unchanged.
    """

    for row in payload["shape_table"]["rows"]:
        label = row["label"]
        if row["gap"] in {"no_measurement", "a_gate_between_the_two_measurements"}:
            # No distribution to compare: nothing is measured at all, or a gate
            # computes after the last measurement on the candidate's wire.
            assert row["instrument"] == "not_applicable", label
            assert row["removable_in_fact"] is None, label
            assert not row["removed_by_the_pass"], label
        else:
            assert row["instrument"] == "exact_outcome_distribution", label
            assert isinstance(row["removable_in_fact"], bool), label


def test_the_shape_table_indexes_the_candidate_and_not_a_lookalike(
    payload: dict,
) -> None:
    """Every row's own gate is the only instruction of its opcode in its program.

    Without this the anchor's opcode-count difference and the pass's length difference
    would both be free to be about a neighbour, and a row would report a reach that
    belongs to something else.
    """

    for row in payload["shape_table"]["rows"]:
        assert row["candidate"] == row["label"].split("|", 1)[0]
        assert row["candidate_wire_count"] in (1, 2)
        assert isinstance(row["candidate_is_diagonal"], bool)


@pytest.mark.parametrize("label", sorted(_SPOT_ROWS))
def test_a_shape_is_measured_both_ways_round(payload: dict, label: str) -> None:
    row = _rows(payload)[label]
    removed, diagonal, removable, in_pipeline = _SPOT_ROWS[label]

    assert row["removed_by_the_pass"] == removed
    assert row["candidate_is_diagonal"] == diagonal
    assert row["removable_in_fact"] == removable
    assert row["removed_in_the_pipeline"] == in_pipeline


def test_the_pass_removes_every_diagonal_candidate_that_a_measurement_follows(
    payload: dict,
) -> None:
    """The line the rule draws, read off the rows rather than restated.

    On the two gaps where nothing stands between the candidate and the measurement on
    any of its wires, the pass has to remove every row whose candidate the matrix
    calls diagonal, and it has to remove nothing else there.
    """

    rows = [
        row
        for row in payload["shape_table"]["rows"]
        if row["gap"] in {"nothing_between", "a_gate_on_another_qubit"}
    ]
    removed = {row["label"] for row in rows if row["removed_by_the_pass"]}
    diagonal = {row["label"] for row in rows if row["candidate_is_diagonal"]}

    assert removed == diagonal
    assert len(removed) == 26
    assert len(rows) == 58


def test_the_form_table_is_not_a_single_verdict(payload: dict) -> None:
    """The forms the pass declines by construction, and the one it does not."""

    table = payload["form_table"]

    assert table["row_count"] == len(_FORMS)
    for row in table["rows"]:
        assert row["removed"] == _FORMS[row["form"]], row["form"]
    assert table["removed_count"] == sum(1 for value in _FORMS.values() if value)
    assert 0 < table["removed_count"] < table["row_count"]


def test_the_qiskit_anchor_is_optional_and_reports_its_own_reach(payload: dict) -> None:
    """The anchor is a cross-check, not a dependency.

    It is asked about the same rows the pass is, and its reach is reported next to the
    pass's rather than folded into it: the two ports are not the same rule and the
    module has to run on a machine with no Qiskit at all.

    The release it was read on is checked before the readings are, because the
    readings are only reproducible on a certified lane. That check fails rather than
    skips, and the reason is in ``qiskit_lane``: no lane that collects this file ever
    installs Qiskit, so a skip would be invisible everywhere it would matter.
    """

    anchor = payload["shape_table"]["reference_anchor"]

    if not anchor["available"]:
        # A skip here is not silence: the sibling test that states the agreement
        # scope pins the readings that must hold when the anchor did not run, so the
        # absent case is measured rather than passed over.
        pytest.skip(f"Qiskit is not installed here: {anchor['reason']}")
    require_certified_lane(anchor["qiskit_version"], recording="this anchor")
    for key, expected in _ANCHOR.items():
        assert anchor[key] == expected, key
    assert anchor["row_count"] == payload["anchor_row_count"]


def test_every_anchor_disagreement_is_one_of_the_three_published_facts(
    payload: dict,
) -> None:
    """An unlisted disagreement is a defect, not a footnote.

    The three facts are that Qiskit's ``diagonal_1q_gates`` names no ``IGate``, that
    ``PhaseGate`` is not a ``U1Gate`` subclass, and that ``CPhaseGate`` is not a
    ``CU1Gate`` subclass. Any other difference between the two ports would mean one of
    them is wrong about a row, so the benchmark raises rather than filing it.
    """

    anchor = payload["shape_table"]["reference_anchor"]

    if not anchor["available"]:
        # A skip here is not silence: the sibling test that states the agreement
        # scope pins the readings that must hold when the anchor did not run, so the
        # absent case is measured rather than passed over.
        pytest.skip(f"Qiskit is not installed here: {anchor['reason']}")
    reasons = anchor["disagreement_reasons"]

    assert len(reasons) == anchor["disagreement_count"]
    assert payload["anchor_disagreement_count"] == anchor["disagreement_count"]
    assert set(reasons) == {
        "i|nothing_between",
        "i|a_gate_on_another_qubit",
        "phase|nothing_between",
        "phase|a_gate_on_another_qubit",
        "cphase|nothing_between",
        "cphase|a_gate_on_another_qubit",
    }
    assert {label.split("|", 1)[0] for label in reasons} == set(
        _ANCHOR_DISAGREEING_OPCODES
    )
    for label, reason in reasons.items():
        assert label.split("|", 1)[0] in reason or "subclass" in reason, label


def test_the_agreement_scope_states_the_case_that_actually_happened(
    payload: dict,
) -> None:
    """The scope is a statement about a run, so it has to name the run.

    FlagQuantum removes ``u3(0, phi, lam)``, ``rx(0)`` and ``ry(0)`` because their
    diagonality is a property of the value, and Qiskit's list names classes, so it has
    no way to express those rows at all. The scope string and the disagreement count
    are what keep a reader from reading the two ports as agreeing everywhere.

    A host with no Qiskit is the case this test exists for as much as the other one:
    the anchor drives its port on no row there, and a scope that still said
    ``row_count`` rows had been driven through both ports -- with a disagreement count
    of zero -- would read as agreement rather than as silence. The absent branch
    therefore asserts the readings that must hold *because* nothing ran, and not a
    skip, so neither branch can pass by having nothing to check.
    """

    scope = payload["anchor_agreement_scope"]
    anchor = payload["shape_table"]["reference_anchor"]

    assert payload["reference_algorithm"] == (
        "qiskit_remove_diagonal_gates_before_measure"
    )

    if anchor["available"]:
        assert payload["anchor_row_count"] == payload["shape_table"]["row_count"]
        assert payload["anchor_disagreement_count"] == _ANCHOR["disagreement_count"]
        assert "actually driven" in scope
        assert "IGate" in scope or "gate-class" in scope
        return

    assert payload["anchor_row_count"] == 0
    assert payload["anchor_disagreement_count"] == 0
    assert anchor["reason"]
    assert "no row was driven" in scope
    assert "not" in scope and "installed here" in scope
    assert "no agreement" in scope
