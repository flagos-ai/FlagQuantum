"""Contract for the exact-identity measurements W9-12 records.

Before this round ``compiler.optimize`` decided a single-qubit instruction was the
identity from a hand-written set of two opcode names -- ``{"i", "id"}`` -- so a
``u3`` whose polar angle had come back to zero was never removed however its two
phases read, and a rotation whose angle had come back to a full turn was never
removed either. The pass now reads the angle triple its own operator declaration
gives it and folds every angle modulo ``4*pi``. The benchmark module answers four
questions. Where does the rule come from? What does it remove that the rule it
replaces could not? Where does it stop, and what does each refusal cost? And what
can the Qiskit anchor be asked about at all?

The recorded answers are that the pass reads
``one_qubit_synthesis.canonical_euler_angles`` -- the single place the module states
the triple -- so the pass function holds **0** opcode string literals and imports
nothing but the module that supplies the triple; that the fold is ``4*pi`` and not
``2*pi`` because a half turn is *minus* the identity, a sign ``CircuitIR`` has no
field to record and ``optimize`` is required not to introduce; that on the ``31``
declared unitaries measured at five angle points the rule reaches **36** rows where
the rule it replaces reached **19**, never claims an identity the runtime matrix
denies, and *declines* **10** rows the matrix does make the identity -- every one of
them published with the mechanism that produces it; that on seven seeded populations
the rule removes **832** instructions alone and a net **+23** once the other passes
have run, with the worst outcome-distribution movement **2.78e-16** against a
control that moves **3.95e-02**; and that the Qiskit anchor -- which does not exist
in 1.2.4 and is measured on 2.0.3 -- disagrees with the rule on **17** rows, every
one of them a row where the matrix is minus the identity, and agrees with the
phase-blind column on all **155**.

These tests hold that evidence in place. They fail if the rule stops being read from
the declaration, if the fold is narrowed back to ``2*pi``, if a row the matrix makes
the identity starts being removed, if a refusal stops being published with its
mechanism, if the two instruments are conflated, if a removal stops being checked
against an execution, or if the anchor is allowed to report a disagreement that no
published reason covers.
"""

import math

import pytest

from benchmarks.compiler_identity_elimination import (
    _MODULAR_FOLD_LIMIT,
    _POINTS,
    _SWEEP_SEED,
    _TWO_PI,
    run_benchmark,
)

pytestmark = pytest.mark.benchmark_contract

#: The rule. Every number here is a property of the pass module's text, of the
#: operator census, or of the runtime gate matrices, so it is a fixture rather than a
#: measurement: if one moves, the pass is resting on something else and its reach
#: numbers are not comparable with these.
_RULE = {
    "pass_module": "flagquantum/compiler/pipeline.py",
    "predicate_module": "flagquantum/compiler/one_qubit_synthesis.py",
    # The two functions' line counts are published by the benchmark but deliberately
    # not pinned here: they move whenever either docstring is edited and would pin
    # prose rather than a property, which is the trap the sibling contract for
    # `remove_diagonal_gates_before_measure` documents for `pass_module_line_count`.
    "pass_function_imports": ["one_qubit_synthesis"],
    "pass_function_opcode_literals": [],
    "predicate_holds_no_angle_table": True,
    "predicate_refuses_a_caller_supplied_matrix": True,
    "the_same_gate_without_a_matrix_is_removed": True,
    "euler_tables_in_module": ["_FIXED_EULER_ANGLES", "_PARAMETERIZED_EULER_ANGLES"],
    "angle_source": (
        "one_qubit_synthesis.canonical_euler_angles, the single place the module "
        "states the triple"
    ),
    "single_qubit_rule_period": 4.0 * math.pi,
    "modular_fold_limit": 2048.0 * math.pi,
    "tolerance": 1e-12,
    "global_phase_field_in_ir": False,
    "declared_single_qubit_opcodes": [
        "h",
        "i",
        "phase",
        "rx",
        "ry",
        "rz",
        "s",
        "sdg",
        "sx",
        "sxdg",
        "t",
        "tdg",
        "u1",
        "u2",
        "u3",
        "x",
        "y",
        "z",
    ],
    "declared_multi_wire_rotations": [
        "cphase",
        "crx",
        "cry",
        "crz",
        "rxx",
        "ryy",
        "rzz",
    ],
    "non_unitary_channels": [
        "amplitude_damping",
        "bit_flip",
        "depolarizing",
        "phase_flip",
    ],
}

#: The grid census. The instrument measures the runtime gate matrix at every row, so
#: ``overremoval_count`` is the correctness gate and it has to stay ``0``; the reach
#: that is *declined* is published beside it rather than netted off, because a rule
#: that were widened without a measurement behind it would move rows from one column
#: to the other and a single total would not show it.
#:
#: ``row_count`` is not the number of distinct programs: twelve of the eighteen
#: single-qubit opcodes take no parameter at all, so the grid re-runs the same program
#: at every angle point. ``distinct_row_count`` is the number to quote.
_CENSUS = {
    "opcode_count": 31,
    "single_qubit_opcode_count": 18,
    "row_count": 155,
    "single_qubit_row_count": 90,
    "multi_wire_row_count": 65,
    "distinct_row_count": 72,
    "measured_removed_count": 46,
    "measured_removed_single_qubit_count": 24,
    "measured_removed_up_to_phase_count": 53,
    "rule_removed_count": 36,
    "rule_removed_single_qubit_count": 22,
    "overremoval_count": 0,
    "predicate_rule_mismatch_count": 0,
    "predicate_claims_an_identity_the_matrix_denies_count": 0,
    "removed_up_to_phase_and_not_exactly": [
        "half_turn|rx",
        "half_turn|rxx",
        "half_turn|ry",
        "half_turn|ryy",
        "half_turn|rz",
        "half_turn|rzz",
        "half_turn|u3",
    ],
    "declined_reach_count": 10,
    "removed_by_point": {
        "generic": ["i"],
        "zero_polar_summing": [
            "cphase",
            "crx",
            "cry",
            "crz",
            "i",
            "phase",
            "rx",
            "rxx",
            "ry",
            "ryy",
            "rz",
            "rzz",
            "u1",
            "u3",
        ],
        "zero_polar_not_summing": [
            "cphase",
            "crx",
            "cry",
            "crz",
            "i",
            "phase",
            "rx",
            "rxx",
            "ry",
            "ryy",
            "rz",
            "rzz",
            "u1",
        ],
        "half_turn": ["cphase", "i", "phase", "u1"],
        "full_turn": [
            "cphase",
            "crx",
            "cry",
            "crz",
            "i",
            "phase",
            "rx",
            "rxx",
            "ry",
            "ryy",
            "rz",
            "rzz",
            "u1",
            "u3",
        ],
    },
    "declined_reach": [
        "full_turn|cphase",
        "half_turn|cphase",
        "full_turn|crx",
        "full_turn|cry",
        "full_turn|crz",
        "half_turn|phase",
        "full_turn|rxx",
        "full_turn|ryy",
        "full_turn|rzz",
        "half_turn|u1",
    ],
    "declined_reach_by_mechanism": {
        "shared_z_rotation_triple": ["half_turn|phase", "half_turn|u1"],
        "two_wire_branch_reads_the_raw_parameter": [
            "full_turn|cphase",
            "full_turn|crx",
            "full_turn|cry",
            "full_turn|crz",
            "full_turn|rxx",
            "full_turn|ryy",
            "full_turn|rzz",
            "half_turn|cphase",
        ],
    },
}

#: The period table. `2*pi` is a full turn for the three opcodes whose matrix is
#: `diag(1, 1, 1, exp(i*theta))`-shaped and a *half* turn for the ten whose matrix
#: carries `theta/2`; that split is the entire reason the fold cannot be `2*pi`, and
#: it is measured from the runtime matrices rather than derived from the opcode names.
_PERIOD = {
    "candidates_in_two_pi": [1.0, 2.0],
    "opcode_count": 13,
    "exact_identity_at_two_pi": ["cphase", "phase", "u1"],
    "exact_identity_at_four_pi": [
        "crx",
        "cry",
        "crz",
        "rx",
        "rxx",
        "ry",
        "ryy",
        "rz",
        "rzz",
        "u3",
    ],
    "no_exact_identity_within_a_full_period": [],
    "no_polar_angle_opcodes": [
        "h",
        "i",
        "s",
        "sdg",
        "sx",
        "sxdg",
        "t",
        "tdg",
        "u2",
        "x",
        "y",
        "z",
    ],
}

#: Per-population instruction counts. Both the clamped column and the signed one are
#: recorded, because ``full_turn_rotations`` is measured *negative*: the ``4*pi`` fold
#: declines a half turn that the superseded ``2*pi`` fold removed, and
#: ``collapse_one_qubit_runs`` then re-expands some programs into more instructions
#: than the superseded rule left. A clamped column alone would report that row as a
#: tie.
#:
#: These are read under whatever the rest of the pipeline does, so a later pass
#: landing on main may move them and they are re-measured rather than assumed.
#:
#: ``full_turn_rotations`` has moved once, when `collapse_one_qubit_runs` learned to
#: delete a run whose product is exactly the identity. That pass sits beside this rule
#: rather than inside it, and it reaches the *superseded* arm further than the shipped
#: one: the shipped rule has already removed part of a run before the fold sees it,
#: where the superseded rule leaves those gates standing and the fold then deletes the
#: whole run. So the superseded count fell by ten and the shipped count by four, and
#: the row reads as more lengthened (-5 -> -11) although neither rule changed. The
#: difference between the two rules is still real -- 139 against 150 -- and
#: `removed_by_the_rule_alone` below, which drives this rule directly, did not move.
#: This is the coupling the note above warns about, in the direction that is easy to
#: misread as a regression in this rule.
_DELTA = {
    "parameter_free_gates": {
        "circuit_count": 30,
        "executed_circuit_count": 30,
        "source_instruction_count": 554,
        "superseded_rule_instruction_count": 167,
        "optimized_instruction_count": 167,
        "removed_by_the_rule_alone": 44,
        "removed_by_the_rule_in_the_pipeline": 0,
        "net_instruction_delta_against_the_superseded_rule": 0,
        "changed_circuit_count": 0,
    },
    "zero_angle_rotations": {
        "circuit_count": 30,
        "executed_circuit_count": 30,
        "source_instruction_count": 554,
        "superseded_rule_instruction_count": 162,
        "optimized_instruction_count": 162,
        "removed_by_the_rule_alone": 267,
        "removed_by_the_rule_in_the_pipeline": 0,
        "net_instruction_delta_against_the_superseded_rule": 0,
        "changed_circuit_count": 0,
    },
    "full_turn_rotations": {
        "circuit_count": 30,
        "executed_circuit_count": 30,
        "source_instruction_count": 554,
        "superseded_rule_instruction_count": 139,
        "optimized_instruction_count": 150,
        "removed_by_the_rule_alone": 181,
        "removed_by_the_rule_in_the_pipeline": 6,
        "net_instruction_delta_against_the_superseded_rule": -11,
        "changed_circuit_count": 19,
    },
    "zero_polar_u3": {
        "circuit_count": 30,
        "executed_circuit_count": 30,
        "source_instruction_count": 810,
        "superseded_rule_instruction_count": 176,
        "optimized_instruction_count": 176,
        "removed_by_the_rule_alone": 101,
        "removed_by_the_rule_in_the_pipeline": 1,
        "net_instruction_delta_against_the_superseded_rule": 0,
        "changed_circuit_count": 5,
    },
    # The next three rows are the only ones later passes co-own, and this is the
    # population group where they have reach. `collapse_two_qubit_runs` composes
    # adjacent two-qubit rotations -- `rzz(a) rzz(b)` into `rzz(a + b)` -- and
    # `merge_commuting_rotations` adds two rotations of one opcode across a proven
    # commuting gap. Both run inside *both* pipelines, so the two post-pipeline counts
    # move together while the source count and `removed_by_the_rule_alone`, which are
    # properties of this rule alone, stay where they were.
    #
    # Each of the two co-owners is attributed by a test that pauses it and re-measures
    # these same rows, so neither move can go silent and neither is inferred from the
    # arithmetic:
    # `test_compiler_two_qubit_optimization.py::test_the_fold_is_what_moved_the_round_18_pipeline_rows`
    # holds this rule and the rotation merge fixed and moves only the fold, and
    # `test_compiler_commutation_cancellation.py::test_the_identity_rows_the_rotation_merge_moved_are_attributed`
    # holds this rule and the fold fixed and moves only the rotation merge. The literals
    # below are the shipped pipeline, which is both co-owners active.
    "two_wire_rotations": {
        "circuit_count": 30,
        "executed_circuit_count": 30,
        "source_instruction_count": 322,
        "superseded_rule_instruction_count": 200,
        "optimized_instruction_count": 200,
        "removed_by_the_rule_alone": 47,
        "removed_by_the_rule_in_the_pipeline": 0,
        "net_instruction_delta_against_the_superseded_rule": 0,
        "changed_circuit_count": 0,
    },
    "mixed": {
        "circuit_count": 30,
        "executed_circuit_count": 30,
        "source_instruction_count": 678,
        "superseded_rule_instruction_count": 396,
        "optimized_instruction_count": 382,
        "removed_by_the_rule_alone": 164,
        "removed_by_the_rule_in_the_pipeline": 14,
        "net_instruction_delta_against_the_superseded_rule": 14,
        "changed_circuit_count": 15,
    },
    "mixed_with_mid_circuit_measures": {
        "circuit_count": 30,
        "executed_circuit_count": 30,
        "source_instruction_count": 761,
        "superseded_rule_instruction_count": 504,
        "optimized_instruction_count": 492,
        "removed_by_the_rule_alone": 159,
        # One lower than before the fold, and the clamped column is why: it is a sum
        # of per-circuit `max(0, legacy - shipped)`, so a fold that shrinks one
        # circuit's two pipelines by different amounts moves it by a circuit, not by
        # the gate count. The signed column below follows it by one, which is the
        # identity this test asserts.
        "removed_by_the_rule_in_the_pipeline": 13,
        "net_instruction_delta_against_the_superseded_rule": 12,
        "changed_circuit_count": 16,
    },
}

#: The anchor. ``RemoveIdentityEquivalent`` does not exist in Qiskit 1.2.4 at all, so
#: the comparison is against 2.0.3, and 2.0.3 is a *band*: it removes any gate whose
#: average gate fidelity with the identity clears a cutoff, where this rule removes
#: only an exact identity. The two therefore cannot agree, and every row where they
#: differ is one where the runtime matrix is *minus* the identity -- a sign this rule
#: cannot drop. The anchor's own fidelity test is sign-blind, so it is compared with
#: the phase-blind column and not with the exact one.
_ANCHOR = {
    "pass_name": "RemoveIdentityEquivalent",
    "qiskit_version": "2.0.3",
    "present_in_qiskit_1_2_4": False,
    "row_count": 155,
    "removed_count": 53,
    "disagreement_with_the_rule_count": 17,
    "agrees_with_the_up_to_phase_column_on_every_row": True,
}

#: Every anchor disagreement, named. All seventeen are rows where the matrix is minus
#: the identity, arriving by one of the two published mechanisms: the shared
#: `rz`/`phase`/`u1` canonical triple, which cannot be told apart at a half period, or
#: the two-wire branch, which reads the raw angle parameter without folding it.
_ANCHOR_DISAGREEMENTS = [
    "full_turn|cphase",
    "full_turn|crx",
    "full_turn|cry",
    "full_turn|crz",
    "full_turn|rxx",
    "full_turn|ryy",
    "full_turn|rzz",
    "half_turn|cphase",
    "half_turn|phase",
    "half_turn|rx",
    "half_turn|rxx",
    "half_turn|ry",
    "half_turn|ryy",
    "half_turn|rz",
    "half_turn|rzz",
    "half_turn|u1",
    "half_turn|u3",
]


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_benchmark()


def test_the_grid_is_the_declared_census(payload: dict) -> None:
    census = payload["rule_contract"]["classification_census"]

    for key, expected in _CENSUS.items():
        assert census[key] == expected, key
    assert census["opcode_count"] == (
        census["single_qubit_opcode_count"]
        + len(_RULE["declared_multi_wire_rotations"])
        + len(_RULE["non_unitary_channels"])
        + 2  # `ccx` and `cswap`
    )
    assert (
        census["single_qubit_row_count"] + census["multi_wire_row_count"]
        == census["row_count"]
    )
    assert census["row_count"] == len(census["rows"])
    # The grid is `arity-1 opcodes x points` plus `arity>=2 opcodes x points`, so the
    # row count is not the opcode count and must not be read as one.
    assert census["row_count"] > census["opcode_count"]
    assert census["distinct_row_count"] < census["row_count"]
    assert (
        census["measured_removed_count"]
        + (census["row_count"] - census["measured_removed_count"])
        == census["row_count"]
    )
    # `removed_by_point` is the *measured* column, not the rule's: it is the exact
    # runtime matrix at each angle point, so it sums to the measured total.
    assert (
        sum(len(names) for names in census["removed_by_point"].values())
        == census["measured_removed_count"]
    )


def test_the_rule_is_read_from_the_declaration_and_not_from_a_name_list(
    payload: dict,
) -> None:
    """The gap this round closes, pinned as a property of the pass module's text.

    The old single-qubit branch was a two-element string set. The new one holds no
    opcode string literal at all and imports the module that states the angle triple,
    so the rule cannot drift away from the declaration without failing here.
    """

    contract = payload["rule_contract"]

    for key, expected in _RULE.items():
        assert contract[key] == expected, key
    assert contract["pass_function_line_count"] > 0
    assert contract["predicate_function_line_count"] >= (
        contract["pass_function_line_count"]
    )
    assert contract["pass_function_opcode_literals"] == []
    assert contract["pass_function_imports"] == ["one_qubit_synthesis"]
    assert contract["predicate_holds_no_angle_table"] is True
    assert "canonical_euler_angles" in contract["angle_source"]
    # Fail-closed in the corner where the name stops describing the operator: the
    # declared triple of `i` is the identity, so a rule that read the triple alone
    # would remove a gate a caller had already overridden with its own matrix.
    assert contract["predicate_refuses_a_caller_supplied_matrix"] is True
    assert contract["the_same_gate_without_a_matrix_is_removed"] is True
    # The declaration itself is what the fold is applied to, and the tables the module
    # already owned stay the only place the fixed gates' angles are written down.
    assert contract["single_qubit_rule_period"] == 4.0 * math.pi
    assert contract["single_qubit_rule_period"] == 2.0 * _TWO_PI
    assert contract["modular_fold_limit"] == _MODULAR_FOLD_LIMIT
    assert contract["modular_fold_limit"] == 2048.0 * math.pi


def test_the_fold_is_four_pi_because_a_half_turn_is_minus_the_identity(
    payload: dict,
) -> None:
    """The measurement the whole design rests on, and the sign IR cannot record.

    If the fold were narrowed to `2*pi` this fails twice over: the period table's
    `exact_identity_at_four_pi` column would lose the ten opcodes that only return to
    `+I` at a full period, and the census would stop naming the half-turn refusals.
    """

    contract = payload["rule_contract"]
    periods = payload["period_table"]

    assert contract["global_phase_field_in_ir"] is False
    for key, expected in _PERIOD.items():
        assert periods[key] == expected, key
    assert periods["opcode_count"] == len(periods["table"])
    assert set(periods["exact_identity_at_two_pi"]).isdisjoint(
        periods["exact_identity_at_four_pi"]
    )
    # Every opcode with a polar angle ends up at `+I` within one full period; the ones
    # that do not appear in either list have no polar angle to move at all.
    assert set(periods["exact_identity_at_two_pi"]) | set(
        periods["exact_identity_at_four_pi"]
    ) == set(periods["table"])
    assert set(periods["no_polar_angle_opcodes"]).isdisjoint(periods["table"])
    # And the split is not a name list: the three that return at a half period are
    # exactly the three whose declared triple shares one tabulated form.
    assert "phase" in periods["exact_identity_at_two_pi"]
    assert "u1" in periods["exact_identity_at_two_pi"]
    assert "cphase" in periods["exact_identity_at_two_pi"]
    for name in ("rx", "ry", "rz", "u3"):
        assert periods["table"][name]["identity_up_to_phase_at"] == _TWO_PI, name
        assert periods["table"][name]["exact_identity_at"] == 2.0 * _TWO_PI, name
        assert periods["table"][name]["rule_removes_at"] is True, name
    # A two-wire rotation reaches `+I` at a full period too and the rule still declines
    # it, because the two-wire branch reads the raw parameter without folding it.
    for name in _RULE["declared_multi_wire_rotations"]:
        # `cphase` is the one two-wire rotation whose matrix is a bare phase on the
        # controlled subspace, so it returns to `+I` at a *half* period like `phase`
        # does; the other six carry `theta/2` and need the full period.
        expected = _TWO_PI if name == "cphase" else 2.0 * _TWO_PI
        assert periods["table"][name]["exact_identity_at"] == expected, name
        assert periods["table"][name]["rule_removes_at"] is False, name


def test_no_row_is_removed_that_the_runtime_matrix_denies(payload: dict) -> None:
    """The correctness gate, measured against the gate matrices rather than asserted."""

    census = payload["rule_contract"]["classification_census"]

    assert census["overremoval_count"] == 0
    assert census["overremovals"] == []
    assert census["predicate_claims_an_identity_the_matrix_denies_count"] == 0
    assert census["predicate_claims_an_identity_the_matrix_denies"] == []
    # The predicate and the shipped pass are the same rule; a mismatch would mean the
    # loop is deciding with something other than the predicate under test.
    assert census["predicate_rule_mismatch_count"] == 0
    assert census["predicate_rule_mismatches"] == []
    # The three columns are ordered, and the ordering is what makes the reach readable:
    # what the rule removes is a subset of what is exactly the identity, which is a
    # subset of what is the identity up to a phase.
    assert census["rule_removed_count"] <= census["measured_removed_count"]
    assert (
        census["measured_removed_count"] <= census["measured_removed_up_to_phase_count"]
    )
    assert (
        census["rule_removed_single_qubit_count"]
        <= census["measured_removed_single_qubit_count"]
    )
    assert len(census["removed_up_to_phase_and_not_exactly"]) == (
        census["measured_removed_up_to_phase_count"] - census["measured_removed_count"]
    )


def test_the_rule_is_strictly_wider_than_the_one_it_replaces(payload: dict) -> None:
    """What the predicate earns, stated against the branch it replaced rather than a sketch.

    The deleted branch had two halves and both are restated so that the comparison is
    with what shipped. The single-qubit half removed `{"i", "id"}`; the two-wire half
    removed any declared rotation whose parameters were all zero, and that half is
    *retained*. So the reach grows by the single-qubit rows alone, and the measurement
    says so rather than leaving it to be inferred.
    """

    census = payload["rule_contract"]["classification_census"]
    points = len(_POINTS)
    two_wire = _RULE["declared_multi_wire_rotations"]

    # The rule's own reach per point, taken from what the rule removes rather than from
    # what the matrix makes removable: the measured column minus the refusals this
    # round publishes. `removed_by_point` alone is the matrix column and would make the
    # rule look wider than it is.
    declined = {
        point: {
            row.split("|")[1]
            for row in census["declined_reach"]
            if row.startswith(f"{point}|")
        }
        for point in _POINTS
    }
    new_reach = {
        f"{point}|{name}"
        for point, names in census["removed_by_point"].items()
        for name in names
        if name not in declined[point]
    }
    old_reach = (
        {f"{point}|i" for point in _POINTS}
        | {f"zero_polar_summing|{name}" for name in two_wire}
        | {f"zero_polar_not_summing|{name}" for name in two_wire}
    )
    assert len(old_reach) == 19
    assert len(new_reach) == 36
    assert old_reach <= new_reach
    earned = new_reach - old_reach
    assert len(earned) == 17
    # Every earned row is single-wire: the two-wire half already had its reach, and a
    # parameter-free gate cannot be a newly-found identity.
    assert all(row.split("|")[1] not in two_wire for row in earned)
    assert all(row.split("|")[1] != "i" for row in earned)
    assert len(census["removed_by_point"]) == points


def test_the_two_zero_polar_points_are_the_discrimination_the_old_rule_could_not_make(
    payload: dict,
) -> None:
    """One `u3`, and the whole point of reading the declaration instead of the name."""

    census = payload["rule_contract"]["classification_census"]
    summing = set(census["removed_by_point"]["zero_polar_summing"])
    not_summing = set(census["removed_by_point"]["zero_polar_not_summing"])

    assert summing - not_summing == {"u3"}
    assert not_summing - summing == set()
    assert "u3" in summing
    assert "u3" not in not_summing
    # `u2` never removes at any point: its polar angle is the constant `pi/2`.
    for names in census["removed_by_point"].values():
        assert "u2" not in names
    for names in census["removed_by_point_up_to_phase"].values():
        assert "u2" not in names


def test_every_declined_reach_is_published_with_the_mechanism_that_produces_it(
    payload: dict,
) -> None:
    """A refusal with no mechanism is indistinguishable from an oversight.

    Both mechanisms are second-order consequences of reading the declaration, and both
    are named: the shared `rz`/`phase`/`u1` triple, which cannot separate a half turn of
    `phase` from a half turn of `rz`, and the two-wire branch, which reads the raw angle
    parameter.
    """

    census = payload["rule_contract"]["classification_census"]
    by_mechanism = census["declined_reach_by_mechanism"]

    assert census["declined_reach_count"] == len(census["declined_reach"])
    assert census["declined_reach"] == _CENSUS["declined_reach"]
    assert set(by_mechanism) == {
        "shared_z_rotation_triple",
        "two_wire_branch_reads_the_raw_parameter",
    }
    flattened = [row for rows in by_mechanism.values() for row in rows]
    assert sorted(flattened) == sorted(census["declined_reach"])
    assert len(flattened) == len(set(flattened))
    # The shared-triple rows are the two single-wire ones; every other refusal is the
    # two-wire branch, which is the branch that was retained rather than rewritten.
    assert by_mechanism["shared_z_rotation_triple"] == [
        "half_turn|phase",
        "half_turn|u1",
    ]
    assert all(
        row.split("|")[1] not in _RULE["declared_single_qubit_opcodes"]
        or row in by_mechanism["shared_z_rotation_triple"]
        for row in by_mechanism["two_wire_branch_reads_the_raw_parameter"]
    )
    # A row the predicate is not asked about is a row of the *retained* branch. Every
    # two-wire refusal is such a row -- the branch is asked about it and answers no --
    # and the only refusals that are not are the two single-wire rows the shared triple
    # produces, which is exactly the split the mechanisms describe.
    not_asked = set(census["two_wire_rows_the_predicate_is_not_asked_about"])
    assert (
        len(not_asked) == census["two_wire_rows_the_predicate_is_not_asked_about_count"]
    )
    declined = set(census["declined_reach"])
    assert declined - not_asked == set(by_mechanism["shared_z_rotation_triple"])
    assert not_asked >= (declined & not_asked)


def test_the_band_is_the_shipped_tolerance_and_both_edges_hold(payload: dict) -> None:
    """The rule introduces no new tolerance and reaches a full period either side of zero."""

    boundary = payload["boundary_table"]
    assert boundary["residues"] == payload["boundary_table"]["residues"]
    for name, family in boundary["families"].items():
        assert family["removed_at_zero"] is True, name
        assert family["removed_at_the_tolerance"] is True, name
        assert family["kept_just_outside_the_tolerance"] is True, name
        # A single-wire rotation is exactly `+I` at `4*pi` and exactly `-I` at `2*pi`,
        # so the fold must reach the first and refuse the second. A two-wire rotation
        # is a different matter and the next test speaks for it.
        if family["arity"] == 1:
            assert family["removed_at_the_fold_edge"] is True, name
            assert family["at_four_pi"]["is_exactly_identity"] is True, name
            assert family["at_four_pi"]["removed"] is True, name
            # `phase` and `u1` are `+I` at a half period and are still declined; every
            # other single-wire family is `-I` there. The band is reported per family
            # precisely because these two would otherwise hide in an average.
            assert family["at_two_pi"]["is_exactly_identity"] is (
                name in {"phase", "u1"}
            ), name
        else:
            assert family["removed_at_the_fold_edge"] is False, name
            assert family["at_four_pi"]["is_exactly_identity"] is True, name
            assert family["at_four_pi"]["removed"] is False, name
    # `phase`/`u1` are the exception above and the reason the band is reported per
    # family: they *are* the identity at `2*pi` and are still declined.
    for name in ("phase", "u1"):
        family = boundary["families"][name]
        assert family["at_two_pi"]["is_exactly_identity"] is True, name
        assert family["at_two_pi"]["removed"] is False, name


def test_the_fold_limit_refuses_rather_than_inventing_a_removal(payload: dict) -> None:
    fold = payload["boundary_table"]["fold"]

    assert fold["limit_radians"] == _MODULAR_FOLD_LIMIT
    assert fold["limit_in_two_pi"] == 1024.0
    assert fold["at_limit_removed"] is True
    assert fold["past_limit_removed"] is False
    assert fold["far_past_limit_removed"] is False
    assert "float64" in payload["rule_contract"]["modular_fold_limit_rationale"]


def test_a_population_that_was_not_shipped_shorter_says_so(payload: dict) -> None:
    rows = payload["pipeline_delta"]

    for row in rows:
        label = row["label"]
        expected = _DELTA[label]
        for key, value in expected.items():
            assert row[key] == value, (label, key)
        assert (
            row["source_instruction_count"] > row["optimized_instruction_count"]
        ), label
        assert row["removed_by_the_rule_alone"] > 0, label
        assert row["removed_by_the_rule_in_the_pipeline"] >= 0, label
        assert row["changed_circuit_count"] <= row["circuit_count"], label
        assert row["executed_circuit_count"] == row["circuit_count"], label
        # The clamped column can never go negative, so the signed one carries the sign,
        # and this is the identity that ties the two populations together. The clamped
        # column is a sum of per-circuit `max(0, ...)`, so it does not compose into the
        # same identity; reading them as if it did is how a lengthened population would
        # get reported as a tie.
        assert row["net_instruction_delta_against_the_superseded_rule"] == (
            row["superseded_rule_instruction_count"]
            - row["optimized_instruction_count"]
        ), label
        assert (
            row["removed_by_the_rule_in_the_pipeline"]
            >= row["net_instruction_delta_against_the_superseded_rule"]
        ), label
    # Exactly one population was lengthened, it is named, and it is lengthened by the
    # half-turn refusal rather than by an arithmetic slip.
    lengthened = [
        row["label"]
        for row in rows
        if row["net_instruction_delta_against_the_superseded_rule"] < 0
    ]
    assert lengthened == ["full_turn_rotations"]
    assert set(rows[0]) == set(_DELTA[rows[0]["label"]]) | {
        "label",
        "max_probability_vector_difference",
    }
    assert (
        sum(row["net_instruction_delta_against_the_superseded_rule"] for row in rows)
        > 0
    )
    assert sum(row["removed_by_the_rule_alone"] for row in rows) == 963


def test_the_pass_is_measured_against_an_execution_on_every_population(
    payload: dict,
) -> None:
    """The correctness gate: no removal moved what a device would report.

    The instrument here is the *probability* vector and not the statevector, and the
    distinction is deliberate. `remove_diagonal_gates_before_measure` legitimately
    changes amplitudes while leaving a terminal measurement's distribution alone, so a
    statevector comparison at this level would report that pass's correct change as if
    it belonged to this rule. The statevector instrument lives in the control below,
    where the removal under inspection is driven by hand.
    """

    for row in payload["pipeline_delta"]:
        assert (
            row["max_probability_vector_difference"]
            <= payload["execution_control"]["tolerance"]
        ), row["label"]


def test_the_instrument_has_controls_that_moved_and_controls_that_did_not(
    payload: dict,
) -> None:
    """A comparison that cannot fail is not evidence.

    Two controls delete an instruction by hand that *is* an exact identity and must be
    invisible to both instruments. A third deletes one that is only the identity up to
    a global phase -- invisible in the distribution, visible in the statevector -- and
    that pair of readings is what proves the statevector instrument is live. A fourth
    is near a multiple and a fifth is plainly wrong, and both must move the
    distribution.
    """

    control = payload["execution_control"]
    rows = control["rows"]

    exact = rows["a_removal_that_is_exact"]
    full_period = rows["a_full_period_removal"]
    phase = rows["a_removal_declined_because_it_is_only_a_global_phase"]
    near = rows["a_removal_declined_because_it_is_only_near_a_multiple"]
    wrong = rows["a_removal_that_is_wrong"]

    assert exact["kept_by_the_rule"] is False
    assert full_period["kept_by_the_rule"] is False
    for row in (exact, full_period):
        assert row["max_outcome_distribution_difference"] == 0.0
        assert row["statevector"]["max_entry_difference"] <= control["tolerance"]
    # The refusal: the rule keeps the gate, and dropping it by hand would move the
    # statevector by a full `sqrt(2)` while leaving the distribution alone.
    assert phase["kept_by_the_rule"] is True
    assert phase["max_outcome_distribution_difference"] == 0.0
    assert phase["statevector"]["max_entry_difference"] > 1.0
    assert phase["statevector"]["fidelity_deficit"] <= control["tolerance"]
    assert near["kept_by_the_rule"] is True
    assert wrong["kept_by_the_rule"] is True
    assert wrong["max_outcome_distribution_difference"] > control["floor"]
    assert control["tolerance"] < control["floor"]


def test_the_anchor_is_compared_on_the_column_its_own_instrument_can_see(
    payload: dict,
) -> None:
    """The Qiskit anchor is a fidelity band with a sign-blind test, and it is cited as one.

    `RemoveIdentityEquivalent` is absent from Qiskit 1.2.4, so the comparison is against
    2.0.3. It removes any gate whose average gate fidelity with the identity clears a
    cutoff rather than only an exact identity, and its fidelity test cannot see the sign,
    so it is compared with the phase-blind column. Every one of the 17 rows where it
    differs from this rule is a row where the runtime matrix is minus the identity, and
    the benchmark raises rather than reporting an unclassified disagreement.
    """

    anchor = payload["reference_anchor"]
    if not anchor["available"]:
        pytest.skip(f"the Qiskit anchor is not importable here: {anchor['reason']}")

    for key, expected in _ANCHOR.items():
        assert anchor[key] == expected, key
    assert (
        payload["anchor_disagreement_count"]
        == _ANCHOR["disagreement_with_the_rule_count"]
    )
    assert payload["anchor_row_count"] == _ANCHOR["row_count"]
    assert "phase-blind" in payload["anchor_agreement_scope"]
    assert anchor["row_count"] == len(
        payload["rule_contract"]["classification_census"]["rows"]
    )
    disagreements = anchor["disagreements_with_the_rule"]
    assert sorted(disagreements) == _ANCHOR_DISAGREEMENTS
    assert anchor["disagreement_with_the_rule_count"] == len(disagreements)
    # Sign-blind, so the anchor and the phase-blind column agree everywhere -- which is
    # what makes the disagreement count a statement about phases alone.
    assert anchor["agrees_with_the_up_to_phase_column_on_every_row"] is True
    minus = set(
        payload["rule_contract"]["classification_census"][
            "removed_up_to_phase_and_not_exactly"
        ]
    )
    # Every single-wire disagreement is a `-I` row; the two-wire ones are `-I` at a
    # half turn and `+I` at a full turn, and the two-wire branch declines both.
    assert {row for row in disagreements if row.startswith("half_turn")} == minus | {
        "half_turn|cphase",
        "half_turn|phase",
        "half_turn|u1",
    }
    assert "present_in_qiskit_1_2_4" in anchor
    assert anchor["present_in_qiskit_1_2_4"] is False


def test_the_payload_declares_what_it_is_and_what_it_may_not_be_used_for(
    payload: dict,
) -> None:
    """A local microbenchmark is not a scalability claim."""

    assert payload["schema"] == "flagquantum_compiler_identity_elimination_benchmark_v1"
    assert payload["seed"] == {"sweep": _SWEEP_SEED}
    assert payload["artifact_classification"] == "local_compiler_microbenchmark"
    assert payload["scalability_claim_allowed"] is False
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["reference_algorithm"] == "qiskit_remove_identity_equivalent"
    assert set(_POINTS) == set(
        payload["rule_contract"]["classification_census"]["removed_by_point"]
    )
