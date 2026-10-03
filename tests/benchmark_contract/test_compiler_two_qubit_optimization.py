"""Contract for the two-qubit run-folding measurement W9-13 records.

The benchmark answers two questions that fail differently and are therefore asked
separately. **How much is there to fold** is the reach, measured against the runtime
gate table. **How often is the fold wrong** is `false_yes` -- runs the pass replaced
with a gate whose matrix differs from the run's own -- and it has to stay zero,
because a false acceptance changes the program and `CircuitIR` has no field for the
global phase a wrong answer would leak.

The third question -- did the target get cheaper -- is measured here as well, and
this is the round that found it cannot decide the question on its own. See
``test_the_target_legal_count_is_not_monotone_in_program_length``.
"""

import random

import pytest

from benchmarks.compiler_identity_elimination import (
    _CIRCUITS_PER_POPULATION as IDENTITY_CIRCUIT_COUNT,
)
from benchmarks.compiler_identity_elimination import (
    _POPULATIONS as IDENTITY_POPULATIONS,
)
from benchmarks.compiler_identity_elimination import (
    _substituted_rule,
    _superseded_rule,
)
from benchmarks.compiler_two_qubit_optimization import (
    BOUNDARY_SAMPLE_COUNT,
    FOLD_POLICIES,
    POPULATION_SEEDS,
    RUN_LENGTHS,
    _fold_patch,
    run_benchmark,
)
from flagquantum.compiler.pipeline import optimize

pytestmark = pytest.mark.benchmark_contract

#: Mean optimized length per source run length. Pinned because a fold that stopped
#: composing -- or one that started re-spelling a run as an equal-length one --
#: would move these, and the direction is what the pass claims.
_MEAN_OPTIMIZED_LENGTH = {
    2: 1.7625,
    3: 2.8363,
    4: 3.9238,
    5: 4.9675,
    6: 5.9612,
}

#: The four policies the reach table is measured under, in the order the module
#: declares them. ``off`` is the pass patched out, ``shipped`` is the module's own
#: rule, ``identity`` keeps only the product-is-the-identity arm, and ``disabled``
#: turns the vocabulary decline off.
assert FOLD_POLICIES == ("off", "identity", "shipped", "disabled")

#: The basis the fold is a clear win on, and the reach it must keep there: no
#: circuit may end up with a larger target-legal gate count, and the improvement has
#: to stay an order of magnitude above the handful of circuits a rounding change
#: could move.
_WINNING_BASIS = "ibm-rz-sx-cx"
_MINIMUM_IMPROVED = 40

#: The basis where the *target-legal* count goes up, and the assertion that says the
#: count is not a instrument for this pass. On ``ibm-heron-cz`` -- which publishes
#: ``cz`` and no ``cx`` -- the shipped rule raises the total, and so does the
#: ``identity`` arm, which only ever *deletes* a run whose product is exactly the
#: identity. A deletion cannot need more gates; an instrument that says it does is
#: measuring the legalizer's sensitivity to the surrounding program, not the fold.
#: If the legalizer is ever made monotone in program length this assertion should be
#: inverted rather than deleted, and the reach table re-read in both directions.
_LOSSY_BASIS = "ibm-heron-cz"


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_benchmark()


def test_measurement_is_classified_as_a_local_microbenchmark(payload: dict) -> None:
    assert (
        payload["schema"] == "flagquantum_compiler_two_qubit_optimization_benchmark_v1"
    )
    assert payload["artifact_classification"] == "local_compiler_microbenchmark"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert payload["reference_algorithm"] == "qiskit_collect_cliffords"


def test_the_population_is_the_whole_declared_two_qubit_group(payload: dict) -> None:
    # A fold measured on a subset of the group cannot claim the group. The declared
    # arity-2 unitary set is 11 opcodes and the population is all of it.
    declared = payload["declared_two_qubit_unitary_opcodes"]
    assert len(declared) == 11
    assert set(declared) == {
        "cx",
        "cy",
        "cz",
        "swap",
        "crx",
        "cry",
        "crz",
        "cphase",
        "rxx",
        "ryy",
        "rzz",
    }
    assert [row["label"] for row in payload["populations"]] == [
        label for label, _, _ in POPULATION_SEEDS
    ]


def test_every_fold_is_exact(payload: dict) -> None:
    """The evidence that replaces the global-phase field this IR does not have.

    The comparison is the run's own product against the matrix of the gate the pass
    emitted, both from the runtime gate table, so a dropped `-1` shows up here. The
    tolerance is float64 accumulation over a composed product, not a phase.
    """

    for row in payload["run_table"]:
        assert row["worst_product_gap"] < 1e-12, row["run_length"]


def test_the_fold_shortens_a_run_and_never_lengthens_one(payload: dict) -> None:
    rows = {row["run_length"]: row for row in payload["run_table"]}
    assert set(rows) == set(RUN_LENGTHS)
    folded_by_length = []
    for length, row in rows.items():
        assert row["trial_count"] == 800, length
        assert row["folded_gates"] <= row["source_gates"], length
        assert row["mean_folded_gates"] == pytest.approx(
            _MEAN_OPTIMIZED_LENGTH[length], abs=1e-4
        ), length
        if length > 2:
            # The fold is taken only when it is strictly shorter, so the composed
            # length can never exceed the source length.
            assert row["folded_gates"] < row["source_gates"], length
        folded_by_length.append(row["mean_folded_gates"])
    # A longer run composes further, which is what composition buys over cancellation.
    assert folded_by_length == sorted(folded_by_length)
    assert folded_by_length[-1] > folded_by_length[0]


def test_the_whole_program_statevector_is_preserved(payload: dict) -> None:
    state = payload["state_evidence"]
    assert state["entangler_present"] is True
    assert state["trial_count"] == 1200
    assert state["max_raw_state_difference"] < 1e-11
    # Non-vacuity: the population has to actually shrink, and the fold has to have
    # actually been taken, or this compares a program with itself.
    assert state["circuits_folded"] > 800
    assert state["instructions_after"] < state["instructions_before"]
    assert state["reduction_ratio"] > 1.2


def test_the_boundary_sweep_finds_a_reach_and_never_a_false_yes(payload: dict) -> None:
    """The soundness instrument, and the number that licenses the pass.

    The first two lengths are enumerated over the whole 18-atom alphabet of declared
    two-qubit gates at the angles the fold reaches, the third is a seeded sample
    because the enumeration is ``18 ** 4``. Every run the pass replaced is compared
    against the run's own product from the runtime gate table.
    """

    boundary = payload["boundary"]
    assert boundary["atom_count"] == 18
    assert boundary["false_yes"] == 0
    assert boundary["worst_replacement_gap"] < 1e-12
    # Non-vacuity: the sweep has to have actually exercised the pass.
    assert boundary["checked_against_the_gate_table"] > 300
    assert boundary["runs_the_pass_replaced"] > 300
    assert boundary["runs_the_pass_deleted"] > 50
    rows = {row["run_length"]: row for row in boundary["runs_per_length"]}
    assert rows[2]["sweep"] == "enumerated"
    assert rows[2]["run_count"] == 18**2
    assert rows[3]["sweep"] == "enumerated"
    assert rows[3]["run_count"] == 18**3
    assert rows[4]["sweep"] == "seeded_sample"
    assert rows[4]["run_count"] == BOUNDARY_SAMPLE_COUNT
    assert rows[4]["enumerated_run_count"] == 18**4
    # The reach is why the sweep is not vacuous: runs whose product *is* the
    # identity exist at every enumerated length.
    assert rows[2]["identity_products"] == 7
    assert rows[3]["identity_products"] == 24
    assert rows[4]["identity_products"] > 0


def test_the_fold_is_a_win_on_a_basis_that_publishes_cx(payload: dict) -> None:
    """The per-circuit claim, on every population, with no circuit regressing.

    A total can hide a program that grew behind programs that shrank, so the
    regressions are asserted per circuit and the improvement is asserted as a floor
    rather than a pin: the count on the winning side is read after the target's
    lowering and rounds differently on a different machine.
    """

    rows = [row for row in payload["decline_reach"] if row["basis"] == _WINNING_BASIS]
    assert len(rows) == len(POPULATION_SEEDS)
    for row in rows:
        label = row["label"]
        assert row["shipped_improved"] >= _MINIMUM_IMPROVED, label
        assert row["shipped_regressed"] == 0, label
        assert row["shipped_legal"] < row["off_legal"], label
        # The intermediate program is the pass's own output, with no synthesis in
        # it, so this part is exact rather than banded.
        assert row["shipped_optimized"] < row["off_optimized"], label
        # And the identity arm is a strict win on its own, which is the reach
        # `remove_identity_gates` cannot have.
        assert row["identity_regressed"] == 0, label
        assert row["identity_improved"] > 0, label


def test_turning_the_decline_off_is_not_free(payload: dict) -> None:
    """The measurement that makes the vocabulary decline a rule, not a preference.

    Without it the intermediate program is shorter still -- which is the tempting
    argument for removing it -- and the target-legal count regresses on real
    circuits. Both directions are asserted: it is shorter, and it is worse.
    """

    rows = [row for row in payload["decline_reach"] if row["basis"] == _WINNING_BASIS]
    assert rows
    shorter = 0
    regressed = 0
    for row in rows:
        assert row["disabled_optimized"] <= row["shipped_optimized"], row["label"]
        if row["disabled_optimized"] < row["shipped_optimized"]:
            shorter += 1
        if row["disabled_regressed"] > 0:
            regressed += 1
    # Non-vacuity on both sides of the argument.
    assert shorter > 0
    assert regressed > 0


def test_the_target_legal_count_is_not_monotone_in_program_length(
    payload: dict,
) -> None:
    """The instrument that could not decide this pass, and the proof of it.

    ``legalize_native_gates`` returns a *program*, and its cost is not a function of
    the input program's length: removing an exact identity can change how the
    surrounding gates are blocked and synthesized, and the target's gate count can
    go **up**. The ``identity`` arm is the proof, because it only ever deletes a run
    whose product is exactly the identity -- and on a ``cz``-only basis it raises the
    total on real circuits.

    This is recorded rather than worked around: it means the legal count is evidence
    about the target's synthesis, and a fold's *own* claim has to rest on the
    intermediate program plus the exactness and reach measurements above.
    """

    rows = [row for row in payload["decline_reach"] if row["basis"] == _LOSSY_BASIS]
    assert rows
    for row in rows:
        assert row["identity_improved"] > 0, row["label"]
    assert any(row["identity_regressed"] > 0 for row in rows)
    # And the deletion can make the total larger, not merely a circuit or two.
    assert any(row["identity_legal"] > row["off_legal"] for row in rows)


def test_no_run_is_deleted_unless_its_product_is_the_identity(payload: dict) -> None:
    """Deletion is the sharpest edge: a wrong deletion changes the program.

    Every deletion in the boundary sweep is a run whose product the same instrument
    found to be the identity, so the two counts below are the same statement read two
    ways, and the sweep's reach is what makes the statement non-vacuous.
    """

    boundary = payload["boundary"]
    deleted = boundary["runs_the_pass_deleted"]
    assert deleted > 0
    identity_products = sum(
        row["identity_products"] for row in boundary["runs_per_length"]
    )
    # The sweep enumerates whole runs, so every deleted run is one of the reaches
    # the same sweep counted -- deletion is a subset of the identity reach, never a
    # separate mechanism.
    assert deleted <= identity_products


def test_the_fold_is_what_moved_the_round_18_pipeline_rows() -> None:
    """The one measurement this round changed outside its own files, attributed.

    ``compiler_identity_elimination`` publishes seven populations measured through the
    *whole* pipeline, so a pass added later can move a row it does not own. This one
    did: the fold composes adjacent two-qubit rotations, which is what the two-wire
    and mixed populations are built out of.

    The instrument is the fold patched out of the loop, which isolates its
    contribution by construction -- every other pass is the same object both times.
    Two things are asserted. The pre-fold totals reproduce the values that module
    pinned before this round, so the move is attributable to this pass by measurement
    rather than by argument. And no other population moves at all, so the attribution
    stays a statement about three rows rather than about the pipeline in general. If a
    later pass starts composing two-qubit runs the second assertion fails, which is
    where those pins would otherwise have to be re-derived by hand.
    """

    #: ``population -> ((superseded rule, shipped rule) with the fold out, ... with it
    #: in)``. Each pair is the row's two post-pipeline counts, which are exactly what
    #: `compiler_identity_elimination` pins.
    expected = {
        "two_wire_rotations": ((245, 245), (201, 201)),
        "mixed": ((406, 391), (398, 383)),
        "mixed_with_mid_circuit_measures": ((511, 498), (505, 493)),
    }
    #: Populations the fold has no reach in, whose rows must therefore not move.
    untouched = {label for label, _ in IDENTITY_POPULATIONS if label not in expected}
    assert untouched  # non-vacuity: there has to be a control group

    measured: dict[str, dict[str, tuple[int, int]]] = {}
    for label, factory in IDENTITY_POPULATIONS:
        circuits = [
            factory(random.Random(seed)) for seed in range(IDENTITY_CIRCUIT_COUNT)
        ]
        for policy in ("off", "shipped"):
            with _fold_patch(policy):
                with _substituted_rule(_superseded_rule):
                    superseded = sum(len(optimize(circuit)) for circuit in circuits)
                shipped = sum(len(optimize(circuit)) for circuit in circuits)
            measured.setdefault(label, {})[policy] = (superseded, shipped)

    for label, (before, after) in expected.items():
        assert measured[label]["off"] == before, label
        assert measured[label]["shipped"] == after, label
        assert before != after, label
    for label in untouched:
        assert measured[label]["off"] == measured[label]["shipped"], label
    # And where it did move at all, it moved in the pass's own claimed direction.
    for label in expected:
        assert measured[label]["shipped"][0] <= measured[label]["off"][0], label
        assert measured[label]["shipped"][1] <= measured[label]["off"][1], label


def test_the_qiskit_anchor_reports_a_no_op_rather_than_claiming_one(
    payload: dict,
) -> None:
    """``OptimizeCliffords`` is not the pass this module implements, and says so.

    On a gate-level circuit Qiskit's ``OptimizeCliffords`` leaves the circuit
    unchanged and ``CollectCliffords`` emits one opaque ``clifford`` operation, which
    needs a native Clifford type FlagQuantum does not have. The anchor records the
    measured no-op; it is not an agreement claim.
    """

    anchor = payload["qiskit_anchor"]
    if not anchor["available"]:
        pytest.skip(f"Qiskit not importable: {anchor['reason']}")
    optimize_cliffords = anchor["optimize_cliffords"]
    assert optimize_cliffords["gates_after"] == optimize_cliffords["gates_before"]
    assert optimize_cliffords["opcodes_after"] == [
        "cx",
        "cz",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
    ]
    # CollectCliffords does shrink the circuit, and the operation it leaves behind
    # is the opaque one -- which is the reason the port cannot use it. The non-
    # Clifford gates survive and split the circuit, so the count is not one gate.
    collect_cliffords = anchor["collect_cliffords"]
    assert collect_cliffords["gates_after"] < collect_cliffords["gates_before"]
    assert "clifford" in collect_cliffords["opcodes_after"]
    assert "t" in collect_cliffords["opcodes_after"]
