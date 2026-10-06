"""Contract for the one-qubit run-folding measurement W9-05 records.

The benchmark module answers one question: what does composing a same-wire run of
single-qubit gates into one exact sequence buy, given that `CircuitIR` has no
global-phase field? The recorded answer is that the fold shortens a run to a mean
of 1.33 to 1.88 gates, that it reproduces the program's statevector exactly, and
that it deliberately declines a run one target vocabulary already spells.

These tests hold that evidence in place. The exactness test is the one that must
not be weakened: it is the evidence that replaces the global-phase field Qiskit
has and this IR does not.
"""

import pytest

from benchmarks.compiler_one_qubit_optimization import (
    RUN_LENGTHS,
    SINGLE_QUBIT_OPCODES,
    run_benchmark,
)
from tests.benchmark_contract.qiskit_lane import require_certified_lane

pytestmark = pytest.mark.benchmark_contract

#: Mean optimized length per source run length. Pinned so that a change which
#: stops emitting the determinant -- and therefore stops being exact -- cannot
#: pass as an improvement.
#:
#: These were re-measured when the declared-inverse pass landed in the same
#: pipeline, and only two of the five moved. ``optimize`` is the shipped pipeline,
#: so the population this table measures is the one that pass also runs on: on a
#: run like ``sx tdg t`` the pair cancels to ``sx`` before the fold can reach
#: ``u3 rz``, which is one gate shorter. Three of the 800 length-3 runs and one of
#: the 800 length-5 runs move, so 1.5625 becomes 1.5588 and 1.8413 becomes 1.8400.
#: Both directions of that interaction are recorded rather than only this one:
#: ``benchmarks/compiler_inverse_cancellation.py`` measures the two orders of the
#: two passes on its own populations and reports where cancelling first is the
#: *longer* program instead. Nothing here is loosened by the re-measurement -- the
#: determinant, the matrix gap, and the ordering of the ratios are all still
#: asserted exactly as before, and a fold that regressed would raise the means.
_MEAN_OPTIMIZED_LENGTH = {
    2: 1.3325,
    3: 1.5588,
    4: 1.7237,
    5: 1.8400,
    6: 1.8838,
}

#: What the target bases reach on the ``rz``/``sx`` program population. The four
#: that publish a z-rotation and a pulse agree; ``clifford-t`` publishes neither
#: and refuses every circuit.
#:
#: The two counts on the *declining* side are exact. That path leaves an
#: ``rz``/``sx`` run exactly as it found it and every one of those opcodes is
#: native to these bases, so the number is the program's own length and no
#: synthesis touches it.
#:
#: The post-lowering count on the *folded* side is not exact, and pinning it as
#: an integer was measured wrong: the fold emits a ``u3`` whose angles come from
#: ``atan2``, and whether the target's Euler synthesis can then spell that ``u3``
#: one gate shorter depends on whether an angle lands exactly on ``0`` or ``pi``
#: -- a transcendental rounding, not a program property. The same program
#: population gives 907 here, 893 in the CI lane, and 898 to 908 when the input
#: angles are displaced by a single part in ``1e15``. The claim is the direction,
#: so the direction is what is asserted, with the folded side bounded by ratio.
_LEGAL_GATES_DECLINING = 771
_OPTIMIZED_GATES_DECLINING = 771
_OPTIMIZED_GATES_NOT_DECLINING = 384
_LEGAL_GATES_NOT_DECLINING_LOCAL = 907
_LEGAL_GATES_NOT_DECLINING_SPREAD = 30
_REFUSING_BASIS = "clifford-t"

#: The identity population's legal gate count under the decline as it was, and the
#: band it is allowed to land in. The first draft of this made both arms exact, on
#: the reasoning that neither one synthesises here -- and CI returned 684 where the
#: development checkout returned 685 on the very first run. That is the lesson the
#: ``_LEGAL_GATES_NOT_DECLINING_LOCAL`` band above already carries: this count is
#: read *after* the target's lowering, so it is a synthesis outcome and it rounds
#: differently on a different machine. The assertions are on directions with margin
#: instead.
#:
#: The difference between the two rules is the number that matters, and it has to
#: clear the 8 gates the same exemption moves on the seeded control population by an
#: order of magnitude. Locally it is 174 and in CI 173; the floor is deliberately far
#: below either so that it fails only when the exemption stops working, not when the
#: lowering rounds.
_IDENTITY_LEGAL_GATES_STRICT = 685
_IDENTITY_STRICT_SPREAD = 3
_IDENTITY_EXEMPTION_MINIMUM_GAIN = 150


@pytest.fixture(scope="module")
def payload() -> dict:
    return run_benchmark()


def test_measurement_is_classified_as_a_local_microbenchmark(payload: dict) -> None:
    assert (
        payload["schema"] == "flagquantum_compiler_one_qubit_optimization_benchmark_v1"
    )
    assert payload["artifact_classification"] == "local_compiler_microbenchmark"
    assert payload["distribution_semantics"] == "single_device_fast_path"
    assert payload["scalability_claim_allowed"] is False
    assert payload["reference_algorithm"] == "qiskit_optimize_1q_gates"
    assert len(payload["declared_single_qubit_unitary_opcodes"]) == len(
        SINGLE_QUBIT_OPCODES
    )


def test_the_population_is_the_whole_declared_single_qubit_group(
    payload: dict,
) -> None:
    # A fold measured on a subset of the group cannot claim the group. The
    # declared arity-1 unitary set is 18 opcodes and the population is all of it.
    assert len(SINGLE_QUBIT_OPCODES) == 18
    assert set(SINGLE_QUBIT_OPCODES) == set(
        payload["declared_single_qubit_unitary_opcodes"]
    )
    assert "u2" in SINGLE_QUBIT_OPCODES
    assert "u3" in SINGLE_QUBIT_OPCODES
    assert "rz" in SINGLE_QUBIT_OPCODES


def test_every_fold_is_exact(payload: dict) -> None:
    """The evidence that replaces the global-phase field this IR does not have.

    Qiskit's pass may drop the leftover determinant onto ``dag.global_phase``.
    This one may not, so it has to reproduce the run matrix including the phase.
    The tolerance is float64 accumulation over a composed run, not a phase: a
    dropped phase on these entangled programs shows up far above it.
    """

    for row in payload["fold"]:
        assert row["worst_matrix_gap"] < 1e-12, row["run_length"]


def test_the_whole_program_statevector_is_preserved(payload: dict) -> None:
    state = payload["state"]
    assert state["entangler_present"] is True
    assert state["trial_count"] == 1200
    assert state["max_raw_state_difference"] < 1e-11
    # Non-vacuity: the population has to actually shrink, and the fold has to
    # actually have been taken, or this compares a program with itself.
    assert state["circuits_folded"] > 1000
    assert state["instructions_after"] < state["instructions_before"]
    assert state["reduction_ratio"] > 1.4


def test_the_determinant_is_emitted_rather_than_dropped(payload: dict) -> None:
    """The measured size of the gap to Qiskit's global-phase field.

    A bare ``U3`` cannot carry the run's determinant, so the folds that are not a
    bare ``U3`` are the folds that cost a second gate here and one gate in
    Qiskit. If a future IR gains a global-phase field this number is what it
    would remove.
    """

    split = payload["phase_split"]
    assert split["trial_count"] == 1000
    assert 0.0 < split["bare_share"] < 1.0
    assert split["needing_an_emitted_rotation"] > 500


def test_the_fold_shortens_a_run(payload: dict) -> None:
    rows = {row["run_length"]: row for row in payload["fold"]}
    assert set(rows) == set(RUN_LENGTHS)
    for length, row in rows.items():
        assert row["trial_count"] == 800, length
        assert row["mean_optimized_length"] == pytest.approx(
            _MEAN_OPTIMIZED_LENGTH[length], abs=1e-4
        ), length
        # The fold is only taken when it is strictly shorter, so no run grows.
        assert row["optimized_gates"] <= row["source_gates"], length
        assert row["reduction_ratio"] > 1.0, length
    # Longer runs fold further, which is what composition buys over cancellation.
    ratios = [rows[length]["reduction_ratio"] for length in RUN_LENGTHS]
    assert ratios == sorted(ratios)


def test_the_vocabulary_decline_saves_gates_after_lowering(payload: dict) -> None:
    """The measurement that makes the decline a rule rather than a preference.

    Without the decline the fold is strictly shorter on the intermediate program
    (384 against 771) and strictly longer after the target's own lowering (907
    against 771 here). Both halves have to hold, or the rule is not justified.

    The ``shipped`` arm is in the table because the rule now has an exemption, and
    the exemption must not be able to answer this test for it: the comparison is
    made against the shipped rule as it behaves on this population, which is 763
    rather than 771, and the direction is unchanged. If a future change made the
    fold fire on the rest of these runs, the shipped arm would fall toward the
    no-decline arm and the direction asserted on the last line would be what
    caught it.

    Both halves are asserted as directions, because only the directions survive a
    change of platform's transcendental library. See the note on
    `_LEGAL_GATES_NOT_DECLINING_LOCAL`: the folded side's post-lowering count is a
    synthesis outcome, and this test measures it as a ratio rather than pinning an
    integer that only holds on the machine it was first read on.
    """

    measured = {row["label"]: row for row in payload["basis_reach"]}
    assert len(measured) == 5
    for label, row in measured.items():
        if label == _REFUSING_BASIS:
            assert row["refused_circuits"] == 80, label
            assert row["strict_rule"]["legal_gates"] == 0, label
            assert row["shipped_rule"]["legal_gates"] == 0, label
            continue
        assert row["strict_rule"]["optimized_gates"] == _OPTIMIZED_GATES_DECLINING
        assert row["strict_rule"]["legal_gates"] == _LEGAL_GATES_DECLINING
        assert (
            row["no_decline_rule"]["optimized_gates"] == _OPTIMIZED_GATES_NOT_DECLINING
        )
        # The exemption moves this population by eight gates, against a gain of
        # 176 on the identity population. The asymmetry is the evidence that the
        # exemption is about the identity rather than about the vocabulary.
        assert (
            row["shipped_rule"]["optimized_gates"] == _OPTIMIZED_GATES_DECLINING - 8
        ), label
        assert row["shipped_rule"]["legal_gates"] < row["strict_rule"]["legal_gates"]
        assert row["refused_circuits"] == 0, label
        assert (
            row["no_decline_rule"]["optimized_gates"]
            < row["shipped_rule"]["optimized_gates"]
        )
        # Folding looks better on the intermediate program and ends worse after
        # the target's own lowering. The second half is the rule's whole
        # justification, so it is asserted as a direction with margin: the local
        # reading is 907 and the CI lane's is 893, both well inside the band.
        folded_legal = row["no_decline_rule"]["legal_gates"]
        assert (
            abs(folded_legal - _LEGAL_GATES_NOT_DECLINING_LOCAL)
            <= _LEGAL_GATES_NOT_DECLINING_SPREAD
        ), (label, folded_legal)
        assert folded_legal > row["shipped_rule"]["legal_gates"], label
        # Non-vacuity: the ratio has to be about a synthesis that actually ran, so
        # the folded side must still be larger than its own intermediate program.
        assert folded_legal > row["no_decline_rule"]["optimized_gates"], label


def test_the_identity_exemption_shrinks_a_local_program(payload: dict) -> None:
    """What the exemption buys, and the control that bounds what it means.

    The identity population is built out of runs whose product is the identity and
    whose opcodes spell one z-rotation/pulse vocabulary, so it is the only
    population that can see the change at all. The seeded ``rz``/``sx`` population
    is carried beside it as the control: if the exemption were a general
    relaxation of the decline, that population would move with it, and it moves by
    eight gates instead of by 176.
    """

    measured = {row["label"]: row for row in payload["identity_reach"]}
    assert set(measured) == {"identity_vocabulary", "random_vocabulary"}

    identity = measured["identity_vocabulary"]
    strict = identity["strict_rule"]["legal_gates"]
    shipped = identity["shipped_rule"]["legal_gates"]
    assert abs(strict - _IDENTITY_LEGAL_GATES_STRICT) <= _IDENTITY_STRICT_SPREAD, strict
    assert shipped < strict
    assert strict - shipped >= _IDENTITY_EXEMPTION_MINIMUM_GAIN
    # Per circuit, not in total: a program that grew behind programs that shrank
    # is exactly what the aggregate would hide.
    assert identity["circuits_improved"] > 0
    assert identity["circuits_regressed"] == 0
    assert identity["worst_raw_state_difference"] < 1e-12

    control = measured["random_vocabulary"]
    assert control["circuits_regressed"] == 0
    assert control["worst_raw_state_difference"] < 1e-12
    # The control moves in the same direction and by far less. Asserting the
    # ratio rather than two integers is what keeps this a statement about the
    # exemption's reach rather than about the seed.
    identity_gain = strict - shipped
    control_gain = (
        control["strict_rule"]["legal_gates"] - control["shipped_rule"]["legal_gates"]
    )
    assert control_gain >= 0
    # Non-vacuity on the control: ten times zero is zero, so the ratio below would
    # hold for an exemption that did nothing at all. The control has to have been
    # seen to move, however little.
    assert control_gain > 0
    assert identity_gain > 10 * control_gain


def test_the_exemption_is_not_the_decline(payload: dict) -> None:
    """The reach the exemption leaves, measured per circuit on both populations.

    This is the evidence that the change is the part of the decline that was never
    doing anything rather than the decline itself. On the seeded population,
    disabling the decline entirely is a net loss: more circuits get worse than get
    better, and by more gates. On the identity population the same arm is a gain,
    which is honest to record and is the reason the decline is not simply deleted
    here.
    """

    measured = {row["label"]: row for row in payload["identity_reach"]}
    control = measured["random_vocabulary"]
    assert (
        control["disabled_rule_regresses"]["circuits"]
        > control["disabled_rule_improves"]["circuits"]
    )
    assert (
        control["disabled_rule_regresses"]["gates"]
        > control["disabled_rule_improves"]["gates"]
    )
    # Non-vacuity: the further gain on the identity population is real and is
    # recorded rather than hidden, so it must not have silently become zero. If it
    # ever does, this file's account of why the decline stays is out of date.
    identity = measured["identity_vocabulary"]
    assert identity["disabled_rule_improves"]["gates"] > 0
    assert identity["disabled_rule_regresses"]["gates"] == 0


def test_no_run_is_deleted_unless_its_product_is_the_identity(payload: dict) -> None:
    """The fail-closed side, swept densely rather than argued.

    A deletion is the direction that needs evidence, because the vocabulary
    decline was refusing these runs and the exemption removes that refusal. The
    sweep is asserted to have deleted something -- a clean sweep over a population
    that deleted nothing would prove nothing -- and every deletion is asserted to
    be within the tolerance of the identity.
    """

    sweep = payload["identity_boundary"]["sweep"]
    assert sweep["run_count"] == 60000
    assert sweep["readable_run_count"] > 50000
    assert sweep["runs_emitting_nothing"] > 0
    assert sweep["false_yes_count"] == 0
    assert (
        sweep["worst_identity_distance_over_emptied"] < sweep["distance_band_asserted"]
    )


def test_a_minus_identity_run_is_written_back_rather_than_deleted(
    payload: dict,
) -> None:
    """The one edge the exemption must not cross, on named shapes and in the sweep.

    ``-I`` is a global phase. ``CircuitIR`` has no field for one, so a run whose
    product is ``-I`` has to keep a gate. Both halves are asserted: the sweep saw
    products equal to ``-I``, and it deleted none of them. Without the first
    assertion the second would hold on a population that never reached the edge.
    """

    sweep = payload["identity_boundary"]["sweep"]
    assert sweep["minus_identity_product_count"] > 0
    assert sweep["minus_identity_emptied_count"] == 0

    cases = {case["case"]: case for case in payload["identity_boundary"]["cases"]}
    for label, case in cases.items():
        if case["product_is_identity"]:
            assert case["emits_nothing"] is True, label
        if case["product_is_minus_identity"]:
            assert case["emits_nothing"] is False, label
    minus_i = cases["minus_i_of_two_half_turns"]
    assert minus_i["product_is_minus_identity"] is True
    # The run is refused rather than rewritten: one `rz` would be shorter than the
    # two gates it replaces, but the fold declines a run in this vocabulary, and
    # `test_a_minus_identity_vocabulary_run_is_not_deleted` in the unit suite pins
    # that from the pass's own side.
    assert minus_i["replacement_length"] is None
    assert cases["i_identity_of_four_half_pi_pulses"]["replacement_length"] == 0


def test_the_qiskit_anchor_reports_the_global_phase_split(payload: dict) -> None:
    """Qiskit's own ``Optimize1qGates``, when this machine has Qiskit.

    The anchor is skipped rather than failed when Qiskit is absent: this module is
    a local benchmark and CI does not install Qiskit for it. The two counts are
    not asserted equal, because they are not measuring the same thing -- the
    anchor's pass moves the determinant onto a field this IR lacks. What is
    asserted is that the split was measured and that Qiskit's fold is correct.
    """

    anchor = payload["reference_anchor"]
    if not anchor["available"]:
        pytest.skip(f"Qiskit not importable: {anchor['reason']}")
    # The lane, not the pinned revision string: the mean counts below are read off
    # whichever Qiskit is installed, and a reading that does not name its lane
    # cannot be told apart from one taken on a lane that no longer exists. The
    # lane has to be one this repository certifies, because comparing these counts
    # outside the lanes the anchor family was measured on is a comparison with an
    # instrument nobody calibrated.
    require_certified_lane(anchor["qiskit_version"], recording="this anchor")
    assert anchor["pass"] == "Optimize1qGates(basis=['u3','u1'])"
    assert anchor["agrees_up_to_global_phase"] is True
    assert anchor["worst_overlap_gap"] < 1e-12
    without_phase = anchor["qiskit_needed_no_global_phase"]
    with_phase = anchor["qiskit_needed_global_phase"]
    # Both halves of the split are populated, and the runs Qiskit needed the
    # phase for are exactly the runs where its advantage over the port is
    # largest: that is the field, not the composition, doing the work.
    assert without_phase["run_count"] > 0
    assert with_phase["run_count"] > 0
    assert without_phase["run_count"] + with_phase["run_count"] == 4000
    assert (
        with_phase["port_mean_gates"] - with_phase["qiskit_mean_gates"]
        > without_phase["port_mean_gates"] - without_phase["qiskit_mean_gates"]
    )
