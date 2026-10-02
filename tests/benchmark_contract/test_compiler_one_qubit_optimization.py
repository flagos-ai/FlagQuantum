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

pytestmark = pytest.mark.benchmark_contract

#: Mean optimized length per source run length. Pinned so that a change which
#: stops emitting the determinant -- and therefore stops being exact -- cannot
#: pass as an improvement.
_MEAN_OPTIMIZED_LENGTH = {
    2: 1.3325,
    3: 1.5625,
    4: 1.7237,
    5: 1.8413,
    6: 1.8838,
}

#: What the target bases reach on the ``rz``/``sx`` program population. The four
#: that publish a z-rotation and a pulse agree; ``clifford-t`` publishes neither
#: and refuses every circuit.
_LEGAL_GATES_DECLINING = 771
_LEGAL_GATES_NOT_DECLINING = 907
_OPTIMIZED_GATES_DECLINING = 771
_OPTIMIZED_GATES_NOT_DECLINING = 384
_REFUSING_BASIS = "clifford-t"


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
    against 771). Both halves have to hold, or the rule is not justified.
    """

    measured = {row["label"]: row for row in payload["basis_reach"]}
    assert len(measured) == 5
    for label, row in measured.items():
        if label == _REFUSING_BASIS:
            assert row["refused_circuits"] == 80, label
            assert row["declined_rule"]["legal_gates"] == 0, label
            continue
        assert row["declined_rule"]["optimized_gates"] == _OPTIMIZED_GATES_DECLINING
        assert row["declined_rule"]["legal_gates"] == _LEGAL_GATES_DECLINING
        assert (
            row["no_decline_rule"]["optimized_gates"] == _OPTIMIZED_GATES_NOT_DECLINING
        )
        assert row["no_decline_rule"]["legal_gates"] == _LEGAL_GATES_NOT_DECLINING
        assert row["refused_circuits"] == 0, label
        assert (
            row["no_decline_rule"]["optimized_gates"]
            < row["declined_rule"]["optimized_gates"]
        )
        assert (
            row["no_decline_rule"]["legal_gates"] > row["declined_rule"]["legal_gates"]
        )


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
