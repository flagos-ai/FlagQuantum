"""Complete original-sized fresh random circuits, finite readout and independent states."""

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq

pytest.importorskip("numpy")

import numpy as np

from examples.synthesis.applications.benchmarking.quantum_volume.overview.ansatz import (
    build_program,
    haar_unitary,
    make_schedule,
)
from examples.synthesis.applications.benchmarking.quantum_volume.overview.workflow import (
    Experiment,
    observed_heavy_count,
    run_experiment,
    source_round_significant,
    summarize,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def experiment() -> Experiment:
    return run_experiment()


@pytest.mark.parametrize("size", (3, 4, 5, 6))
def test_all_original_trials_match_two_independent_full_states(
    experiment: Experiment, size: int
) -> None:
    trials = tuple(trial for trial in experiment.trials if trial.size == size)
    assert len(trials) == 10
    assert {trial.index for trial in trials} == set(range(10))
    for trial in trials:
        assert trial.state_error < 2e-11
        assert trial.named_probability_error < 2e-11
        assert sum(count for _, count in trial.counts) == 100
        assert all(0 <= word < 1 << size for word, _ in trial.counts)
        assert trial.resources.n_qubits == size
        assert trial.resources.operation_counts["cx"] == 3 * size * (size // 2)
        assert trial.resources.channel_count == 0


def test_all_forty_native_programs_are_retained(experiment: Experiment) -> None:
    assert len(experiment.programs) == len(experiment.trials) == 40
    assert len({id(program) for program in experiment.programs}) == 40
    assert all(set(program.registers) == {"target"} for program in experiment.programs)
    assert (
        sum(sum(count for _, count in trial.counts) for trial in experiment.trials)
        == 4000
    )


def test_source_statistics_and_first_failure_are_independently_recomputed(
    experiment: Experiment,
) -> None:
    power = 0
    accepting = True
    for row in experiment.sizes:
        counts = [
            trial.observed_heavy_count
            for trial in experiment.trials
            if trial.size == row.size
        ]
        mean = sum(counts) / 10
        deviation = (sum((count - mean) ** 2 for count in counts) / 10) ** 0.5 / 1000
        score = sum(counts) / 1000
        assert row.observed_heavy_probability == pytest.approx(score, abs=1e-15)
        assert row.source_standard_deviation == pytest.approx(deviation, abs=1e-15)
        assert row.lower_bound == pytest.approx(score - 2 * deviation, abs=1e-15)
        if accepting and score - 2 * deviation >= 2 / 3:
            power = row.size
        else:
            accepting = False
    assert experiment.source_volume == 2**power
    changed = tuple(
        replace(trial, observed_heavy_count=1) if trial.size == 4 else trial
        for trial in experiment.trials
    )
    assert summarize(changed).source_volume == 8


def test_observed_median_includes_ties_and_excludes_unobserved_bins() -> None:
    assert observed_heavy_count(((0, 4), (1, 3), (2, 3), (3, 1))) == 10
    assert observed_heavy_count(((0, 1), (1, 1))) == 2
    assert observed_heavy_count(((31, 100),)) == 100
    with pytest.raises(ValueError, match="positive"):
        observed_heavy_count(((0, 0),))
    with pytest.raises(ValueError, match="unique"):
        observed_heavy_count(((0, 1), (0, 2)))


def test_fresh_random_matrices_and_permutation_pairing_are_reproducible() -> None:
    rng = np.random.RandomState(0)
    permutations = random.Random(0)
    first = make_schedule(6, rng, permutations)
    second = make_schedule(6, rng, permutations)
    assert first != second
    assert first == make_schedule(6, np.random.RandomState(0), random.Random(0))
    assert len(first) == 18
    assert len({matrix for _, _, matrix in first}) == 18
    for offset in range(0, 18, 3):
        assert sorted(
            wire for a, b, _ in first[offset : offset + 3] for wire in (a, b)
        ) == list(range(6))
    for _, _, matrix in first:
        unitary = np.asarray(matrix)
        np.testing.assert_allclose(
            unitary.conj().T @ unitary, np.eye(4), atol=2e-14, rtol=0
        )
    factor = haar_unitary(4, np.random.RandomState(19))
    np.testing.assert_allclose(factor.conj().T @ factor, np.eye(4), atol=2e-14, rtol=0)


def test_asymmetric_matrix_and_nonadjacent_tuple_order() -> None:
    # A fresh complex asymmetric schedule distinguishes both tuple and register order.
    schedule = make_schedule(5, np.random.RandomState(17), random.Random(29))
    from examples.synthesis.applications.benchmarking.quantum_volume.overview.workflow import (
        check_and_sample,
    )

    trial = check_and_sample(build_program(5, schedule), 5, schedule, index=0, seed=7)
    assert trial.state_error < 2e-11
    assert trial.named_probability_error < 2e-11
    program = build_program(5, schedule)
    state = fq.run(program).statevector()
    assert isinstance(state, torch.Tensor) and state.numel() == 32


def test_shape_index_and_original_size_admission() -> None:
    schedule = make_schedule(3, np.random.RandomState(0), random.Random(0))
    for size in (True, 2, 7):
        with pytest.raises(ValueError, match="sizes"):
            build_program(size, schedule)
    with pytest.raises(ValueError, match="layers"):
        build_program(3, schedule[:-1])
    first, *rest = schedule
    for wires in ((0, 0), (-1, 2), (0, 3), (True, 1)):
        with pytest.raises(ValueError, match="distinct"):
            build_program(3, ((wires[0], wires[1], first[2]), *rest))


@pytest.mark.parametrize("invalid", (float("nan"), float("inf"), -0.1))
def test_invalid_display_uncertainty_is_rejected(invalid: float) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        source_round_significant(invalid)


def test_zero_and_one_significant_digit_display() -> None:
    assert source_round_significant(0) == 0
    assert source_round_significant(0.003340658617698) == 0.003


def test_every_source_stage_has_a_hashed_native_notebook_cell() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/applications/benchmarking/quantum_volume/overview"
    )
    provenance = json.loads((directory / "provenance.json").read_text())
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    assert [row["index"] for row in provenance["reviewed_code_cells"]] == [
        2,
        4,
        6,
        7,
        8,
        10,
        13,
        16,
        18,
        19,
        20,
    ]
    assert len(provenance["original_sha256"]) == hashlib.sha256().digest_size * 2
    for row in provenance["reviewed_code_cells"]:
        assert notebook["cells"][row["native_code_cell"]]["cell_type"] == "code"


def test_existing_unitary_validation_and_gate_budgets_remain_effective() -> None:
    from examples.synthesis.applications.benchmarking.quantum_volume.overview.ansatz import (
        schedule_function,
    )
    from flagquantum import synth
    from flagquantum.synth import Output, QArray, QBit, allocate, qfunc

    schedule = make_schedule(3, np.random.RandomState(0), random.Random(0))
    sequence = schedule_function(schedule)

    @qfunc
    def root(target: Output[QArray[QBit]]) -> None:
        allocate(3, target)
        sequence(target)

    with pytest.raises(ValueError, match="budget|operation"):
        synth.synthesize(root, max_operations=1)
    with pytest.raises(ValueError, match="qubit|width|budget"):
        synth.synthesize(root, max_qubits=2)
    bad_matrix = ((1 + 0j, 0j, 0j, 0j),) * 4
    broken = ((schedule[0][0], schedule[0][1], bad_matrix), *schedule[1:])
    with pytest.raises(ValueError, match="unitary"):
        build_program(3, broken)
    with pytest.raises(ValueError, match="forty"):
        summarize(())
