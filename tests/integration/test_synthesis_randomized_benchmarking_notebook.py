"""Complete original-sized randomized sequence/inverse simulator experiment."""

import asyncio
import json
import random
from pathlib import Path

import pytest
import torch

import flagquantum as fq

pytest.importorskip("numpy")

from examples.synthesis.applications.benchmarking.randomized_benchmarking.overview.ansatz import (
    GATES,
    LENGTHS,
    Gate,
    build_program,
    fresh_sequences,
    make_model,
)
from examples.synthesis.applications.benchmarking.randomized_benchmarking.overview.reference import (
    circuit_reference,
    dense_reference,
)
from examples.synthesis.applications.benchmarking.randomized_benchmarking.overview.workflow import (
    MODES,
    Experiment,
    Mode,
    execute_batches,
    execute_trial,
    run_experiment,
    source_label_mismatches,
    synthesize_all,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def experiment() -> Experiment:
    return run_experiment()


@pytest.mark.parametrize("length", LENGTHS)
def test_all_original_lengths_have_both_actual_local_execution_routes(
    experiment: Experiment, length: int
) -> None:
    rows = [trial for trial in experiment.trials if trial.length == length]
    assert {row.mode for row in rows} == set(MODES)
    assert len(rows) == 2
    for row in rows:
        assert row.shots == 1024
        assert sum(row.counts) == row.shots
        assert row.counts == (1024, 0)
        assert row.survival == 1
        assert row.state_error < 2e-12
        assert row.representation_error < 2e-12
        assert row.resources.n_qubits == 1
        assert row.resources.operation_counts.get("cx", 0) == 0
        assert row.resources.channel_count == 0


def test_original_product_order_batching_and_result_identity(
    experiment: Experiment,
) -> None:
    assert experiment.batch_sizes == (3, 3, 3, 1)
    assert tuple((row.length, row.mode) for row in experiment.trials) == tuple(
        (length, mode) for length in LENGTHS for mode in MODES
    )
    assert len(experiment.programs) == 5
    assert len({id(program) for program in experiment.programs}) == 5
    assert tuple(map(len, experiment.sequences)) == LENGTHS
    assert sum(sum(row.counts) for row in experiment.trials) == 10240
    assert len({row.seed for row in experiment.trials}) == 10
    assert experiment.source_label_mismatches == tuple(range(1, 9))


def test_fresh_seeded_source_sequences_preserve_filter_order() -> None:
    rng = random.Random(0)
    reference = tuple(
        tuple(rng.choice(("id", "rz", "sx", "x")) for _ in range(length))
        for length in (5, 10, 15, 20, 25)
    )
    assert fresh_sequences() == reference
    assert fresh_sequences()[0] == ("x", "x", "id", "sx", "x")
    assert fresh_sequences(1) != reference
    assert GATES == ("id", "rz", "sx", "x")


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("initial", (0, 1, 2))
def test_forward_prefixes_independently_protect_gate_and_absolute_phase_semantics(
    length: int, initial: int
) -> None:
    sequence = next(word for word in fresh_sequences() if len(word) == length)
    for inverse in (False, True):
        actual = (
            fq.run(build_program(sequence, with_inverse=inverse, initial=initial))
            .statevector()
            .flatten()
        )
        circuit = circuit_reference(sequence, with_inverse=inverse, initial=initial)
        dense = torch.from_numpy(
            dense_reference(sequence, with_inverse=inverse, initial=initial)
        )
        torch.testing.assert_close(actual, circuit, atol=2e-12, rtol=0)
        torch.testing.assert_close(actual, dense, atol=2e-12, rtol=0)
        assert actual.numel() == 2


@pytest.mark.parametrize("gate", GATES)
def test_single_gate_complete_columns_include_scalar_phase(gate: Gate) -> None:
    for initial in (0, 1, 2):
        actual = (
            fq.run(build_program((gate,), with_inverse=False, initial=initial))
            .statevector()
            .flatten()
        )
        expected = torch.from_numpy(
            dense_reference((gate,), with_inverse=False, initial=initial)
        )
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)
    if gate == "sx":
        actual = (
            fq.run(build_program(("sx",), with_inverse=False)).statevector().flatten()
        )
        rx_only = torch.tensor([2**-0.5, -1j * 2**-0.5], dtype=torch.complex128)
        assert float((actual - rx_only).abs().max()) > 0.5


@pytest.mark.parametrize("seed", (-1, True, 1.5, 2**63))
def test_invalid_seed_refused_before_random_generation(seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        fresh_sequences(seed)


def test_invalid_shapes_modes_and_counts_fail_without_submission(
    experiment: Experiment,
) -> None:
    with pytest.raises(ValueError, match="five"):
        asyncio.run(
            synthesize_all(
                tuple(make_model(gates) for gates in experiment.sequences[:-1])
            )
        )
    with pytest.raises(ValueError, match="one program"):
        asyncio.run(execute_batches(experiment.programs[:-1], experiment.sequences))
    with pytest.raises(ValueError, match="ten"):
        source_label_mismatches(experiment.trials[:-1])
    with pytest.raises(ValueError, match="admitted"):
        execute_trial(
            experiment.programs[0],
            experiment.sequences[0],
            "statevector",
            shots=True,
            seed=0,
        )


def test_notebook_source_stages_hashes_and_native_provenance() -> None:
    root = Path(__file__).resolve().parents[2]
    directory = (
        root
        / "examples/synthesis/applications/benchmarking/randomized_benchmarking/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [1, 3, 5, 7]
    assert all(len(row["sha256"]) == 64 for row in proof["reviewed_code_cells"])
    assert all(
        not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


@pytest.mark.parametrize("mode", ("mps", "auto", "unknown"))
def test_invalid_local_mode_is_not_silently_substituted(
    experiment: Experiment, mode: Mode
) -> None:
    with pytest.raises(ValueError, match="admitted"):
        execute_trial(
            experiment.programs[0], experiment.sequences[0], mode, shots=1024, seed=0
        )
