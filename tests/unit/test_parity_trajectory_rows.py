"""Re-test the two trajectory rows against the runner they describe.

The parity matrix is a contract, so a sentence in it is a claim. Two rows in it
described the same artefact and disagreed: the noise row said no batched
trajectory result was recorded here, while the execution row said trajectory
simulation exists and only its batching was narrower. This module settles the
disagreement by measuring the runner first, then asserting the contract says
what the measurement says.

The measurement is the interesting half. ``run_noisy_statevector`` is the public
path, and it returns a ``BatchedStatevectorTrajectoryResult`` whose
``trajectory_batch_size`` is the caller's, so the claim is falsifiable in the
strong sense: a batch size that changed a seeded result would refute the row,
not merely fail to confirm it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and older
    import tomli as tomllib

import pytest
import torch

import flagquantum as fq
import flagquantum.noise as fqn
from flagquantum.noise import depolarizing_channel
from flagquantum.runtime.executors.statevector import (
    BatchedStatevectorTrajectoryResult,
    run_noisy_statevector,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "contracts/cudaq-parity-matrix.toml"

TRAJECTORIES = 37
SEED = 19
SERIAL_BATCH = 1
BATCHED_BATCH = 11

EXECUTOR = "flagquantum/runtime/executors/statevector/noisy.py"
THIS_TEST = "tests/unit/test_parity_trajectory_rows.py"

# The phrasing that was false. It is kept here as the needle the contract must
# not contain again, in any row.
ABSENCE_NEEDLES = (
    "No batched trajectory result",
    "no batched trajectory result",
)


def _rows() -> dict[str, dict[str, Any]]:
    document = tomllib.loads(MATRIX.read_text(encoding="utf-8"))
    return {
        row["id"]: row
        for domain in document["domains"]
        for row in domain["capabilities"]
    }


def _seeded_pair() -> (
    tuple[BatchedStatevectorTrajectoryResult, BatchedStatevectorTrajectoryResult]
):
    circuit = fq.Circuit(2).h(0).cx(0, 1)
    model = fqn.NoiseModel().add("cx", depolarizing_channel(0.2))
    serial = run_noisy_statevector(
        circuit,
        model,
        trajectories=TRAJECTORIES,
        trajectory_batch_size=SERIAL_BATCH,
        seed=SEED,
        retain_trajectories=True,
    )
    batched = run_noisy_statevector(
        circuit,
        model,
        trajectories=TRAJECTORIES,
        trajectory_batch_size=BATCHED_BATCH,
        seed=SEED,
        retain_trajectories=True,
    )
    return serial, batched


def test_the_public_trajectory_runner_returns_a_batched_result() -> None:
    """The runner returns the batched type and records the batch it was given."""

    serial, batched = _seeded_pair()

    assert isinstance(serial, BatchedStatevectorTrajectoryResult)
    assert isinstance(batched, BatchedStatevectorTrajectoryResult)
    assert serial.trajectory_batch_size == SERIAL_BATCH
    assert batched.trajectory_batch_size == BATCHED_BATCH
    assert serial.summary()["backend"] == "batched_statevector_trajectory"
    assert batched.summary()["trajectory_batch_size"] == BATCHED_BATCH
    assert batched.summary()["trajectories"] == TRAJECTORIES
    assert len(batched.trajectory_ids) == TRAJECTORIES


def test_batch_size_does_not_change_a_seeded_result() -> None:
    """Two batch sizes over one seed produce one result, which is the row's claim."""

    serial, batched = _seeded_pair()

    assert serial.retained_states is not None
    assert batched.retained_states is not None
    assert serial.retained_states.shape == batched.retained_states.shape
    assert torch.equal(serial.retained_states, batched.retained_states)
    assert serial.trajectory_seeds == batched.trajectory_seeds
    assert torch.equal(serial.expectation_z, batched.expectation_z)


def test_the_noise_row_records_the_batched_result_it_has() -> None:
    """The row that claimed no batched result exists now records the measurement."""

    row = _rows()["backend_trajectory_noise"]

    assert row["status"] == "supported"
    assert row["priority"] == "now"
    assert row["dependency_class"] == "none"

    evidence = row.get("evidence") or []
    assert EXECUTOR in evidence, f"{EXECUTOR} is missing from {evidence}"
    assert THIS_TEST in evidence, f"{THIS_TEST} is missing from {evidence}"

    text = row.get("reason", "") + " " + row.get("baseline_gap", "")
    for needle in ABSENCE_NEEDLES:
        assert needle not in text, (
            f"backend_trajectory_noise still says {needle!r}, so it denies the "
            "result this module just measured"
        )

    gap = row.get("baseline_gap", "")
    assert "MPS" in gap, (
        "the gap has to name the residue the measurement leaves: batches are "
        "executed for statevector trajectories and not for the MPS ones"
    )


def test_no_row_in_the_matrix_denies_the_batched_trajectory_result() -> None:
    """A false absence claim must not move to a neighbouring row."""

    offenders = [
        row["id"]
        for row in _rows().values()
        for needle in ABSENCE_NEEDLES
        if needle in row.get("reason", "") + " " + row.get("baseline_gap", "")
    ]

    assert not offenders, (
        "these rows deny a batched trajectory result, which this module measures "
        f"through the registered runner: {offenders}"
    )
