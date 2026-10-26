"""Scientific oracles and full optimizer-recovery scenarios for CPU training."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

np = pytest.importorskip("numpy")

# Producers import the optional numerical oracle, so guard it before these imports.
from benchmarks.training_comparison import (  # noqa: E402
    check_parity,
    ground_energy,
    train_case,
)
from benchmarks.training_engines import Engine, build_expectation  # noqa: E402
from benchmarks.training_workloads import (  # noqa: E402
    Task,
    dataset,
    dense_value,
    initial_parameters,
)

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.integration]


@pytest.fixture(autouse=True)
def double_precision() -> Iterator[None]:
    previous_dtype = torch.get_default_dtype()
    previous_threads = torch.get_num_threads()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    yield
    torch.set_default_dtype(previous_dtype)
    torch.set_num_threads(previous_threads)


def test_dense_heisenberg_reference_has_known_dimer_energy() -> None:
    parameters = np.zeros(26)
    assert dense_value("heisenberg", parameters, np.zeros(2)) == pytest.approx(-6)
    assert ground_energy() == pytest.approx(-3 - 2 * np.sqrt(3))


def test_dataset_has_disjoint_fixed_splits_and_both_labels() -> None:
    train, validation, test = dataset()
    assert (len(train.labels), len(validation.labels), len(test.labels)) == (64, 32, 32)
    all_rows = [
        tuple(row)
        for split in (train, validation, test)
        for row in split.features.tolist()
    ]
    assert len(set(all_rows)) == 128
    for split in (train, validation, test):
        assert set(split.labels.tolist()) == {0, 1}


@pytest.mark.parametrize("task", ["classifier", "heisenberg"])
@pytest.mark.parametrize(
    "engine",
    [
        "flagquantum",
        "flagquantum_torch_module",
        "flagquantum_jax_module",
        "flagquantum_jax_kernel",
        "pennylane_backprop",
        "pennylane_adjoint",
        "tensorcircuit_torch",
        "tensorcircuit_jax",
    ],
)
def test_simulator_gradients_match_independent_dense_finite_difference(
    task: Task, engine: Engine
) -> None:
    package = (
        "jax"
        if engine.startswith("flagquantum_jax")
        else (
            None
            if engine.startswith("flagquantum")
            else "tensorcircuit" if engine.startswith("tensorcircuit") else "pennylane"
        )
    )
    if package is not None and importlib.util.find_spec(package) is None:
        pytest.skip(f"optional comparison dependency {package} is absent")
    if engine == "tensorcircuit_jax" and importlib.util.find_spec("jax") is None:
        pytest.skip("optional JAX comparison dependency is absent")
    parameters = initial_parameters(task, 29)
    parity = check_parity(
        task, parameters, dataset()[0], build_expectation(engine, task)
    )
    assert parity.passed


@pytest.mark.parametrize("task", ["classifier", "heisenberg"])
def test_checkpoint_restores_complete_adam_trajectory(
    task: Task, tmp_path: Path
) -> None:
    result = train_case(task, "flagquantum", 11, smoke=True, directory=tmp_path)
    assert result["completed_steps"] == 3
    assert result["checkpoint_replay_max_parameter_error"] == 0
    assert result["checkpoint_replay_max_gradient_error"] == 0
    assert result["checkpoint_replay_max_optimizer_error"] == 0
    assert result["scalability_claim_allowed"] is False
    assert (
        result["assessment"]["test_accuracy"] is None
        if task == "heisenberg"
        else 0 <= result["assessment"]["test_accuracy"] <= 1
    )


def test_parity_detects_wrong_quantum_expectation() -> None:
    parameters = initial_parameters("classifier", 11)

    def wrong(values: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return values[0] + features[0]

    assert not check_parity("classifier", parameters, dataset()[0], wrong).passed
