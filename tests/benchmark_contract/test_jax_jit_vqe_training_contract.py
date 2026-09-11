"""Contract checks for the end-to-end JAX JIT VQE training case."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[2] / "benchmarks" / "jax_jit_vqe_training.py"
    spec = importlib.util.spec_from_file_location("jax_jit_vqe_training", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_crossover_requires_jax_cumulative_time_to_overtake() -> None:
    module = _module()
    torch_run = {"cumulative_wall_seconds": [1.0, 2.0, 3.0, 4.0]}
    jax_run = {"cumulative_wall_seconds": [4.0, 3.0, 2.5, 2.7]}
    assert module._first_crossover(torch_run, jax_run) == 3


def test_no_crossover_is_reported_as_none() -> None:
    module = _module()
    torch_run = {"cumulative_wall_seconds": [1.0, 2.0, 3.0]}
    jax_run = {"cumulative_wall_seconds": [4.0, 4.1, 4.2]}
    assert module._first_crossover(torch_run, jax_run) is None
