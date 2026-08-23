from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "paper" / "sc27" / "select_scientific_pilot.py"
SPEC = importlib.util.spec_from_file_location("sc27_pilot_selection", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def payload(learning_rate: float, seed: int, *, success: bool, seconds: float) -> dict:
    return {
        "n_sites": 20,
        "ansatz_depth": 8,
        "requested_max_bond": 256,
        "cutoff": 0.0,
        "anisotropy": 1.0,
        "field": 0.0,
        "initial_state": "dimer_singlet",
        "parameterization": "bond_resolved",
        "parameter_scale": 0.02,
        "optimizer": "adam",
        "precision": "float32",
        "steps": 100,
        "world_size": 1,
        "learning_rate": learning_rate,
        "seed": seed,
        "completed": True,
        "fallback_events": [],
        "source_identity": {
            "commit": "a" * 40,
            "container_digest": "sha256:" + "b" * 64,
            "source_dirty": False,
            "raw_log_sha256": f"{seed:064x}",
        },
        "protocol": {"independent_run_index": seed - 40},
        "reference": {"method": "sparse_exact_diagonalization", "energy": -10.0},
        "convergence_tolerance": 1e-3,
        "optimization_trace": [{"optimizer_step": step} for step in range(100)],
        "optimizer_step_seconds": [1.0] * 100,
        "target_reached": success,
        "time_to_target_seconds": seconds if success else None,
    }


def grid() -> list[dict]:
    success = {
        0.005: {41, 42},
        0.01: {41, 42, 43},
        0.02: {41, 42},
    }
    return [
        payload(
            learning_rate,
            seed,
            success=seed in success[learning_rate],
            seconds=10.0 + seed,
        )
        for learning_rate in MODULE.LEARNING_RATES
        for seed in MODULE.SEEDS
    ]


def test_selects_success_count_before_time_to_target() -> None:
    result = MODULE.select(grid())
    assert result["primary_experiment_unblocked"] is True
    assert result["selected"]["learning_rate"] == 0.01
    assert result["selected"]["maximum_optimizer_steps"] == 100


def test_blocks_primary_when_no_candidate_succeeds_twice() -> None:
    payloads = grid()
    for item in payloads:
        item["target_reached"] = item["seed"] == 41
        item["time_to_target_seconds"] = 10.0 if item["seed"] == 41 else None
    result = MODULE.select(payloads)
    assert result["primary_experiment_unblocked"] is False
    assert result["selected"] is None
