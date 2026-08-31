from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).parents[2]
MODULE = load(
    ROOT / "paper" / "sc27" / "audit_scientific_matrix.py",
    "sc27_science_matrix",
)
FIXTURE = load(
    ROOT / "tests" / "benchmark_contract" / "test_sc27_scientific_payload_audit.py",
    "sc27_science_fixture",
)


def selection() -> dict:
    return {
        "primary_experiment_unblocked": True,
        "selected": {"learning_rate": 0.01, "maximum_optimizer_steps": 100},
    }


def point(world: int, run: int, *, success: bool = True) -> dict:
    payload = copy.deepcopy(FIXTURE.valid_payload())
    payload["world_size"] = world
    payload["node_count"] = 2 if world == 16 else 1
    payload["seed"] = MODULE.RUN_TO_SEED[run]
    payload["learning_rate"] = 0.01
    payload["steps"] = 100
    payload["protocol"]["independent_run_index"] = run
    payload["rank_placement"] = FIXTURE.hardware(world)
    payload["rank_ownership"] = [{"rank": rank} for rank in range(world)]
    payload["local_memory_bytes_by_rank"] = [10] * world
    payload["communication_bytes_by_rank"] = [20] * world
    payload["ownership"]["optimizer_bytes_per_rank"] = [100] * world
    payload["discarded_weight_by_step"] = [0.0] * 100
    payload["optimizer_step_seconds"] = [1.0] * 100
    payload["source_identity"]["raw_log_sha256"] = f"{world * 10 + run:064x}"
    if world == 1:
        payload["ownership"].update(
            {
                "primal_state": "single_device",
                "adjoint_state": "single_device",
                "parameter_gradient": "single_device",
                "optimizer_state": "single_device",
            }
        )
    payload["target_reached"] = success
    payload["convergence_certified"] = success
    payload["target_reached_step"] = 2 if success else None
    payload["time_to_target_seconds"] = 2.0 if success else None
    payload["relative_energy_error"] = 5e-4 if success else 2e-3
    return payload


def matrix() -> list[dict]:
    return [
        point(world, run) for world in MODULE.WORLD_SIZES for run in MODULE.RUN_TO_SEED
    ]


def test_accepts_pilot_bound_primary_matrix() -> None:
    result = MODULE.audit(matrix(), selection())
    assert result["paper_ready_scientific_matrix"] is True
    assert result["success_count_by_world_size"] == {
        str(world): 3 for world in MODULE.WORLD_SIZES
    }


def test_rejects_learning_rate_drift_or_low_success_rate() -> None:
    payloads = matrix()
    payloads[0]["learning_rate"] = 0.02
    for payload in payloads:
        if payload["world_size"] == 16 and payload["seed"] != 41:
            payload["target_reached"] = False
            payload["convergence_certified"] = False
            payload["target_reached_step"] = None
            payload["time_to_target_seconds"] = None
            payload["relative_energy_error"] = 2e-3
    result = MODULE.audit(payloads, selection())
    assert "pilot:world=1:run=1:learning_rate_drift" in result["blockers"]
    assert "science:world=16:success_count_1_below_required_2" in result["blockers"]
