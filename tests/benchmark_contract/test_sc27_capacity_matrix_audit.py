from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "paper" / "sc27" / "audit_capacity_matrix.py"
SPEC = importlib.util.spec_from_file_location("sc27_capacity_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

WORKLOAD = {
    "n_wires": 35,
    "name": "full_width_linear_hea",
    "layers": 1,
    "gate_count": 69,
    "active_wire_count": 35,
    "parameter_count": 35,
    "entanglement": "directed_cnot_linear",
    "observable": "Z(17)",
    "dtype": "complex64",
    "seed": 20260722,
    "optimizer": "adam",
    "learning_rate": 0.01,
    "semantic_sha256": "c" * 64,
}


def hardware(world: int) -> list[dict]:
    return [
        {
            "rank": rank,
            "hostname": "node0" if rank < 8 else "node1",
            "gpu_uuid": f"GPU-{rank}",
            "pci_bus_id": f"0000:{rank:02x}:00.0",
            "device_name": "NVIDIA A800",
            "total_memory_bytes": 80 << 30,
            "compute_capability": "8.0",
            "multiprocessor_count": 108,
            "power_limit_watts": "400",
            "persistence_mode": "Enabled",
            "compute_mode": "Default",
            "max_sm_clock_mhz": "1410",
            "max_memory_clock_mhz": "1593",
            "cpu_affinity": [rank],
            "torch_cpu_thread_count": 1,
            "identity_complete": True,
            "communication_environment": {},
            "topology": {"captured": True, "nvidia_smi_topology_sha256": "e" * 64},
        }
        for rank in range(world)
    ]


def identity(run: int) -> dict:
    return {
        "commit": "a" * 40,
        "container_digest": "sha256:" + "b" * 64,
        "source_dirty": False,
        "raw_log_sha256": f"{run:064x}",
    }


def oom(run: int) -> dict:
    return {
        "world_size": 1,
        "node_count": 1,
        "rank_placement": hardware(1),
        "environment": {"driver_version": "test"},
        "source_identity": identity(run),
        "protocol": {"independent_run_index": run},
        "workload": copy.deepcopy(WORKLOAD),
        "observed_outcome": "cuda_oom",
        "single_device_oom_observed": True,
        "expectation_met": True,
        "oom_type": "cuda_out_of_memory",
        "fallback_events": [],
    }


def completion(run: int) -> dict:
    return {
        "world_size": 16,
        "node_count": 2,
        "rank_placement": hardware(16),
        "environment": {"driver_version": "test"},
        "source_identity": identity(run + 10),
        "protocol": {
            "independent_run_index": run,
            "capacity_evidence": True,
            "warmup": 0,
            "repetitions": 1,
        },
        "workload": copy.deepcopy(WORKLOAD),
        "ownership": {
            "primal_state": "sharded_across_ranks",
            "adjoint_state": "sharded_across_ranks",
            "optimizer_state": "owner_sharded_across_ranks",
            "full_quantum_state_materialized": False,
        },
        "optimizer": {"executed": True, "parameters_equal_across_ranks": True},
        "correctness": {
            "passed": True,
            "initial_value": 0.25,
            "initial_gradients": [0.01] * 35,
        },
        "ablation": {"id": "full"},
        "fallback_events": [],
    }


def test_accepts_three_matched_oom_and_completion_runs() -> None:
    result = MODULE.audit(
        [oom(run) for run in (1, 2, 3)],
        [completion(run) for run in (1, 2, 3)],
    )
    assert result["paper_ready_capacity_expansion"] is True


def test_rejects_workload_drift_or_unsealed_oom() -> None:
    ooms = [oom(run) for run in (1, 2, 3)]
    completions = [completion(run) for run in (1, 2, 3)]
    ooms[0]["source_identity"]["raw_log_sha256"] = None
    completions[-1]["workload"]["parameter_count"] = 34
    result = MODULE.audit(ooms, completions)
    assert result["paper_ready_capacity_expansion"] is False
    assert any("raw_log_not_sealed" in item for item in result["blockers"])
    assert any(
        "parameter_count_not_frozen_value" in item for item in result["blockers"]
    )
