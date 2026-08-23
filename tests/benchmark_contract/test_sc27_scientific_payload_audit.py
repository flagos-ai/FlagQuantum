from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "paper" / "sc27" / "audit_scientific_payload.py"
SPEC = importlib.util.spec_from_file_location("sc27_science_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def hardware(world: int) -> list[dict]:
    return [{
        "rank": rank, "hostname": "node0", "gpu_uuid": f"GPU-{rank}",
        "pci_bus_id": f"0000:{rank:02x}:00.0", "device_name": "NVIDIA A800",
        "total_memory_bytes": 80 << 30, "compute_capability": "8.0",
        "multiprocessor_count": 108, "power_limit_watts": "400",
        "persistence_mode": "Enabled", "compute_mode": "Default",
        "max_sm_clock_mhz": "1410", "max_memory_clock_mhz": "1593",
        "cpu_affinity": [rank], "torch_cpu_thread_count": 1,
        "identity_complete": True, "communication_environment": {},
        "topology": {"captured": True, "nvidia_smi_topology_sha256": "f" * 64},
    } for rank in range(world)]


def valid_payload() -> dict:
    world = 2
    return {
        "schema": "flagquantum.distributed_mps_heisenberg_vqe.v1",
        "workload": "open_boundary_antiferromagnetic_xxz_heisenberg_vqe",
        "n_sites": 256,
        "ansatz_depth": 8,
        "requested_max_bond": 256,
        "anisotropy": 1.0,
        "field": 0.0,
        "initial_state": "dimer_singlet",
        "parameterization": "bond_resolved",
        "optimizer": "adam",
        "precision": "float32",
        "seed": 41,
        "world_size": world,
        "steps": 2,
        "completed": True,
        "source_identity": {
            "commit": "a" * 40,
            "source_dirty": False,
            "workload_sha256": "b" * 64,
            "command": "torchrun benchmark.py",
            "container_digest": "sha256:" + "c" * 64,
            "raw_log_sha256": "d" * 64,
        },
        "protocol": {
            "independent_run_index": 1,
            "target_relative_energy_error": 1e-3,
        },
        "environment": {
            "python": "3.12",
            "torch": "2.13",
            "cuda": "13.0",
            "device_name": "A800",
            "driver_version": "test",
        },
        "reference": {
            "method": "independent_dmrg",
            "energy": -100.0,
            "artifact_sha256": "e" * 64,
            "independent_of_flagquantum": True,
        },
        "best_variational_energy": -99.95,
        "relative_energy_error": 5e-4,
        "energy_error_per_site": 0.05 / 256,
        "convergence_certified": True,
        "target_reached": True,
        "target_reached_step": 2,
        "time_to_target_seconds": 2.0,
        "optimizer_step_seconds": [1.0, 1.0],
        "rank_placement": hardware(world),
        "rank_ownership": [{"rank": 0}, {"rank": 1}],
        "local_memory_bytes_by_rank": [10, 10],
        "communication_bytes_by_rank": [20, 20],
        "discarded_weight_by_step": [0.0, 0.0],
        "ownership": {
            "primal_state": "sharded_across_ranks",
            "adjoint_state": "sharded_across_ranks",
            "parameter_gradient": "owner_sharded_across_ranks",
            "optimizer_state": "owner_sharded_across_ranks",
            "full_mps_materialized": False,
            "optimizer_bytes_per_rank": [100, 100],
        },
        "fallback_events": [],
    }


def test_accepts_frozen_primary_run() -> None:
    result = MODULE.audit(valid_payload())
    assert result["paper_ready_independent_run"] is True
    assert result["target_reached"] is True


def test_rejects_self_reference_and_workload_drift() -> None:
    payload = valid_payload()
    payload["reference"]["method"] = "flagquantum"
    payload["n_sites"] = 128
    result = MODULE.audit(payload)
    assert result["paper_ready_independent_run"] is False
    assert "reference:not_independent_dmrg" in result["blockers"]
    assert "workload:n_sites_not_frozen_value" in result["blockers"]


def test_failed_target_is_retained_but_not_an_artifact_blocker() -> None:
    payload = valid_payload()
    payload["relative_energy_error"] = 2e-3
    payload["convergence_certified"] = False
    payload["target_reached"] = False
    payload["target_reached_step"] = None
    payload["time_to_target_seconds"] = None
    result = MODULE.audit(payload)
    assert result["paper_ready_independent_run"] is True
    assert result["target_reached"] is False
