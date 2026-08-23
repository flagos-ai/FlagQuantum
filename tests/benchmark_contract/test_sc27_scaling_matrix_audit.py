from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "paper" / "sc27" / "audit_scaling_matrix.py"
SPEC = importlib.util.spec_from_file_location("sc27_scaling_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def hardware(world: int) -> list[dict]:
    return [{
        "rank": rank, "hostname": "node0" if rank < 8 else "node1",
        "gpu_uuid": f"GPU-{rank}", "pci_bus_id": f"0000:{rank:02x}:00.0",
        "device_name": "NVIDIA A800", "total_memory_bytes": 80 << 30,
        "compute_capability": "8.0", "multiprocessor_count": 108,
        "power_limit_watts": "400", "persistence_mode": "Enabled",
        "compute_mode": "Default", "max_sm_clock_mhz": "1410",
        "max_memory_clock_mhz": "1593", "cpu_affinity": [rank],
        "torch_cpu_thread_count": 1, "identity_complete": True,
        "communication_environment": {},
        "topology": {"captured": True, "nvidia_smi_topology_sha256": "e" * 64},
    } for rank in range(world)]


def point(world: int, run: int, *, mode: str) -> dict:
    n_wires = 31 if mode == "strong" else 28 + (world.bit_length() - 1)
    state_semantics = "sharded_across_ranks" if world > 1 else "single_device_fast_path"
    return {
        "schema": "flagquantum.statevector.training_scaling.v1",
        "benchmark": "statevector_differentiable_training_scaling",
        "artifact_class": "measured_production_run",
        "benchmark_evidence_class": "sc27_candidate",
        "non_release_evidence": False,
        "release_gate_allowed": True,
        "scalability_blockers": [],
        "world_size": world,
        "node_count": 2 if world == 16 else 1,
        "rank_placement": hardware(world),
        "source_identity": {
            "commit": "a" * 40,
            "source_dirty": False,
            "workload_sha256": f"{world * 10 + run:064x}",
            "command": "torchrun benchmark.py",
            "container_digest": "sha256:" + "b" * 64,
            "raw_log_sha256": f"{world * 100 + run:064x}",
        },
        "environment": {
            "python": "3.12",
            "torch": "2.13",
            "cuda": "13",
            "device_name": "A800",
            "driver_version": "x",
        },
        "protocol": {
            "gradient_method": "reversible_adjoint",
            "synchronization": "rank_max",
            "warmup": 5,
            "repetitions": 20,
            "independent_run_index": run,
        },
        "workload": {
            "n_wires": n_wires,
            "name": "full_width_linear_hea",
            "layers": 8,
            "gate_count": (2 * n_wires - 1) * 8,
            "parameter_count": n_wires * 8,
            "observable": f"Z({n_wires // 2})",
            "dtype": "complex64",
            "seed": MODULE.RUN_TO_SEED[run],
            "optimizer": "adam",
            "optimizer_ownership": "owner_sharded_across_ranks",
            "learning_rate": 0.01,
        },
        "ownership": {
            "primal_state": state_semantics,
            "adjoint_state": state_semantics,
            "parameter_gradient": "replicated_across_ranks",
            "optimizer_state": "owner_sharded_across_ranks",
            "full_quantum_state_materialized": False,
            "parameter_bytes_per_rank": [100] * world,
            "gradient_bytes_per_rank": [100] * world,
            "optimizer_bytes_per_rank": [100] * world,
        },
        "optimizer": {
            "name": "adam",
            "executed": True,
            "step_seconds": 0.01,
            "parameters_equal_across_ranks": True,
        },
        "value_and_grad": {"samples_seconds": [1.0] * 20},
        "training_step": {"samples_seconds": [1.01] * 20},
        "memory": {"peak_bytes_by_rank": [1000] * world},
        "communication": {
            "bytes_by_rank": [0 if world == 1 else 100] * world,
            "profile_by_rank": [{
                "included_in_timing_samples": False,
                "communication_kernels_observed": world > 1,
                "communication_device_seconds_union": 0.0 if world == 1 else 0.1,
                "overlap_fraction_of_communication": 0.0 if world == 1 else 0.5,
                "gradient_all_reduce_kernel_count": 0 if world == 1 else 1,
                "gradient_all_reduce_overlap_fraction": 0.0 if world == 1 else 0.5,
            }] * world,
            "profiling_step_excluded_from_timing_samples": True,
        },
        "correctness": {
            "passed": True,
            "reference_artifact_sha256": f"{n_wires * 100 + MODULE.RUN_TO_SEED[run]:064x}",
        },
        "fallback_events": [],
        "ablation": {"id": "full"},
    }


def matrix(mode: str) -> list[dict]:
    return [
        point(world, run, mode=mode)
        for world in MODULE.WORLD_SIZES
        for run in MODULE.RUN_TO_SEED
    ]


def test_accepts_complete_strong_scaling_matrix() -> None:
    result = MODULE.audit(matrix("strong"), mode="strong")
    assert result["paper_ready_scaling_matrix"] is True


def test_accepts_frozen_weak_scaling_rule() -> None:
    result = MODULE.audit(matrix("weak"), mode="weak")
    assert result["paper_ready_scaling_matrix"] is True


def test_rejects_missing_point_and_workload_drift() -> None:
    payloads = matrix("strong")
    payloads.pop()
    result = MODULE.audit(payloads, mode="strong")
    assert result["paper_ready_scaling_matrix"] is False
    assert any(item.startswith("matrix:missing_point") for item in result["blockers"])

    payloads = matrix("strong")
    payloads[-1]["workload"]["n_wires"] = 30
    result = MODULE.audit(payloads, mode="strong")
    assert "workload:world=16:run=3:n_wires_not_frozen_value" in result["blockers"]


def test_rejects_reference_drift_for_same_workload_and_seed() -> None:
    payloads = matrix("strong")
    payloads[-1]["correctness"]["reference_artifact_sha256"] = "f" * 64
    result = MODULE.audit(payloads, mode="strong")
    assert "pairing:world=16:run=3:correctness_reference_mismatch" in result["blockers"]
