from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "paper" / "sc27" / "audit_payload.py"
SPEC = importlib.util.spec_from_file_location("sc27_audit_payload", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def hardware(world: int) -> list[dict]:
    return [
        {
            "rank": rank,
            "hostname": "node0",
            "gpu_uuid": f"GPU-{rank}",
            "pci_bus_id": f"0000:{rank:02x}:00.0",
            "device_name": "NVIDIA A800",
            "total_memory_bytes": 80 << 30,
            "compute_capability": "8.0",
            "multiprocessor_count": 108,
            "power_limit_watts": "400.00",
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


def valid_payload() -> dict:
    world = 2
    return {
        "schema": "flagquantum.statevector.sc27_run.v1",
        "benchmark": "flagquantum_adjoint_training_step",
        "artifact_class": "measured_production_run",
        "benchmark_evidence_class": "sc27_candidate",
        "non_release_evidence": False,
        "release_gate_allowed": True,
        "scalability_blockers": [],
        "world_size": world,
        "source_identity": {
            "commit": "a" * 40,
            "source_dirty": False,
            "workload_sha256": "b" * 64,
            "command": "torchrun benchmark.py",
            "container_digest": "sha256:" + "c" * 64,
            "raw_log_sha256": "d" * 64,
        },
        "environment": {
            "python": "3.12",
            "torch": "2.13",
            "cuda": "13.0",
            "device_name": "NVIDIA A800-SXM4-80GB",
            "driver_version": "test",
        },
        "protocol": {
            "gradient_method": "reversible_adjoint",
            "synchronization": "rank_max",
            "warmup": 5,
            "repetitions": 20,
            "independent_run_index": 1,
        },
        "ownership": {
            "primal_state": "sharded_across_ranks",
            "adjoint_state": "sharded_across_ranks",
            "parameter_gradient": "replicated_across_ranks",
            "optimizer_state": "replicated_across_ranks",
            "full_quantum_state_materialized": False,
            "parameter_bytes_per_rank": 992,
            "gradient_bytes_per_rank": 992,
            "optimizer_bytes_per_rank": 1984,
        },
        "optimizer": {
            "name": "adam",
            "executed": True,
            "step_seconds": 0.001,
            "parameters_equal_across_ranks": True,
        },
        "value_and_grad": {"samples_seconds": [1.0] * 20},
        "training_step": {"samples_seconds": [1.001] * 20},
        "memory": {"peak_bytes_by_rank": [10, 10]},
        "communication": {
            "bytes_by_rank": [20, 20],
            "profile_by_rank": [
                {
                    "included_in_timing_samples": False,
                    "communication_kernels_observed": True,
                    "communication_device_seconds_union": 0.1,
                    "overlap_fraction_of_communication": 0.5,
                    "gradient_all_reduce_kernel_count": 1,
                    "gradient_all_reduce_overlap_fraction": 0.5,
                }
            ]
            * world,
            "profiling_step_excluded_from_timing_samples": True,
        },
        "rank_placement": hardware(world),
        "fallback_events": [],
        "correctness": {"passed": True},
    }


def test_accepts_complete_independent_run() -> None:
    result = MODULE.audit(valid_payload())
    assert result["paper_ready_independent_run"] is True
    assert result["blockers"] == []


def test_rejects_forward_only_or_unmeasured_optimizer_payload() -> None:
    payload = valid_payload()
    payload["ownership"]["adjoint_state"] = "unmeasured"
    payload["optimizer"]["executed"] = False
    payload["fallback_events"] = None
    result = MODULE.audit(payload)
    assert result["paper_ready_independent_run"] is False
    assert "ownership:adjoint_state_not_sharded" in result["blockers"]
    assert "optimizer:not_executed" in result["blockers"]
    assert "execution:fallback_events_nonempty_or_missing" in result["blockers"]


def test_rejects_short_protocol_and_rank_shape_mismatch() -> None:
    payload = valid_payload()
    payload["protocol"].update(
        {"warmup": 2, "repetitions": 5, "independent_run_index": 0}
    )
    payload["memory"]["peak_bytes_by_rank"] = [10]
    result = MODULE.audit(payload)
    assert result["paper_ready_independent_run"] is False
    assert "protocol:warmup_below_5" in result["blockers"]
    assert "protocol:repetitions_below_20" in result["blockers"]
    assert "shape:memory.peak_bytes_by_rank_length_not_world_size" in result["blockers"]


def test_accepts_required_single_gpu_scaling_baseline() -> None:
    payload = valid_payload()
    payload["world_size"] = 1
    payload["rank_placement"] = hardware(1)
    payload["memory"]["peak_bytes_by_rank"] = [10]
    payload["communication"]["bytes_by_rank"] = [0]
    payload["communication"]["profile_by_rank"] = [
        {
            "included_in_timing_samples": False,
            "communication_kernels_observed": False,
            "communication_device_seconds_union": 0.0,
            "overlap_fraction_of_communication": 0.0,
            "gradient_all_reduce_kernel_count": 0,
            "gradient_all_reduce_overlap_fraction": 0.0,
        }
    ]
    payload["ownership"].update(
        {
            "primal_state": "single_device_fast_path",
            "adjoint_state": "single_device_fast_path",
        }
    )
    result = MODULE.audit(payload)
    assert result["paper_ready_independent_run"] is True
