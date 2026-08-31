from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "paper" / "sc27" / "audit_external_payload.py"
SPEC = importlib.util.spec_from_file_location("sc27_external_audit", MODULE_PATH)
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
            "power_limit_watts": "400",
            "persistence_mode": "Enabled",
            "compute_mode": "Default",
            "max_sm_clock_mhz": "1410",
            "max_memory_clock_mhz": "1593",
            "cpu_affinity": [rank],
            "torch_cpu_thread_count": 1,
            "identity_complete": True,
            "communication_environment": {},
            "topology": {"captured": True, "nvidia_smi_topology_sha256": "f" * 64},
        }
        for rank in range(world)
    ]


def valid_payload() -> dict:
    return {
        "schema": "external.v1",
        "benchmark": "pennylane_adjoint_training",
        "world_size": 2,
        "rank_placement": hardware(2),
        "source_identity": {
            "commit": "a" * 40,
            "source_dirty": False,
            "workload_sha256": "b" * 64,
            "command": "mpirun benchmark.py",
            "container_digest": "sha256:" + "c" * 64,
            "raw_log_sha256": "d" * 64,
        },
        "environment": {
            "python": "3.12",
            "torch": "2.13",
            "cuda": "13.0",
            "device_name": "A800",
            "driver_version": "test",
        },
        "workload": {
            "name": "full_width_linear_hea",
            "n_wires": 31,
            "layers": 8,
            "parameter_count": 248,
            "observable": "Z(15)",
            "dtype": "complex64",
            "seed": 41,
        },
        "protocol": {
            "interface": "torch",
            "gradient_method": "adjoint",
            "warmup": 5,
            "repetitions": 20,
            "independent_run_index": 1,
        },
        "value_and_grad": {"samples_seconds": [1.0] * 20},
        "training_step": {"samples_seconds": [1.001] * 20},
        "memory": {"peak_bytes_by_rank": [10, 10]},
        "communication": {
            "bytes_by_rank": None,
            "measurement": "not exposed by external API",
        },
        "optimizer": {
            "name": "adam",
            "executed": True,
            "step_seconds": 0.001,
            "parameters_equal_across_ranks": True,
        },
        "ownership": {
            "parameter_bytes_per_rank": [992, 992],
            "gradient_bytes_per_rank": [992, 992],
            "optimizer_bytes_per_rank": [1984, 1984],
        },
        "correctness": {
            "values_and_gradients_finite": True,
            "reference_artifact_sha256": "e" * 64,
            "value_absolute_error": 1e-7,
            "value_relative_error": 1e-7,
            "gradient_max_absolute_error": 1e-6,
            "gradient_relative_l2_error": 1e-6,
            "parameter_count_matches": True,
            "passed": True,
        },
        "fallback_events": [],
    }


def test_accepts_fair_external_run_with_declared_unobservable_communication() -> None:
    result = MODULE.audit(valid_payload())
    assert result["paper_ready_external_run"] is True


def test_rejects_short_or_non_adjoint_external_run() -> None:
    payload = valid_payload()
    payload["protocol"].update({"warmup": 2, "gradient_method": "parameter-shift"})
    result = MODULE.audit(payload)
    assert result["paper_ready_external_run"] is False
    assert "protocol:warmup_below_5" in result["blockers"]
    assert "semantics:gradient_method_not_matched_reverse_mode" in result["blockers"]


def test_requires_reason_when_communication_is_not_measured() -> None:
    payload = valid_payload()
    payload["communication"] = {"bytes_by_rank": None}
    result = MODULE.audit(payload)
    assert "communication:unmeasured_without_reason" in result["blockers"]
