from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "paper" / "sc27" / "audit_forward_payload.py"
SPEC = importlib.util.spec_from_file_location("sc27_forward_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def hardware() -> list[dict]:
    return [{
        "rank": 0, "hostname": "node0", "gpu_uuid": "GPU-0",
        "pci_bus_id": "0000:00:00.0", "device_name": "NVIDIA A800",
        "total_memory_bytes": 80 << 30, "compute_capability": "8.0",
        "multiprocessor_count": 108, "power_limit_watts": "400",
        "persistence_mode": "Enabled", "compute_mode": "Default",
        "max_sm_clock_mhz": "1410", "max_memory_clock_mhz": "1593",
        "cpu_affinity": [0], "torch_cpu_thread_count": 1,
        "identity_complete": True, "communication_environment": {},
        "topology": {"captured": True, "nvidia_smi_topology_sha256": "e" * 64},
    }]


def valid_payload() -> dict:
    return {
        "schema": "flagquantum.custatevec_statevector_compare.v1",
        "benchmark": "flagquantum_native_vs_custatevec_forward",
        "comparison_class": "forward_subsystem_only",
        "world_size": 1,
        "rank_placement": hardware(),
        "source_identity": {
            "commit": "a" * 40,
            "source_dirty": False,
            "workload_sha256": "b" * 64,
            "command": "python benchmark.py",
            "container_digest": "sha256:" + "c" * 64,
            "raw_log_sha256": "d" * 64,
        },
        "environment": {
            "python": "3.12",
            "cuda": "13",
            "device_name": "A800",
            "cuquantum": "x",
            "cupy": "x",
            "driver_version": "x",
        },
        "workload": {
            "name": "full_width_linear_hea_forward_state",
            "n_wires": 31,
            "layers": 8,
            "parameter_count": 248,
            "dtype": "complex64",
            "seed": 41,
        },
        "protocol": {
            "independent_run_index": 1,
            "warmup": 5,
            "repetitions": 20,
            "execution_order": "counterbalanced_by_sample",
        },
        "flagquantum": {"samples_seconds": [1.0] * 20},
        "custatevec": {"samples_seconds": [0.5] * 20},
        "correctness": {
            "passed": True,
            "max_abs_error": 1e-6,
            "absolute_tolerance": 3e-5,
        },
        "stability": {"comparison_claim_allowed": True},
    }


def test_accepts_counterbalanced_forward_only_run() -> None:
    result = MODULE.audit(valid_payload())
    assert result["paper_ready_forward_run"] is True
    assert result["headline_value_and_gradient_speedup_allowed"] is False


def test_rejects_gradient_claim_or_sequential_order() -> None:
    payload = valid_payload()
    payload["comparison_class"] = "value_and_gradient"
    payload["protocol"]["execution_order"] = "flagquantum_then_custatevec"
    result = MODULE.audit(payload)
    assert "semantics:not_forward_subsystem_only" in result["blockers"]
    assert "protocol:execution_order_not_counterbalanced" in result["blockers"]
