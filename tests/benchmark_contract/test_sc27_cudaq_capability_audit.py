from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "paper" / "sc27" / "audit_cudaq_capability.py"
SPEC = importlib.util.spec_from_file_location("sc27_cudaq_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def hardware() -> list[dict]:
    return [
        {
            "rank": 0,
            "hostname": "node0",
            "gpu_uuid": "GPU-0",
            "pci_bus_id": "0000:00:00.0",
            "device_name": "NVIDIA A800",
            "total_memory_bytes": 80 << 30,
            "compute_capability": "8.0",
            "multiprocessor_count": 108,
            "power_limit_watts": "400",
            "persistence_mode": "Enabled",
            "compute_mode": "Default",
            "max_sm_clock_mhz": "1410",
            "max_memory_clock_mhz": "1593",
            "cpu_affinity": [0],
            "torch_cpu_thread_count": 1,
            "identity_complete": True,
            "communication_environment": {},
            "topology": {"captured": True, "nvidia_smi_topology_sha256": "e" * 64},
        }
    ]


def valid_payload() -> dict:
    return {
        "schema": "flagquantum.external.cudaq_gradient_capability.v1",
        "comparison_class": "explicitly_unsupported_or_requires_matched_run",
        "world_size": 1,
        "rank_placement": hardware(),
        "source_identity": {
            "commit": "a" * 40,
            "source_dirty": False,
            "workload_sha256": "b" * 64,
            "command": "python probe.py",
            "container_digest": "sha256:" + "c" * 64,
            "raw_log_sha256": "d" * 64,
        },
        "environment": {"python": "3.12", "cudaq": "x", "driver_version": "x"},
        "workload": {"requested_operation": "value_and_full_reverse_gradient"},
        "protocol": {"independent_run_index": 1},
        "capability": {
            "exported_gradient_symbols": ["ParameterShift"],
            "matched_reverse_mode_symbols": [],
            "matched_value_and_gradient_available": False,
        },
        "official_documentation": {"url": "https://example", "accessed": "2026-08-11"},
        "fallback_events": [],
    }


def test_accepts_version_bound_unsupported_result() -> None:
    result = MODULE.audit(valid_payload())
    assert result["paper_ready_unsupported_result"] is True
    assert result["headline_speedup_allowed"] is False


def test_requires_matched_run_when_reverse_mode_appears() -> None:
    payload = valid_payload()
    payload["capability"]["matched_reverse_mode_symbols"] = ["AdjointGradient"]
    payload["capability"]["matched_value_and_gradient_available"] = True
    result = MODULE.audit(payload)
    assert "capability:matched_method_exists_run_it_instead" in result["blockers"]
