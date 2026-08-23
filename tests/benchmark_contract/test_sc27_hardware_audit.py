from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[2] / "paper" / "sc27" / "audit_hardware.py"
SPEC = importlib.util.spec_from_file_location("sc27_hardware_audit", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def placement(rank: int, *, hostname: str = "node0") -> dict:
    return {
        "rank": rank,
        "hostname": hostname,
        "gpu_uuid": f"GPU-{hostname}-{rank}",
        "pci_bus_id": f"0000:{rank:02x}:00.0",
        "device_name": "NVIDIA A800-SXM4-80GB",
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
        "communication_environment": {"NCCL_ALGO": None},
        "topology": {
            "captured": True,
            "nvidia_smi_topology_sha256": ("a" if hostname == "node0" else "b") * 64,
        },
    }


def payload() -> dict:
    return {
        "world_size": 2,
        "rank_placement": [placement(0), placement(1)],
        "environment": {"driver_version": "600.1"},
    }


def test_accepts_complete_rank_identity_and_topology() -> None:
    result = MODULE.audit(payload())
    assert result["hardware_identity_complete"] is True


def test_rejects_missing_identity_duplicate_uuid_and_topology_drift() -> None:
    value = payload()
    value["rank_placement"][1]["gpu_uuid"] = value["rank_placement"][0]["gpu_uuid"]
    value["rank_placement"][1]["pci_bus_id"] = None
    value["rank_placement"][1]["topology"]["nvidia_smi_topology_sha256"] = "c" * 64
    result = MODULE.audit(value)
    assert result["hardware_identity_complete"] is False
    assert "placement:gpu_uuid_not_unique" in result["blockers"]
    assert "rank=1:pci_bus_id_missing" in result["blockers"]
    assert "rank=1:pci_bus_id_invalid" in result["blockers"]
    assert "host=node0:topology_inconsistent" in result["blockers"]


def test_rejects_non_a800_or_missing_driver() -> None:
    value = payload()
    value["rank_placement"][0]["device_name"] = "NVIDIA H100"
    value["environment"]["driver_version"] = None
    result = MODULE.audit(value)
    assert "rank=0:unexpected_device" in result["blockers"]
    assert "environment:driver_version_missing" in result["blockers"]


def test_rejects_truncated_pci_bus_id() -> None:
    value = payload()
    value["rank_placement"][0]["pci_bus_id"] = "16"
    result = MODULE.audit(value)
    assert "rank=0:pci_bus_id_invalid" in result["blockers"]
