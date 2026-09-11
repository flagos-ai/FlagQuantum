"""Contract tests for the two-node CUDA statevector probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "tools" / "probe_cuda_multinode_statevector.py"
_A800_ARTIFACT = (
    _ROOT / "artifacts" / "cuda_multinode_statevector_a800_jp171_jp172_20260907.json"
)
_SPEC = importlib.util.spec_from_file_location("cuda_multinode_probe", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_network_observation_requires_the_declared_socket_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "nccl.log"
    log.write_text("NCCL INFO NET/Socket : Using [0]ens22f0:192.0.2.3\n")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "ens22f0")
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")

    observation = _MODULE._network_observation(log)

    assert observation["route"] == "socket"
    assert observation["socket_transport_observed"] is True
    assert observation["configured_interface_observed"] is True
    assert observation["evidence_level"] == "observed_debug_log"


def test_network_observation_rejects_a_missing_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "nccl.log"
    log.write_text("NCCL INFO NET/Socket : Using [0]eth0:10.0.0.1\n")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "ens22f0")
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")

    with pytest.raises(RuntimeError, match="does not confirm"):
        _MODULE._network_observation(log)


def test_checked_in_a800_multinode_evidence_is_narrow_and_self_consistent() -> None:
    payload = json.loads(_A800_ARTIFACT.read_text(encoding="utf-8"))
    evidence = payload["evidence"]
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )

    assert payload["evidence_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert evidence["status"] == "passed"
    assert evidence["scope"] == {
        "distribution_semantics": "sharded_across_ranks",
        "dtype": "complex128",
        "execution": "forward_statevector",
        "local_world_size": 1,
        "n_wires": 5,
        "node_count": 2,
        "world_size": 2,
    }
    assert evidence["scalability_claim_allowed"] is False
    assert evidence["release_gate_allowed"] is False

    metrics = evidence["observations"]["numerical_metrics"]
    assert metrics["statevector_max_abs_error"] <= 1e-10
    assert metrics["statevector_norm_error"] <= 1e-10
    assert metrics["statevector_infidelity"] <= 1e-10

    network = evidence["observations"]["network"]
    assert network["route"] == "socket"
    assert network["configured_interface"] == "ens22f0"
    assert network["socket_transport_observed"] is True
    assert network["configured_interface_observed"] is True

    ranks = evidence["observations"]["rank_records"]
    assert {item["rank"] for item in ranks} == {0, 1}
    assert len({item["hostname_sha256"] for item in ranks}) == 2
    assert len({item["device_uuid"] for item in ranks}) == 2
    assert all(item["rank_ownership"]["local_amplitudes"] == 16 for item in ranks)
    assert all(item["inter_node_communication_count"] > 0 for item in ranks)
    assert all(item["inter_node_communication_bytes"] > 0 for item in ranks)
    assert all(item["intra_node_communication_count"] == 0 for item in ranks)
    assert "production_performance_not_measured" in evidence["claim_blockers"]
