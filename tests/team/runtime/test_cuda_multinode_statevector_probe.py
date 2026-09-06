"""Contract tests for the two-node CUDA statevector probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "tools" / "probe_cuda_multinode_statevector.py"
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
