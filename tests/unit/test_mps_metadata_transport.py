from __future__ import annotations

import json

import torch

from flagquantum.runtime.backends.mps import metadata_transport


def test_all_gather_json_uses_variable_length_tensor_payloads(monkeypatch):
    local = {"rank": 0, "records": []}
    remote = {"rank": 1, "records": [{"bond": 2, "discarded_weight": 0.125}]}
    local_bytes = json.dumps(local, separators=(",", ":"), sort_keys=True).encode()
    remote_bytes = json.dumps(remote, separators=(",", ":"), sort_keys=True).encode()

    monkeypatch.setattr(metadata_transport.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(metadata_transport.dist, "get_world_size", lambda: 2)

    def fake_all_gather(outputs, value):
        if value.dtype == torch.int64:
            outputs[0].fill_(len(local_bytes))
            outputs[1].fill_(len(remote_bytes))
            return
        outputs[0].copy_(value)
        outputs[1].zero_()
        outputs[1][: len(remote_bytes)] = torch.tensor(
            list(remote_bytes), dtype=torch.uint8
        )

    monkeypatch.setattr(metadata_transport.dist, "all_gather", fake_all_gather)

    assert metadata_transport.all_gather_json(local) == (local, remote)


def test_collective_device_uses_current_cuda_device_for_nccl(monkeypatch):
    monkeypatch.setattr(metadata_transport.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(metadata_transport.torch.cuda, "current_device", lambda: 3)

    assert metadata_transport._collective_device() == torch.device("cuda", 3)
