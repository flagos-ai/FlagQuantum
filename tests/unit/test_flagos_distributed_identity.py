from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flagquantum.runtime.distributed import (
    DistributedIdentity,
    context,
    require_verified_flagcx,
)
from flagquantum.runtime.distributed.identity import (
    DistributedIdentityError,
    build_distributed_identity,
)

pytestmark = pytest.mark.unit


def test_backend_inference_uses_platform_availability(monkeypatch):
    platform = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setattr(context, "get_platform_runtime", lambda kind: platform)
    assert context._infer_backend() == "nccl"


def test_flagos_identity_is_fail_closed_without_provider_evidence():
    identity = build_distributed_identity(
        outer_backend="flagos",
        logical_device="flagos:1",
        rank=1,
        world_size=2,
        process_group_initialized=True,
        platform_identity={
            "provider": "torch_fl",
            "provider_version": "test-version",
        },
    )

    summary = identity.to_dict()
    assert summary["schema"] == "flagquantum_distributed_identity_v1"
    assert summary["outer_backend"] == "flagos"
    assert summary["provider"] == "torch_fl"
    assert summary["provider_version"] == "test-version"
    assert summary["inner_backend"] is None
    assert summary["inner_backend_verified"] is False
    assert summary["flagcx_route_status"] == "unverified"
    assert summary["host_staging_observed"] is None
    assert summary["communication_claim_allowed"] is False
    assert "flagcx_inner_backend_unverified" in summary["blockers"]
    assert "host_staging_unverified" in summary["blockers"]

    with pytest.raises(DistributedIdentityError, match="not verified"):
        require_verified_flagcx(identity)


def test_flagcx_identity_rejects_unverified_or_host_staged_claims():
    with pytest.raises(ValueError, match="verified FlagCX"):
        DistributedIdentity(
            outer_backend="flagos",
            logical_device="flagos:0",
            rank=0,
            world_size=2,
            process_group_initialized=True,
            provider="torch_fl",
            inner_backend="flagcx",
            inner_backend_verified=False,
            flagcx_route_verified=True,
            host_staging_observed=False,
        )

    with pytest.raises(ValueError, match="verified FlagCX"):
        DistributedIdentity(
            outer_backend="flagos",
            logical_device="flagos:0",
            rank=0,
            world_size=2,
            process_group_initialized=True,
            provider="torch_fl",
            inner_backend="flagcx",
            inner_backend_verified=True,
            flagcx_route_verified=True,
            host_staging_observed=True,
        )


def test_flagos_backend_and_device_must_be_requested_together(monkeypatch):
    activated = False

    def fail_if_activated(local_rank: int):
        nonlocal activated
        activated = True
        raise AssertionError(f"unexpected activation for local rank {local_rank}")

    monkeypatch.setattr(context.dist, "is_available", lambda: True)
    monkeypatch.setattr(context, "activate_flagos_device", fail_if_activated)

    with pytest.raises(ValueError, match="requires device='flagos"):
        context.init_torch_distributed(backend="flagos", device="cpu")
    with pytest.raises(ValueError, match="requires backend='flagos"):
        context.init_torch_distributed(backend="gloo", device="flagos")
    with pytest.raises(ValueError, match="conflicts with LOCAL_RANK"):
        context.init_torch_distributed(
            backend="flagos", device="flagos:1", local_rank=0
        )

    assert activated is False


def test_flagos_initialization_uses_public_process_group_boundary(monkeypatch):
    state = {"initialized": False}
    calls: list[dict[str, object]] = []

    def init_process_group(**kwargs):
        calls.append(kwargs)
        state["initialized"] = True

    monkeypatch.setattr(context.dist, "is_available", lambda: True)
    monkeypatch.setattr(context.dist, "is_initialized", lambda: state["initialized"])
    monkeypatch.setattr(context.dist, "init_process_group", init_process_group)
    monkeypatch.setattr(context.dist, "get_backend", lambda: "flagos")
    monkeypatch.setattr(context.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(context.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        context,
        "activate_flagos_device",
        lambda local_rank: (
            torch.device("cpu"),
            {
                "provider": "torch_fl",
                "provider_version": "test-version",
                "device_type": "flagos",
            },
        ),
    )

    distributed_context = context.init_torch_distributed(
        backend="flagos",
        device="flagos",
        rank=1,
        world_size=2,
        local_rank=1,
        force_initialize=True,
    )

    assert len(calls) == 1
    assert calls[0]["backend"] == "flagos"
    assert calls[0]["rank"] == 1
    assert calls[0]["world_size"] == 2
    assert distributed_context.backend == "flagos"
    assert distributed_context.initialized is True
    assert distributed_context.initialized_by_flagquantum is True
    assert distributed_context.identity is not None
    assert distributed_context.identity.logical_device == "flagos:1"
    assert distributed_context.identity.provider == "torch_fl"
    assert distributed_context.identity.flagcx_route_verified is False
    assert (
        distributed_context.summary()["distributed_identity"]["flagcx_route_status"]
        == "unverified"
    )


def test_flagos_attach_rejects_an_existing_different_backend(monkeypatch):
    monkeypatch.setattr(context.dist, "is_available", lambda: True)
    monkeypatch.setattr(context.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(context.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(
        context,
        "activate_flagos_device",
        lambda local_rank: (torch.device("cpu"), {"provider": "torch_fl"}),
    )

    with pytest.raises(RuntimeError, match="does not match"):
        context.init_torch_distributed(
            backend="flagos",
            device="flagos",
            rank=0,
            world_size=2,
            local_rank=0,
        )
