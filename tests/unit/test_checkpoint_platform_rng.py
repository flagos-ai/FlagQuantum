"""Checkpoint RNG uses the owning platform and reads legacy CPU payloads."""

from types import SimpleNamespace

import pytest
import torch

from flagquantum.runtime import training_state

pytestmark = pytest.mark.unit


def test_optional_platform_rng_round_trip_and_legacy_cpu_payload(monkeypatch):
    restored = []
    state = torch.tensor([3, 7], dtype=torch.uint8)
    cuda = SimpleNamespace(is_available=lambda: False)
    platform = SimpleNamespace(
        is_available=lambda: True,
        rng_state=lambda device: state.clone(),
        restore_rng_state=lambda device, value: restored.append((device, value)),
    )
    monkeypatch.setattr(
        training_state,
        "get_platform_runtime",
        lambda kind: cuda if kind == "cuda" else platform,
    )
    seed = training_state.SeedContract(13, 17, 19, (0, 13))
    module = torch.nn.Linear(2, 2, device="meta")
    payload = training_state._rng_state(seed, module)
    assert payload["torch_cuda"] is None
    assert set(payload["platform_rng"]) == {"meta"}
    assert training_state._restore_rng(payload) == seed
    assert restored[0][0] == torch.device("meta")
    torch.testing.assert_close(restored[0][1], state)
    legacy = training_state._rng_state(seed, torch.nn.Linear(2, 2))
    assert "platform_rng" not in legacy
    assert training_state._restore_rng(legacy) == seed
