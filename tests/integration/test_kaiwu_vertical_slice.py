from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import numpy as np
import pytest

import flagquantum.remote.kaiwu.client as client_module
from flagquantum.ecosystem.kaiwu import KaiwuSampler
from flagquantum.remote.kaiwu import KaiwuSDKClient, KaiwuSDKEnvironment

pytestmark = pytest.mark.integration


class _ImmediateOptimizer:
    created_options: ClassVar[list[dict[str, object]]] = []
    solve_calls: ClassVar[int] = 0

    def __init__(self, **options: object) -> None:
        type(self).created_options.append(options)

    def solve(self, matrix: np.ndarray) -> np.ndarray:
        assert matrix.tolist() == [
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.0],
        ]
        type(self).solve_calls += 1
        return np.asarray([[1, -1, 1]] * 10, dtype=np.int8)


def test_sampler_sdk_client_vertical_slice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Compose Ecosystem and Remote without claiming provider execution."""

    _ImmediateOptimizer.created_options = []
    _ImmediateOptimizer.solve_calls = 0
    checkpoint_manager = SimpleNamespace(save_dir="original")
    fake_sdk = SimpleNamespace(
        common=SimpleNamespace(CheckpointManager=checkpoint_manager),
        cim=SimpleNamespace(
            CIMOptimizer=_ImmediateOptimizer,
            TaskMode=SimpleNamespace(OPTIMIZATION="optimization", SAMPLING="sampling"),
        ),
    )
    monkeypatch.setattr(
        client_module,
        "_initialize_preflighted_kaiwu_license",
        Mock(return_value=KaiwuSDKEnvironment("1.3.1", "3.10.18")),
    )
    monkeypatch.setattr(
        client_module, "_preflight_kaiwu_sdk", Mock(return_value=fake_sdk)
    )

    client = KaiwuSDKClient(checkpoint_dir=tmp_path)
    sampler = KaiwuSampler(
        client=client,
        task_name="vertical-slice",
        project_no="CPQC-test",
        requested_samples=10,
        max_remote_calls=1,
    )
    matrix = np.asarray(
        [[0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]],
        dtype=np.float64,
    )

    samples = sampler.solve(matrix)

    assert samples.shape == (10, 3)
    assert samples.dtype == np.int8
    assert sampler.remote_call_count == 1
    assert _ImmediateOptimizer.solve_calls == 1
    assert len(_ImmediateOptimizer.created_options) == 1
    assert _ImmediateOptimizer.created_options[0]["project_no"] == "CPQC-test"
    assert checkpoint_manager.save_dir == "original"

    result = sampler.last_result
    assert result is not None
    assert result.energies == (2.0,) * 10
    assert result.metadata["fallback_occurred"] is False
    assert result.metadata["provider_task_id_available"] is False
    assert result.metadata["provider_target_available"] is False

    receipt = sampler.receipts[0]
    recovery_path = client.recovery_receipt_path(receipt)
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    assert recovery["receipt"]["task_name"] == receipt.task_name
    assert recovery["receipt"]["project_no"] == "CPQC-test"
    assert recovery_path.stat().st_mode & 0o777 == 0o600

    repeated = sampler.solve(matrix)
    assert np.array_equal(repeated, samples)
    assert _ImmediateOptimizer.solve_calls == 1
