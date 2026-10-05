from __future__ import annotations

import importlib
import math
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from examples.qdiffusion_kaiwu.qdiffusion_system_development_probe import (
    _load_pinned_qdiffusion_api,
    _run_qdiffusion_slice,
    _validate_imported_module_tree,
)
from flagquantum.ecosystem.kaiwu import KaiwuSampler
from flagquantum.remote.kaiwu import (
    KaiwuTaskReceipt,
    KaiwuTaskResult,
    new_receipt,
)
from flagquantum.remote.kaiwu.contracts import (
    FrozenIsingMatrix,
    KaiwuTaskMode,
)

pytestmark = pytest.mark.integration
SOURCE_CONFORMANCE = "FLAGQUANTUM_TEST_KAIWU_SOURCE"


def _require_plugin_source() -> tuple[object, Path]:
    if os.environ.get(SOURCE_CONFORMANCE) != "1":
        pytest.skip(f"set {SOURCE_CONFORMANCE}=1 through the pinned source runner")
    module = importlib.import_module("kaiwu.torch_plugin")
    return module, Path(str(module.__file__)).resolve().parents[3]


class _ExactShapeClient:
    def __init__(self) -> None:
        self.submit_calls = 0

    def submit(
        self,
        matrix: FrozenIsingMatrix,
        *,
        task_name: str,
        mode: KaiwuTaskMode,
        requested_samples: int,
        project_no: str | None,
    ) -> KaiwuTaskReceipt:
        self.submit_calls += 1
        return new_receipt(
            task_name=task_name,
            matrix=matrix,
            mode=mode,
            requested_samples=requested_samples,
            project_no=project_no,
            provider_task_id=f"plugin-fake-{self.submit_calls}",
            provider_target="SPQC-plugin-fake",
        )

    def query_status(self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix) -> str:
        del receipt, matrix
        return "Completed"

    def fetch_result(
        self, receipt: KaiwuTaskReceipt, matrix: FrozenIsingMatrix
    ) -> KaiwuTaskResult:
        sample = tuple(1 for _ in matrix)
        samples = tuple(sample for _ in range(receipt.requested_samples))
        energy = -sum(
            sample[row] * matrix[row][column] * sample[column]
            for row in range(len(matrix))
            for column in range(len(matrix))
        )
        return KaiwuTaskResult(
            receipt,
            samples,
            tuple(energy for _ in samples),
            "Completed",
            {"fallback_occurred": False},
        )


def test_sampler_runs_through_plugin_condition_sample_without_sdk_objects() -> None:
    kaiwu_plugin, _ = _require_plugin_source()
    torch.manual_seed(7)
    machine = kaiwu_plugin.BoltzmannMachine(num_nodes=4, device="cpu")
    with torch.no_grad():
        machine.linear_bias.copy_(torch.tensor([0.0, 0.5, -0.25, 0.75]))
        machine.quadratic_coef.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 1.0, -0.5],
                    [0.0, 0.0, -0.75, 0.25],
                    [1.0, -0.75, 0.0, 0.0],
                    [-0.5, 0.25, 0.0, 0.0],
                ]
            )
        )

    client = _ExactShapeClient()
    sampler = KaiwuSampler(
        client=client,
        task_name="plugin-condition",
        requested_samples=10,
        max_remote_calls=2,
        poll_interval=0.01,
    )
    visible = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

    sampled = machine.condition_sample(sampler, visible)

    assert sampled.shape == (20, 4)
    torch.testing.assert_close(sampled[:10, :2], visible[0].expand(10, -1))
    torch.testing.assert_close(sampled[10:, :2], visible[1].expand(10, -1))
    assert bool(((sampled[:, 2:] == 0) | (sampled[:, 2:] == 1)).all())
    assert client.submit_calls == 2
    assert sampler.remote_call_count == 2
    assert len(sampler.receipts) == 2


def test_qdiffusion_development_slice_uses_bounded_flagquantum_sampler() -> None:
    _, plugin_root = _require_plugin_source()
    record = _run_qdiffusion_slice(torch.device("cpu"), plugin_root=plugin_root)

    assert math.isfinite(record["objective"])
    assert record["gradient_norm"] > 0
    assert record["parameter_delta_max"] > 0
    assert record["token_constraints_passed"] is True
    assert 0 < record["remote_call_count"] <= record["remote_call_budget"]
    assert record["task_count"] == record["remote_call_count"]
    assert record["fallback_occurred"] is False
    transfers = record["transfer_accounting"]
    assert transfers["matrix_origin_device"] == "cpu"
    assert transfers["returned_sample_target_device"] == "cpu"
    assert len(transfers["sampler_boundaries"]) >= record["remote_call_count"]
    assert all(
        boundary["input_type"] == "numpy.ndarray"
        and boundary["input_device"] == "cpu"
        and boundary["canonical_device"] == "cpu"
        and boundary["returned_storage"] == "cpu_numpy"
        for boundary in transfers["sampler_boundaries"]
    )


def test_qdiffusion_slice_rejects_preloaded_plugin_outside_reviewed_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plugin_root = tmp_path / "reviewed-plugin"
    package_root = plugin_root / "src" / "kaiwu" / "torch_plugin"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "qdiffusion.py").write_text("", encoding="utf-8")
    namespace = ModuleType("kaiwu")
    namespace.__path__ = []  # type: ignore[attr-defined]
    wrong_plugin = ModuleType("kaiwu.torch_plugin")
    wrong_plugin.__file__ = str(tmp_path / "unreviewed" / "__init__.py")
    wrong_qdiffusion = ModuleType("kaiwu.torch_plugin.qdiffusion")
    wrong_qdiffusion.__file__ = str(tmp_path / "unreviewed" / "qdiffusion.py")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "kaiwu", namespace)
    monkeypatch.setitem(sys.modules, "kaiwu.torch_plugin", wrong_plugin)
    monkeypatch.setitem(sys.modules, "kaiwu.torch_plugin.qdiffusion", wrong_qdiffusion)

    with pytest.raises(RuntimeError, match="outside --plugin-root"):
        _load_pinned_qdiffusion_api(plugin_root)


def test_qdiffusion_slice_rejects_transitive_module_outside_reviewed_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plugin_root = tmp_path / "reviewed-plugin"
    package_root = plugin_root / "src" / "kaiwu" / "torch_plugin"
    package_root.mkdir(parents=True)
    package_init = package_root / "__init__.py"
    qdiffusion_source = package_root / "qdiffusion.py"
    package_init.write_text("", encoding="utf-8")
    qdiffusion_source.write_text("", encoding="utf-8")
    namespace = ModuleType("kaiwu")
    namespace.__path__ = [str(package_root.parent)]  # type: ignore[attr-defined]
    torch_plugin = ModuleType("kaiwu.torch_plugin")
    torch_plugin.__file__ = str(package_init)
    qdiffusion = ModuleType("kaiwu.torch_plugin.qdiffusion")
    qdiffusion.__file__ = str(qdiffusion_source)
    wrong_transitive = ModuleType("kaiwu.torch_plugin.abstract_boltzmann_machine")
    wrong_transitive.__file__ = str(tmp_path / "unreviewed" / "abstract.py")
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in tuple(sys.modules):
        if name == "kaiwu.torch_plugin" or name.startswith("kaiwu.torch_plugin."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "kaiwu", namespace)
    monkeypatch.setitem(sys.modules, "kaiwu.torch_plugin", torch_plugin)
    monkeypatch.setitem(sys.modules, "kaiwu.torch_plugin.qdiffusion", qdiffusion)
    monkeypatch.setitem(
        sys.modules,
        "kaiwu.torch_plugin.abstract_boltzmann_machine",
        wrong_transitive,
    )

    with pytest.raises(RuntimeError, match="abstract_boltzmann_machine.*outside"):
        _load_pinned_qdiffusion_api(plugin_root)


def test_module_tree_validates_namespace_and_every_transitive_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected_root = tmp_path / "reviewed" / "dplm"
    expected_root.mkdir(parents=True)
    namespace = ModuleType("dplm")
    namespace.__path__ = [str(expected_root)]  # type: ignore[attr-defined]
    workflow = ModuleType("dplm.workflows.train")
    workflow.__file__ = str(expected_root / "workflows" / "train.py")
    monkeypatch.setitem(sys.modules, "dplm", namespace)
    monkeypatch.setitem(sys.modules, "dplm.workflows.train", workflow)

    _validate_imported_module_tree(
        module_prefix="dplm", expected_root=expected_root, label="DPLM"
    )

    workflow.__file__ = str(tmp_path / "unreviewed" / "train.py")
    with pytest.raises(RuntimeError, match="dplm.workflows.train.*outside"):
        _validate_imported_module_tree(
            module_prefix="dplm", expected_root=expected_root, label="DPLM"
        )
