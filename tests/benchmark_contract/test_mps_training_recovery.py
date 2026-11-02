"""Training/recovery behavior and honest fixed-workload report checks."""

from __future__ import annotations

import copy
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

np = pytest.importorskip("numpy")

from benchmarks.build_mps_training_recovery_report import report, validate  # noqa: E402
from benchmarks.internal.evidence.mps_training_recovery import (  # noqa: E402
    execute,
    local_diagnostic,
    tensor_tree,
    tree_max_error,
)
from tools.run_mps_capacity_series import point_command  # noqa: E402

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.integration]
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "benchmarks/manifests/mps_training_recovery_v1.json"


@pytest.mark.parametrize("world", (1, 2, 4, 8, 16))
def test_deployment_does_not_change_training_geometry(world: int) -> None:
    command = point_command(
        world,
        0,
        Path("/tmp/source"),
        Path("/tmp/out"),
        "localhost",
        29660,
        training_series=True,
    )
    assert command[-4:-2] == [
        "--recovery-protocol",
        "/tmp/source/benchmarks/manifests/mps_training_recovery_v1.json",
    ]
    assert "--n-wires" not in command and "--layers" not in command
    assert command[command.index("--nproc-per-node") + 1] == str(min(world, 8))


def test_truncated_derivative_error_is_separate_from_sharding() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    approximate = local_diagnostic(protocol["initial_parameters"], protocol)
    assert approximate["max_bond"] == 4
    assert approximate["derivative_semantics_max_abs_difference"] > 0.1
    protocol.update(n_sites=4, layers=1, max_bond=4)
    exact = local_diagnostic(protocol["initial_parameters"], protocol)
    assert exact["derivative_semantics_max_abs_difference"] < 1e-7


def test_optimizer_state_comparison_detects_changed_moment() -> None:
    original = tensor_tree(
        {"exp_avg": torch.tensor([0.1], dtype=torch.float64), "step": 4}
    )
    changed = copy.deepcopy(original)
    changed["exp_avg"]["values"][0] += 0.01
    assert tree_max_error(original, changed) == pytest.approx(0.01)
    changed["exp_avg"]["dtype"] = "torch.float32"
    with pytest.raises(ValueError, match="metadata"):
        tree_max_error(original, changed)


def test_training_refuses_object_gather(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks.internal.evidence import mps_training_recovery as recovery

    monkeypatch.setattr(recovery.dist, "barrier", lambda: None)
    monkeypatch.setattr(
        recovery.fqxd,
        "train_distributed_mps",
        lambda *args, **kwargs: recovery.dist.all_gather_object([], {}),
    )
    protocol = json.loads(PROTOCOL.read_text())
    with pytest.raises(AssertionError, match="gather forbidden"):
        execute(protocol, torch.device("cpu"), tmp_path, 1)


@pytest.fixture(scope="module")
def measured_recovery(tmp_path_factory: pytest.TempPathFactory) -> dict:
    directory = tmp_path_factory.mktemp("mps-recovery")
    protocol = json.loads(PROTOCOL.read_text())
    protocol.update(n_sites=8, layers=2, steps=4, checkpoint_step=2)
    manifest = directory / "test-protocol.json"
    manifest.write_text(json.dumps(protocol))
    output = directory / "world-2.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc-per-node=2",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=127.0.0.1:0",
            "-m",
            "benchmarks.internal.evidence.mps_training_acceptance",
            "--recovery-protocol",
            str(manifest),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=dict(
            os.environ,
            FQ_MPS_TRAINING_BACKEND="gloo",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        ),
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    point = json.loads(output.read_text())
    # Validation of synthetic mutations below is independent of a developer's
    # dirty checkout. The subprocess retains the actual dirty flag in raw JSON.
    point["source"]["dirty"] = False
    return point


def test_two_rank_updates_checkpoint_and_owner_state_match(
    measured_recovery: dict,
) -> None:
    point = measured_recovery
    validate(point, 2)
    assert point["recovery_passed"]
    assert all(max(r["recovery_errors"].values()) <= 2e-5 for r in point["ranks"])
    assert all(r["full_state_gather_forbidden_during_training"] for r in point["ranks"])
    assert (
        max(d["sharding_objective_max_abs_error"] for d in point["matched_local_mps"])
        < 2e-5
    )
    assert (
        max(
            d["sharding_projected_gradient_max_abs_error"]
            for d in point["matched_local_mps"]
        )
        < 2e-5
    )


@pytest.mark.parametrize(
    "mutation", ("owner", "optimizer", "trace", "protocol", "local_error")
)
def test_report_rejects_corrupt_recovery_evidence(
    measured_recovery: dict, mutation: str
) -> None:
    point = copy.deepcopy(measured_recovery)
    if mutation == "owner":
        point["ranks"][0]["checkpoint_resumed"]["owned_indices"] = [99]
    elif mutation == "optimizer":
        point["ranks"][0]["checkpoint_resumed"]["optimizer_state"]["state"] = {}
    elif mutation == "trace":
        point["ranks"][0]["uninterrupted"]["step_metrics"].pop()
    elif mutation == "local_error":
        point["matched_local_mps"][0]["sharding_projected_gradient_max_abs_error"] = 10
    else:
        point["protocol"]["n_sites"] += 1
    with pytest.raises(ValueError):
        validate(point, 2)


def test_missing_outcome_is_not_treated_as_fast_scaling(
    measured_recovery: dict, tmp_path: Path
) -> None:
    (tmp_path / "world-2.json").write_text(json.dumps(measured_recovery))
    with pytest.raises(ValueError, match="missing deployment outcome"):
        report(tmp_path, [1, 2])
    (tmp_path / "world-1-outcome.json").write_text(
        json.dumps({"world_size": 1, "status": "cuda_oom"})
    )
    payload, text = report(tmp_path, [1, 2])
    assert payload["rows"][0]["measured_speedup"] is None
    assert payload["rows"][0]["measured_efficiency"] is None
    assert payload["unsuccessful_outcomes"][0]["status"] == "cuda_oom"
    assert "cuda_oom" in text
    point = tmp_path / "world-2.json"
    point.with_suffix(".json.gz").write_bytes(
        gzip.compress(point.read_bytes(), mtime=0)
    )
    point.unlink()
    compressed, _ = report(tmp_path, [1, 2])
    assert compressed["rows"] == payload["rows"]
