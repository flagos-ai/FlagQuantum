import os
import subprocess
import sys
from pathlib import Path

import pytest

from flagquantum.runtime.backends.tensor_network import (
    clear_dynamic_tn_checkpoint_writer_lock,
    inspect_dynamic_tn_checkpoint,
)

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_accel]
ROOT = Path(__file__).resolve().parents[2]


def test_two_gpu_shared_dag_reverse_accumulates_before_consuming(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/tn_shared_dag_reverse_runtime.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0,1",
            "FQ_CHECKPOINT_DIR": str(tmp_path / "dynamic-checkpoint"),
        },
    )

    assert completed.stdout.count('"shared_dag_reverse_passed": true') == 2
    assert completed.stdout.count('"accumulated_cotangent_count": 1') == 2
    assert completed.stdout.count('"rank_consensus_validated": true') == 2
    assert completed.stdout.count('"rank_tensor_preflight_validated": true') == 2
    assert completed.stdout.count('"checkpoint_resume_passed": true') == 2
    assert completed.stdout.count('"completed_after_resume": true') == 2
    assert completed.stdout.count('"durable_checkpoint_reload_passed": true') == 2
    checkpoint = tmp_path / "dynamic-checkpoint"
    assert (checkpoint / "manifest.json").is_file()
    assert len(tuple(checkpoint.glob("rank-*.pt"))) == 2
    audit = inspect_dynamic_tn_checkpoint(checkpoint)
    assert audit.loadable is True
    assert audit.writer_lock_exists is False


def test_two_gpu_shared_dag_reverse_rejects_rank_divergent_plans():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/tn_shared_dag_reverse_runtime.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0,1",
            "FQ_EXPECT_PLAN_MISMATCH": "1",
        },
    )

    assert completed.stdout.count('"segment_consensus_fail_closed": true') == 2


def test_two_gpu_shared_dag_reverse_rejects_missing_rank_tape():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/tn_shared_dag_reverse_runtime.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0,1",
            "FQ_EXPECT_TAPE_MISMATCH": "1",
        },
    )

    assert completed.stdout.count('"tensor_preflight_fail_closed": true') == 2


def test_two_gpu_shared_dag_reverse_rejects_corrupt_checkpoint(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/tn_shared_dag_reverse_runtime.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0,1",
            "FQ_CHECKPOINT_DIR": str(tmp_path / "corrupt-checkpoint"),
            "FQ_CORRUPT_CHECKPOINT": "1",
        },
    )

    assert completed.stdout.count('"checkpoint_corruption_fail_closed": true') == 2


def test_two_gpu_shared_dag_reverse_resumes_after_process_group_restart(
    tmp_path,
):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(ROOT / "tests/distributed/tn_shared_dag_reverse_runtime.py"),
    ]
    checkpoint = tmp_path / "restart-checkpoint"
    common_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0,1",
        "FQ_CHECKPOINT_DIR": str(checkpoint),
    }
    saved = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        env={**common_env, "FQ_CHECKPOINT_STAGE": "save"},
    )
    resumed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        env={**common_env, "FQ_CHECKPOINT_STAGE": "resume"},
    )

    assert saved.stdout.count('"durable_checkpoint_stage_saved": true') == 2
    assert resumed.stdout.count('"process_group_restart_resume_passed": true') == 2
    assert resumed.stdout.count('"shared_dag_reverse_passed": true') == 2


def test_two_gpu_shared_dag_reverse_rejects_uncommitted_checkpoint(
    tmp_path,
):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/tn_shared_dag_reverse_runtime.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0,1",
            "FQ_CHECKPOINT_DIR": str(tmp_path / "uncommitted-checkpoint"),
            "FQ_REMOVE_CHECKPOINT_COMMIT": "1",
        },
    )

    assert completed.stdout.count('"uncommitted_checkpoint_fail_closed": true') == 2


def test_two_gpu_shared_dag_reverse_rejects_competing_writer(tmp_path):
    checkpoint = tmp_path / "writer-conflict"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/tn_shared_dag_reverse_runtime.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0,1",
            "FQ_CHECKPOINT_DIR": str(checkpoint),
            "FQ_PRECREATE_CHECKPOINT_LOCK": "1",
        },
    )

    assert completed.stdout.count('"checkpoint_writer_conflict_fail_closed": true') == 2
    audit = inspect_dynamic_tn_checkpoint(checkpoint)
    assert audit.writer_lock_exists is True
    assert audit.writer_lock_identity is not None
    clear_dynamic_tn_checkpoint_writer_lock(
        checkpoint,
        expected_lock_identity=audit.writer_lock_identity,
    )
    assert inspect_dynamic_tn_checkpoint(checkpoint).writer_lock_exists is False


def test_two_gpu_shared_dag_reverse_rematerializes_missing_forward_value():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/tn_shared_dag_reverse_runtime.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0,1",
            "FQ_USE_FORWARD_PROVIDER": "1",
        },
    )

    assert (
        completed.stdout.count('"forward_provider_rematerialization_passed": true') == 2
    )
    assert completed.stdout.count('"rematerialized_forward_value_count": 3') == 2
    assert completed.stdout.count('"rematerialization_operation_count": 4') == 2
    assert completed.stdout.count('"rematerialization_peak_transient_bytes": 192') == 2
    assert (
        completed.stdout.count('"rematerialization_released_transient_values": 1') == 2
    )
    assert completed.stdout.count('"partial_mesh_forward_reduction_passed": true') == 2


def test_two_gpu_shared_dag_reverse_rejects_rematerialization_over_budget():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/tn_shared_dag_reverse_runtime.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "0,1",
            "FQ_USE_FORWARD_PROVIDER": "1",
            "FQ_REJECT_FORWARD_PROVIDER_BUDGET": "1",
        },
    )

    assert completed.stdout.count('"rematerialization_budget_fail_closed": true') == 2
