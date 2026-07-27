"""CI-seeded Gloo lifecycle coverage for ISSUE-052."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]
ROOT = Path(__file__).resolve().parents[2]


def test_two_rank_mps_adam_checkpoint_resume(tmp_path: Path):
    env = dict(os.environ)
    env["FQ_TEST_CHECKPOINT"] = str(tmp_path / "checkpoint")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/mps_training_runtime.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    assert completed.stdout.count('"completed_steps": 3') == 2
    assert completed.stdout.count('"optimizer_state_count": 1') == 2
    assert completed.stdout.count('"rank_useful_work": true') == 2
    assert completed.stdout.count('"full_mps_materialization": false') == 2
    assert completed.stdout.count('"scalability_claim_allowed": false') == 2
    assert completed.stdout.count('"variable_bond_initial_state": true') == 2


@pytest.mark.distributed_accel
@pytest.mark.gpu
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_nccl_mps_training_semantic_matrix(tmp_path: Path, world_size: int):
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CUDA devices")
    env = dict(os.environ)
    env["FQ_TEST_CHECKPOINT"] = str(tmp_path / f"checkpoint-{world_size}")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            str(ROOT / "tests/distributed/mps_training_runtime.py"),
            "--backend",
            "nccl",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert completed.stdout.count('"completed_steps": 3') == world_size
    assert completed.stdout.count('"rank_useful_work": true') == world_size
    assert completed.stdout.count('"full_mps_materialization": false') == world_size
    assert completed.stdout.count('"cleanup_verified": true') == world_size
    assert completed.stdout.count('"variable_bond_initial_state": true') == world_size
