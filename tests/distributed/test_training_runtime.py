"""CI-seeded training/recovery/fault lifecycle tests."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ["train", "fault", "timeout"])
def test_two_rank_training_recovery_and_fail_closed_cleanup(tmp_path: Path, mode: str):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/statevector_training_runtime.py"),
            "--backend",
            "gloo",
            "--checkpoint-dir",
            str(tmp_path / mode),
            "--mode",
            mode,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ),
    )
    assert completed.stdout.count('"cleanup_verified": true') == 2
    if mode == "train":
        assert completed.stdout.count('"optimizer_state_count": 1') == 2
        assert completed.stdout.count('"completed_steps": 4') == 2
    else:
        if mode == "fault":
            assert completed.stdout.count('"cause": "injected_rank_failure"') == 1
            assert completed.stdout.count('"cause": "peer_rank_failure"') == 1
        else:
            assert completed.stdout.count('"cause": "collective_timeout"') == 2
            assert (
                completed.stdout.count(
                    '"last_operation": "rank_did_not_enter_control_barrier"'
                )
                == 1
            )
            assert (
                completed.stdout.count(
                    '"last_operation": "control_plane_monitored_barrier"'
                )
                == 1
            )


def test_two_rank_abrupt_worker_loss_is_bounded_by_torchrun(tmp_path: Path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/statevector_training_runtime.py"),
            "--backend",
            "gloo",
            "--checkpoint-dir",
            str(tmp_path / "crash"),
            "--mode",
            "crash",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ),
    )
    assert completed.returncode != 0
    assert "ChildFailedError" in completed.stderr
