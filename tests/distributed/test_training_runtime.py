"""CI-seeded training/recovery/fault lifecycle tests."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.distributed.statevector_training_runtime import mode_inner_budget_seconds

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]
ROOT = Path(__file__).resolve().parents[2]
MODES = ("train", "fault", "timeout")
# What two ranks cost before the behaviour under test begins: they start, import
# torch, and build a process group. Measured on a shared host at about 8s idle and
# about 38s with the machine oversubscribed five to one, so the allowance below is
# sized from that measurement rather than from the inner budget, which is the
# small term. `tests/unit/test_runtime_harness_deadlines.py` pins both numbers, so
# neither can be quietly shrunk below what was observed.
MEASURED_STARTUP_UNDER_LOAD_SECONDS = 38.0
STARTUP_ALLOWANCE_SECONDS = 120.0


def subprocess_deadline_seconds(mode: str) -> float:
    """Return the wall-clock deadline for one ``--mode <mode>`` subprocess.

    The script asserts its own boundedness from the inside, and it can only do
    so if this deadline lets it. Below the script's own budget this deadline
    fires first and reports ``TimeoutExpired`` instead, which names neither the
    operation that stalled nor the launcher that failed to act.
    """

    return mode_inner_budget_seconds(mode) + STARTUP_ALLOWANCE_SECONDS


@pytest.mark.parametrize("mode", MODES)
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
        timeout=subprocess_deadline_seconds(mode),
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
        timeout=subprocess_deadline_seconds("crash"),
        env=dict(os.environ),
    )
    assert completed.returncode != 0
    assert "ChildFailedError" in completed.stderr
