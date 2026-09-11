import os
import subprocess
import sys
from pathlib import Path

import pytest

# Unique module basename avoids collision with the benchmark-contract test.

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]
ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "tests/distributed/mps_stability_fault_runtime.py"


@pytest.mark.parametrize(
    "mode",
    (
        "topology",
        "generation",
        "manifest_missing",
        "manifest_corrupt",
        "committed_corruption",
        "unshared_checkpoint_root",
        "accidental_overwrite",
        "active_writer_lease",
        "premature_lease_break",
    ),
)
def test_checkpoint_contract_faults_fail_closed(mode, tmp_path):
    env = dict(os.environ, FQ_TEST_CHECKPOINT=str(tmp_path / mode))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(RUNTIME),
            "--mode",
            mode,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert completed.stdout.count(f"expected_failure:{mode}") == 2


def test_explicit_stale_writer_lease_recovery_is_bounded(tmp_path):
    mode = "stale_lease_recovery"
    env = dict(os.environ, FQ_TEST_CHECKPOINT=str(tmp_path / mode))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(RUNTIME),
            "--mode",
            mode,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert completed.stdout.count(f"expected_success:{mode}") == 2


@pytest.mark.parametrize(
    "mode", ("interrupted", "rank_exception", "cuda_oom", "collective_timeout")
)
def test_participant_failure_terminates_launcher(mode, tmp_path):
    env = dict(os.environ, FQ_TEST_CHECKPOINT=str(tmp_path / mode))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--monitor-interval=0.1",
            "--nproc-per-node=2",
            str(RUNTIME),
            "--mode",
            mode,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode != 0
    assert not list((tmp_path / mode).glob("*.tmp")) or mode == "interrupted"
