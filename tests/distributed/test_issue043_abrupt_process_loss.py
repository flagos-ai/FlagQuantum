"""Elastic-launcher acceptance for an uncatchable participant SIGKILL."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]
ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "tests/distributed/statevector_abrupt_failure_runtime.py"


def test_abrupt_rank_loss_terminates_elastic_job_with_diagnostics():
    started = time.monotonic()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--monitor-interval=0.1",
            "--max-restarts=0",
            "--nproc-per-node=2",
            str(RUNTIME),
            "--backend=gloo",
        ],
        cwd=ROOT,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    elapsed = time.monotonic() - started
    combined = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert elapsed < 30
    assert '"event": "injected_abrupt_process_loss"' in combined
    assert "SIGKILL" in combined
    assert "rank" in combined.lower()
    peer_loss_diagnostics = '"event": "peer_process_loss_detected"' in combined or (
        "ProcessGroup" in combined
        and ("ChildFailedError" in combined or "Signal 9" in combined)
    )
    assert peer_loss_diagnostics, combined
