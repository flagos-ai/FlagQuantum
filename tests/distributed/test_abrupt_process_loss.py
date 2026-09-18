"""Elastic-launcher acceptance for an uncatchable participant SIGKILL."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.distributed.statevector_abrupt_failure_runtime import (
    PEER_LOSS_TIMEOUT_SECONDS,
)

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]
ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "tests/distributed/statevector_abrupt_failure_runtime.py"
# The cost of starting two ranks, importing torch, and building a process group,
# plus the survivor's own wait for the launcher to act on the loss it reports.
# Measured on a shared host at about 8s idle and about 38s with the machine
# oversubscribed five to one, so it is sized from that measurement rather than
# from the wait, which is the small term.
# `tests/unit/test_runtime_harness_deadlines.py` pins both numbers, so neither can
# be quietly shrunk below what was observed.
MEASURED_STARTUP_UNDER_LOAD_SECONDS = 38.0
STARTUP_ALLOWANCE_SECONDS = 120.0
SUBPROCESS_DEADLINE_SECONDS = STARTUP_ALLOWANCE_SECONDS + 2 * PEER_LOSS_TIMEOUT_SECONDS


def test_abrupt_rank_loss_terminates_elastic_job_with_diagnostics():
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
        timeout=SUBPROCESS_DEADLINE_SECONDS,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    # A launcher that failed to act on the loss would hang, and the deadline
    # above would raise TimeoutExpired instead of returning here. Boundedness is
    # therefore this returning at all -- asserting an elapsed bound tighter than
    # the deadline would measure the host rather than the launcher, which is what
    # a shared runner makes unpredictable.
    assert completed.returncode != 0
    assert '"event": "injected_abrupt_process_loss"' in combined
    assert "SIGKILL" in combined
    assert "rank" in combined.lower()
    peer_loss_diagnostics = '"event": "peer_process_loss_detected"' in combined or (
        "ProcessGroup" in combined
        and ("ChildFailedError" in combined or "Signal 9" in combined)
    )
    assert peer_loss_diagnostics, combined
