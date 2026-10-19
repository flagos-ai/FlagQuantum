"""The native Jiuding bridge consumes every torchrun rank it requests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]


def test_native_job_bridge_runs_two_statevector_shards() -> None:
    runtime = Path(__file__).with_name("jiuding_native_job_runtime.py")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--master-addr=127.0.0.1",
            "--master-port=29501",
            "--nproc-per-node=2",
            str(runtime),
        ],
        check=True,
        timeout=90,
    )
