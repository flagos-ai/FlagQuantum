from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.distributed_cpu
def test_ddp_and_state_sharding_use_orthogonal_groups_without_double_reduction() -> (
    None
):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            str(ROOT / "tests/distributed/hybrid_parallel_runtime.py"),
        ],
        cwd=ROOT,
        check=True,
        timeout=180,
    )
