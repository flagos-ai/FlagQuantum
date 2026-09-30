from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.distributed
@pytest.mark.distributed_cpu
@pytest.mark.parametrize("world_size", (2, 3))
def test_rank_sharded_mps_adaptive_bond_planning_is_measured(world_size: int) -> None:
    """A sharded run must record and grow the bonds it actually truncated.

    `mps_adaptive_bond_runtime.py` owns the assertions; it fails on a build where
    the sharded two-site kernel discards its split metadata, or where the merged
    adaptive plan takes the growth instruction from one rank's plan only.
    """
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            str(ROOT / "tests/distributed/mps_adaptive_bond_runtime.py"),
        ],
        cwd=ROOT,
        check=True,
        timeout=300,
    )
