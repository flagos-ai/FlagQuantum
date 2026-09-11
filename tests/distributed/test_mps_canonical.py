from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.distributed_cpu
@pytest.mark.parametrize("world_size", (2, 3, 4))
def test_distributed_mps_canonicalization_and_truncation(world_size: int) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            str(ROOT / "tests/distributed/mps_canonical_runtime.py"),
        ],
        cwd=ROOT,
        check=True,
        timeout=180,
    )
