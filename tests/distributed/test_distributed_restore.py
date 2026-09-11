from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.distributed_cpu
def test_checkpoint_restores_on_equivalent_state_sharding_topology(tmp_path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/training_state_runtime.py"),
            "--checkpoint-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        timeout=180,
    )
