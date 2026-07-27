from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.distributed_cpu
def test_module_local_and_sharded_value_gradient_contract() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/module_runtime.py"),
        ],
        cwd=ROOT,
        check=True,
        timeout=120,
    )
