from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.distributed_cpu
def test_same_classifier_code_trains_locally_and_state_sharded() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/hybrid_model_runtime.py"),
        ],
        cwd=ROOT,
        check=True,
        timeout=180,
    )
