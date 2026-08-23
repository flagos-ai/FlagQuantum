import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_accel]
ROOT = Path(__file__).resolve().parents[2]


def test_eight_gpu_dynamic_multi_axis_redistribution():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=8",
            str(ROOT / "tests/distributed/tn_multi_axis_redistribution_runtime.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
    )
    assert (
        completed.stdout.count('"dynamic_multi_axis_redistribution_passed": true') == 8
    )
