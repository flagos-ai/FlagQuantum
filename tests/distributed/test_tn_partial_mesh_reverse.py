import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_accel]
ROOT = Path(__file__).resolve().parents[2]


def test_eight_gpu_partial_mesh_reverse_subgroup_reductions():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=8",
            str(ROOT / "tests/distributed/tn_partial_mesh_reverse_runtime.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
    )
    assert completed.stdout.count('"partial_mesh_reverse_passed": true') == 8
    assert completed.stdout.count('"partial_remesh_passed": true') == 8
    assert completed.stdout.count('"subgroup_collective_count": 2') == 8
