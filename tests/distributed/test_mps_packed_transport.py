import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]
ROOT = Path(__file__).resolve().parents[2]


def test_two_rank_packed_values_mismatch_and_cleanup():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/mps_packed_transport_runtime.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ),
    )
    assert completed.stdout.count('"values_passed": true') == 2
    assert completed.stdout.count('"site_observations_passed": true') == 2
    assert completed.stdout.count('"shape_mismatch_passed": true') == 2
    assert completed.stdout.count('"cleanup_verified": true') == 2
