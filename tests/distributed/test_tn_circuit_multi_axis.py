import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]
ROOT = Path(__file__).resolve().parents[2]


def test_eight_rank_real_circuit_multi_axis_contraction():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=8",
            str(ROOT / "tests/distributed/tn_circuit_multi_axis_runtime.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env=dict(os.environ),
    )

    assert completed.stdout.count('"circuit_multi_axis_passed": true') == 8
    assert completed.stdout.count('"full_target_inputs_materialized": false') == 8
    assert (
        completed.stdout.count('"prefix_full_intermediates_materialized": false') == 8
    )
