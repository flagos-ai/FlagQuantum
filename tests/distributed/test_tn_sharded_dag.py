import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]
ROOT = Path(__file__).resolve().parents[2]


def test_two_rank_multi_operation_sharded_tn_dag():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/tn_sharded_dag_runtime.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=dict(os.environ),
    )

    assert completed.stdout.count('"sharded_dag_passed": true') == 2
    assert completed.stdout.count('"full_state_materialization": false') == 2
    assert completed.stdout.count('"silent_statevector_fallback": false') == 2
