"""CI-seeded Gloo execution for the sharded statevector forward executor."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]
ROOT = Path(__file__).resolve().parents[2]


def test_two_rank_gloo_forward_matches_rank_local_dense_reference():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/statevector_forward_executor.py"),
            "--backend",
            "gloo",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=dict(os.environ),
    )
    assert (
        completed.stdout.count('"distribution_semantics": "sharded_across_ranks"') == 2
    )
    assert completed.stdout.count('"full_state_materialization": false') == 2
