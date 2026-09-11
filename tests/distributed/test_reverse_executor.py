"""CI-seeded two-rank reverse-mode execution."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]
ROOT = Path(__file__).resolve().parents[2]


def test_two_rank_gloo_backward_matches_dense_autograd():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/statevector_reverse_executor.py"),
            "--backend",
            "gloo",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
        env=dict(os.environ),
    )
    assert (
        completed.stdout.count(
            '"backward_distribution_semantics": "sharded_across_ranks"'
        )
        == 2
    )
    assert completed.stdout.count('"backward_uses_full_state_replay": false') == 2
