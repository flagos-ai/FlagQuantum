"""CPU semantic test for rank-owned sliced TN reverse execution."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]
ROOT = Path(__file__).resolve().parents[2]


def test_two_rank_sliced_tn_reverse_executes_only_owned_tasks():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/tn_sliced_reverse_runtime.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.stdout.count('"rank_owned_sliced_reverse_passed": true') == 2
    assert (
        completed.stdout.count('"distribution_semantics": "sharded_across_ranks"') == 2
    )
    assert completed.stdout.count('"scalability_claim_allowed": false') == 2
