from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]
ROOT = Path(__file__).resolve().parents[2]


def test_two_rank_heisenberg_mpo_matches_termwise_value_and_gradient() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/mps_heisenberg_mpo_runtime.py"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env=dict(os.environ),
    )
    assert completed.stdout.count('"heisenberg_mpo_value_passed": true') == 2
    assert completed.stdout.count('"heisenberg_mpo_gradient_passed": true') == 2
    assert completed.stdout.count('"statevector_gradient_parity_passed": true') == 2
    assert completed.stdout.count('"objective_scan_pairs": 1') == 2
