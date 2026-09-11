from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.distributed_cpu
def test_pipeline_order_bounds_and_cleanup(tmp_path: Path) -> None:
    output = tmp_path / "pipeline.json"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "tests/distributed/mps_pipeline_runtime.py",
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).parents[2],
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    records = json.loads(output.read_text())
    assert all(item["value_error"] < 1e-12 for item in records)
    assert all(item["gradient_error"] < 1e-12 for item in records)
    assert all(item["partial_final_count"] == 5 for item in records)
    assert all(item["cancelled_drain_count"] == 2 for item in records)
    assert all(item["batch_one_count"] == 1 for item in records)
    assert all(item["window_drain_points"] == [3, 5] for item in records)
    assert all(item["preflight_fault_drained"] for item in records)
