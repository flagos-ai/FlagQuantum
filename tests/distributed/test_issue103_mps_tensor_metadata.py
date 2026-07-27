import json
import os
import subprocess
import sys

import pytest


@pytest.mark.distributed_cpu
def test_tensor_metadata_cache_and_generation_invalidation(tmp_path):
    output = tmp_path / "metadata.json"
    env = {**os.environ, "PYTHONPATH": ".", "FQ_ISSUE103_OUTPUT": str(output)}
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "tests/distributed/mps_tensor_metadata_runtime.py",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=150,
    )
    assert done.returncode == 0, done.stderr
    records = json.loads(output.read_text())
    assert all(r["value_error"] < 1e-7 and r["gradient_error"] < 1e-6 for r in records)
    assert all(r["warm"]["descriptor_cache_hits"] > 0 for r in records)
    assert all(
        r["warm"]["descriptor_message_count"] < r["cold"]["descriptor_message_count"]
        for r in records
    )
    assert all(r["changed"]["descriptor_cache_invalidations"] > 0 for r in records)
    assert all(r["bond_generation_changed_result_finite"] for r in records)
