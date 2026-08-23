import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]
ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "tests/distributed/mps_checkpoint_cycle_runtime.py"


@pytest.mark.parametrize("optimizer", ("sgd", "adam"))
def test_ten_checkpoint_restart_cycles_match_uninterrupted_training(
    optimizer, tmp_path
):
    env = dict(
        os.environ,
        FQ_TEST_CHECKPOINT=str(tmp_path / optimizer),
        OMP_NUM_THREADS="1",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(RUNTIME),
            "--optimizer",
            optimizer,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["cycles"] == 10
    assert payload["world_size"] == 2
    assert payload["distribution_semantics"] == "sharded_across_ranks"
    assert payload["loss_max_abs_error"] <= 1e-10
    assert payload["parameter_max_abs_error"] <= 1e-10
    assert payload["stale_generation_isolation"] is True
    assert payload["release_gate_allowed"] is False
