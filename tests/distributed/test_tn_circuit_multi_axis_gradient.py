import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_accel]
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_multi_gpu_explicit_multi_axis_parameter_gradient(world_size):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            str(ROOT / "tests/distributed/tn_circuit_multi_axis_gradient_runtime.py"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(world_size)),
        },
    )

    assert completed.stdout.count('"multi_axis_gradient_passed": true') == world_size
    assert completed.stdout.count('"nonfinite_cotangent_count": 0') == world_size
    assert (
        completed.stdout.count('"partial_mesh_continuation_passed": true') == world_size
    )
    assert completed.stdout.count('"product_segment_passed": true') == world_size
    assert (
        completed.stdout.count('"product_segment_rank_consensus_validated": true')
        == world_size
    )
    assert (
        completed.stdout.count('"product_segment_tensor_preflight_validated": true')
        == world_size
    )
    assert completed.stdout.count('"product_segment_completed": true') == world_size
    assert (
        completed.stdout.count('"product_parameter_pullback_passed": true')
        == world_size
    )
    assert (
        completed.stdout.count('"product_parameter_nonfinite_gradient_count": 0')
        == world_size
    )
