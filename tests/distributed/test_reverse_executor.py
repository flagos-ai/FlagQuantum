"""CI-seeded two-rank reverse-mode execution."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]

# The markers sit on the tests, not the module: the gloo run needs no device,
# while the nccl run drives the persistent-layout and Triton branches, which
# exist only on an accelerator. A module-level `distributed_cpu` would put the
# accelerator test in a lane that cannot run it, and a module-level `gpu` would
# take the portable one out of the lane that can.
pytestmark = pytest.mark.distributed


def _run_two_ranks(*script_args: str, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/statevector_reverse_executor.py"),
            *script_args,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=dict(os.environ),
    )


@pytest.mark.distributed_cpu
def test_two_rank_gloo_backward_matches_dense_autograd():
    completed = _run_two_ranks("--backend", "gloo", timeout=90)
    assert (
        completed.stdout.count(
            '"backward_distribution_semantics": "sharded_across_ranks"'
        )
        == 2
    )
    assert completed.stdout.count('"backward_uses_full_state_replay": false') == 2


@pytest.mark.distributed_accel
@pytest.mark.gpu
def test_two_rank_nccl_persistent_layout_backward_matches_dense_autograd():
    """The one run that reaches the reversible-adjoint and fused-Triton branches.

    Every other lane leaves that half of `_explicit_sharded_adjoint` unexecuted:
    `_triton_vjp_adjoint_decision` defaults to `disabled_by_policy`, so the
    accelerated branches are unreachable unless a caller sets both the
    persistent-layout flags and a Triton-capable device, and no lane did. The
    harness asserts the accelerated path produced the gradients; this test
    asserts the harness ran, on both ranks.
    """

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    completed = _run_two_ranks("--backend", "nccl", "--persistent-layout", timeout=900)
    assert (
        completed.stdout.count(
            '"backward_distribution_semantics": "sharded_across_ranks"'
        )
        == 2
    )
    assert completed.stdout.count('"persistent_layout_enabled": true') == 2
