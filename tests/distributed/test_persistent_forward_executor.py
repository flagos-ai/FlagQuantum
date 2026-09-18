"""CI-seeded two-rank persistent-layout forward execution.

The markers sit on the tests, not the module, for the same reason as
`test_reverse_executor.py`: the gloo run needs no device and carries the swap
replay, which is device-agnostic, while the nccl run is the only one that can
reach the Triton-accelerated dispatch paths.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.distributed


def _run_two_ranks(*script_args: str, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/statevector_persistent_forward_executor.py"),
            *script_args,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=dict(os.environ),
    )


def _rank_summaries(
    completed: subprocess.CompletedProcess[str],
) -> list[dict[str, Any]]:
    """One parsed summary per rank, so both are checked and not just counted.

    Each rank writes its own line, but `print` emits the text and the newline in
    two writes, so two ranks can interleave as `{...}{...}` with nothing between
    them. The decoder is driven over the stream rather than over lines.
    """

    decoder = json.JSONDecoder()
    summaries: list[dict[str, Any]] = []
    index = completed.stdout.find("{")
    while index != -1:
        summary, end = decoder.raw_decode(completed.stdout, index)
        summaries.append(summary)
        index = completed.stdout.find("{", end)
    return summaries


@pytest.mark.distributed_cpu
def test_two_rank_gloo_persistent_layout_matches_canonical():
    """Replays layout swaps on a lane that needs no accelerator.

    `persistent_wire_layout` is the only way into the swap replay, and nothing
    passed it to this executor on a CPU lane. The harness asserts the plan is
    non-empty before it runs, so the branch is reached rather than merely
    available.
    """

    completed = _run_two_ranks("--backend", "gloo", timeout=120)
    summaries = _rank_summaries(completed)
    assert len(summaries) == 2
    for summary in summaries:
        assert summary["wire_layout"] == "persistent"
        assert summary["persistent_swap_count"] > 0
        assert summary["cleanup_verified"] is True
        assert (
            summary["persistent_communication_bytes"]
            < summary["canonical_communication_bytes"]
        )


@pytest.mark.distributed_accel
@pytest.mark.gpu
def test_two_rank_nccl_persistent_layout_reaches_triton_paths():
    """The one run that reaches the fused swap, CX-segment and block-fusion paths."""

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    completed = _run_two_ranks("--backend", "nccl", timeout=600)
    summaries = _rank_summaries(completed)
    assert len(summaries) == 2
    for summary in summaries:
        assert summary["wire_layout"] == "persistent"
        assert summary["persistent_swap_count"] > 0
        # The point of this test: the accelerated dispatch paths ran, not just
        # that the state came out right.
        assert summary["triton_execution_count"] > 0
