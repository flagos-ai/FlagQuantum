from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.distributed_cpu
@pytest.mark.parametrize("world_size", (2, 3, 4))
def test_explicit_rank_owned_mps_reverse(world_size: int) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            str(ROOT / "tests/distributed/mps_reverse_runtime.py"),
        ],
        cwd=ROOT,
        check=True,
        timeout=180,
    )


@pytest.mark.gpu
@pytest.mark.distributed_accel
def test_explicit_rank_owned_mps_reverse_nccl() -> None:
    if torch.cuda.device_count() < 2 or not torch.distributed.is_nccl_available():
        pytest.skip("requires two CUDA devices and NCCL")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/mps_reverse_runtime.py"),
            "--backend",
            "nccl",
        ],
        cwd=ROOT,
        check=True,
        timeout=180,
    )


@pytest.mark.distributed_cpu
@pytest.mark.parametrize("mode", ("mismatch", "memory", "sequence"))
def test_mps_reverse_contracts_fail_closed_without_hanging(mode: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/mps_reverse_contract_runtime.py"),
            "--mode",
            mode,
        ],
        cwd=ROOT,
        check=True,
        timeout=60,
    )


@pytest.mark.distributed_cpu
def test_mps_reverse_collective_timeout_terminates_all_ranks() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(ROOT / "tests/distributed/mps_reverse_contract_runtime.py"),
            "--mode",
            "timeout",
        ],
        cwd=ROOT,
        check=False,
        timeout=30,
    )
    assert completed.returncode != 0
