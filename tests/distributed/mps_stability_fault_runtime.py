from __future__ import annotations

import argparse
import datetime
import os
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.backends.mps.training import MPSTrainingError


def circuit(theta):
    return fq.Circuit(4).ry(0, theta).rxx(0, 1, theta).rzz(1, 2, theta).ry(3, theta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "topology",
            "generation",
            "interrupted",
            "rank_exception",
            "cuda_oom",
            "collective_timeout",
        ),
    )
    args = parser.parse_args()
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=3))
    rank = dist.get_rank()
    root = Path(os.environ.get("FQ_TEST_CHECKPOINT", tempfile.mkdtemp()))
    theta = torch.tensor(0.2, requires_grad=True)
    fq.train_distributed_mps(
        circuit(theta),
        steps=1,
        checkpoint_dir=root,
        initial_bond_dimension=2,
    )
    dist.barrier()
    if args.mode == "rank_exception" and rank == 0:
        raise RuntimeError("injected rank exception")
    if args.mode == "cuda_oom" and rank == 0:
        raise torch.OutOfMemoryError("injected CUDA OOM")
    if args.mode == "collective_timeout":
        if rank == 0:
            time.sleep(5)
        dist.barrier()
    if args.mode == "generation" and rank == 0:
        path = root / "rank-0.pt"
        payload = torch.load(path, weights_only=False)
        payload["completed_steps"] = 0
        torch.save(payload, path)
    if args.mode == "interrupted" and rank == 0:
        path = root / "rank-0.pt"
        path.replace(path.with_suffix(".pt.tmp"))
    dist.barrier()
    resumed_theta = torch.tensor(0.2, requires_grad=True)
    try:
        fq.train_distributed_mps(
            circuit(resumed_theta),
            steps=2,
            checkpoint_dir=root,
            resume=True,
            initial_bond_dimension=3 if args.mode == "topology" else 2,
        )
    except MPSTrainingError as exc:
        if args.mode in {"topology", "generation"}:
            print(f"expected_failure:{args.mode}:{exc}", flush=True)
            dist.destroy_process_group()
            return
        raise
    raise AssertionError(f"fault mode {args.mode} did not fail")


if __name__ == "__main__":
    main()
