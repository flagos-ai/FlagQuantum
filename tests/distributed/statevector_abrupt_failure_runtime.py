"""Real process-loss runtime for statevector elastic-launcher acceptance."""

from __future__ import annotations

import argparse
import json
import os
import signal
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--failure-rank", type=int, default=1)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cpu")
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    init_options = {"timeout": timedelta(seconds=5)}
    if args.backend == "nccl":
        init_options["device_id"] = device
    dist.init_process_group(args.backend, **init_options)
    control_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=5))
    try:
        dist.monitored_barrier(
            group=control_group,
            timeout=timedelta(seconds=5),
            wait_all_ranks=True,
        )
        if rank == args.failure_rank:
            print(
                json.dumps(
                    {
                        "event": "injected_abrupt_process_loss",
                        "rank": rank,
                        "signal": "SIGKILL",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            os.kill(os.getpid(), signal.SIGKILL)
        try:
            dist.monitored_barrier(
                group=control_group,
                timeout=timedelta(seconds=5),
                wait_all_ranks=True,
            )
        except RuntimeError as error:
            print(
                json.dumps(
                    {
                        "event": "peer_process_loss_detected",
                        "rank": rank,
                        "phase": "control_plane",
                        "error_type": type(error).__name__,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            raise
        raise AssertionError("peer process loss was not detected")
    finally:
        dist.destroy_process_group(control_group)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
