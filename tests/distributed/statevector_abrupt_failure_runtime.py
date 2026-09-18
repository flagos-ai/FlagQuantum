"""Real process-loss runtime for statevector elastic-launcher acceptance."""

from __future__ import annotations

import argparse
import json
import os
import signal
from datetime import timedelta

import torch
import torch.distributed as dist

# Two deadlines, because the two barriers below answer different questions.
#
# The first barrier only has to outlast peer startup skew, and that skew is set
# by the host's scheduler rather than by this process: the two ranks finish
# importing within 0.03s of each other and the whole group setup takes about
# 0.04s on an idle host, but a loaded host stretches both without bound. A 5s
# deadline here fails the test once the skew passes 5s, which is reachable; 20s
# tolerates four times the skew that was measured to break the old value and
# still leaves the rest of the run inside the test's own 30s bound.
#
# The second barrier is entered after the failing rank has already been killed,
# and its deadline is what bounds the test: the survivor waits, times out, and
# reports the loss. That one stays tight.
STARTUP_TIMEOUT_SECONDS = 20.0
PEER_LOSS_TIMEOUT_SECONDS = 5.0


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
    init_options = {"timeout": timedelta(seconds=STARTUP_TIMEOUT_SECONDS)}
    if args.backend == "nccl":
        init_options["device_id"] = device
    dist.init_process_group(args.backend, **init_options)
    control_group = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=STARTUP_TIMEOUT_SECONDS)
    )
    try:
        dist.monitored_barrier(
            group=control_group,
            timeout=timedelta(seconds=STARTUP_TIMEOUT_SECONDS),
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
                timeout=timedelta(seconds=PEER_LOSS_TIMEOUT_SECONDS),
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
