"""Torchrun multi-step, recovery and fault acceptance for sharded statevector training."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq
import flagquantum.experimental.distributed as fqxd
from flagquantum.runtime.executors.statevector.training import (
    DistributedTrainingError,
    train_distributed_statevector,
)

# `--mode timeout` stalls a control barrier and reports the resulting timeout, so
# it needs a deadline short enough to keep the case quick. Every other mode must
# not inherit that deadline. `timeout_seconds` is one value inside the library,
# and the library derives three separate things from it: the progress-stall
# watchdog, the per-operation deadline around backward, collective, and
# optimizer steps, and the whole-run budget. Handing the deliberate value to
# modes that are not testing a timeout puts a two-second deadline on ordinary
# work, and a loaded host then reports an unintended timeout at the wrong
# operation instead of the failure under test.
DELIBERATE_TIMEOUT_SECONDS = 2.0
ORDINARY_TIMEOUT_SECONDS = 60.0
# The process group's own watchdog. Ordinary work here is sub-second, so this
# only has to outlast a stalled machine rather than a slow operation, and it
# matches the value the MPS fault runtimes already use.
COLLECTIVE_TIMEOUT_SECONDS = 30.0
# Modes that never return on their own: one rank leaves or stalls and the rank
# that remains waits for the elastic launcher to terminate the job, raising an
# assertion about the launcher if this many seconds pass first. That assertion
# is the boundedness claim, so this is how long it needs before it can speak.
LAUNCHER_KILL_WAIT_SECONDS = 60.0
# The library's own default for a training run, read rather than restated so the
# budget below cannot drift away from it.
LIBRARY_DEFAULT_TIMEOUT_SECONDS = float(
    inspect.signature(train_distributed_statevector)
    .parameters["timeout_seconds"]
    .default
)
# The longest each mode may legitimately take, in the terms the mode's own body
# is written in: a training call's budget, plus the launcher wait for the modes
# that end by leaving a rank for the elastic agent to terminate. Listed one mode
# at a time so that adding a mode without deciding its budget fails loudly below
# rather than inheriting a silent guess.
_MODE_INNER_BUDGET_SECONDS = {
    "train": ORDINARY_TIMEOUT_SECONDS,
    "fault": ORDINARY_TIMEOUT_SECONDS,
    "timeout": DELIBERATE_TIMEOUT_SECONDS,
    "crash": LAUNCHER_KILL_WAIT_SECONDS,
    "crash_after_checkpoint": (
        LIBRARY_DEFAULT_TIMEOUT_SECONDS + LAUNCHER_KILL_WAIT_SECONDS
    ),
    "resume": 2 * LIBRARY_DEFAULT_TIMEOUT_SECONDS,
}


def mode_inner_budget_seconds(mode: str) -> float:
    """Return the longest a single ``--mode <mode>`` run may consume.

    A wrapper deadline below this replaces whatever the run asserts from the
    inside -- a launcher that failed to terminate a lost job, an operation that
    outran its budget -- with the wrapper's own timeout, which names neither.
    """

    return _MODE_INNER_BUDGET_SECONDS[mode]


def build(device: torch.device):
    theta = torch.tensor(0.43, device=device, requires_grad=True)
    phi = torch.tensor(-0.21, device=device, requires_grad=True)
    circuit = fq.Circuit(3, device=device).ry(0, theta).rxx(0, 2, phi).rz(1, theta)
    return circuit, (theta, phi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="gloo")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=(
            "train",
            "fault",
            "timeout",
            "crash",
            "crash_after_checkpoint",
            "resume",
        ),
        default="train",
    )
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("--steps must be at least 2 for checkpoint/resume acceptance")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cpu")
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    if world > 1:
        init_kwargs = {
            "backend": args.backend,
            "timeout": timedelta(seconds=COLLECTIVE_TIMEOUT_SECONDS),
        }
        if args.backend == "nccl":
            init_kwargs["device_id"] = device
        dist.init_process_group(**init_kwargs)
    baseline_memory_bytes = (
        int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
    )
    payload = {}
    try:
        if args.mode == "crash":
            if rank == 1:
                # Model abrupt worker loss; torchrun/elastic owns peer cleanup.
                os._exit(137)
            time.sleep(LAUNCHER_KILL_WAIT_SECONDS)
            raise AssertionError("elastic launcher did not terminate surviving rank")
        if args.mode == "crash_after_checkpoint":
            partial_circuit, _ = build(device)
            partial = fqxd.train_distributed_statevector(
                partial_circuit,
                steps=args.steps // 2,
                optimizer="adam",
                lr=0.02,
                checkpoint_dir=args.checkpoint_dir,
            )
            assert partial.completed_steps == args.steps // 2
            if rank == 1:
                os._exit(137)
            time.sleep(LAUNCHER_KILL_WAIT_SECONDS)
            raise AssertionError("watchdog did not terminate surviving rank")
        if args.mode == "resume":
            uninterrupted, uninterrupted_parameters = build(device)
            full = fqxd.train_distributed_statevector(
                uninterrupted, steps=args.steps, optimizer="adam", lr=0.02
            )
            resumed_circuit, resumed_parameters = build(device)
            resumed = fqxd.train_distributed_statevector(
                resumed_circuit,
                steps=args.steps,
                optimizer="adam",
                lr=0.02,
                checkpoint_dir=args.checkpoint_dir,
                resume=True,
                rematerialization_interval=2,
            )
            for actual, expected in zip(
                resumed_parameters, uninterrupted_parameters, strict=True
            ):
                torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(
                torch.tensor(resumed.losses),
                torch.tensor(full.losses[args.steps // 2 :]),
                atol=3e-5,
                rtol=3e-5,
            )
            payload = {
                "summary": resumed.summary(),
                "resumed_parameters": [
                    float(parameter.detach().cpu()) for parameter in resumed_parameters
                ],
                "reference_parameters": [
                    float(parameter.detach().cpu())
                    for parameter in uninterrupted_parameters
                ],
                "reference_tail_losses": list(full.losses[args.steps // 2 :]),
            }
        elif args.mode != "train":
            circuit, _ = build(device)
            try:
                fqxd.train_distributed_statevector(
                    circuit,
                    steps=2,
                    fault_rank=1 if args.mode == "fault" else None,
                    inject_collective_timeout=args.mode == "timeout",
                    timeout_seconds=(
                        DELIBERATE_TIMEOUT_SECONDS
                        if args.mode == "timeout"
                        else ORDINARY_TIMEOUT_SECONDS
                    ),
                )
            except DistributedTrainingError as exc:
                payload = {"error": exc.to_dict()}
            else:
                raise AssertionError("injected failure did not fail closed")
        else:
            uninterrupted, uninterrupted_parameters = build(device)
            full = fqxd.train_distributed_statevector(
                uninterrupted, steps=args.steps, optimizer="adam", lr=0.02
            )
            partial_circuit, _ = build(device)
            partial = fqxd.train_distributed_statevector(
                partial_circuit,
                steps=args.steps // 2,
                optimizer="adam",
                lr=0.02,
                checkpoint_dir=args.checkpoint_dir,
            )
            resumed_circuit, resumed_parameters = build(device)
            resumed = fqxd.train_distributed_statevector(
                resumed_circuit,
                steps=args.steps,
                optimizer="adam",
                lr=0.02,
                checkpoint_dir=args.checkpoint_dir,
                resume=True,
                rematerialization_interval=2,
            )
            for actual, expected in zip(
                resumed_parameters, uninterrupted_parameters, strict=True
            ):
                torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(
                torch.tensor(resumed.losses),
                torch.tensor(full.losses[args.steps // 2 :]),
                atol=3e-5,
                rtol=3e-5,
            )
            checkpoint = torch.load(
                args.checkpoint_dir / f"rank-{rank}.pt",
                map_location=device,
                weights_only=False,
            )
            owned = tuple(checkpoint["owned_indices"])
            optimizer_state = checkpoint["optimizer_state"]
            state_count = (
                0 if optimizer_state is None else len(optimizer_state["state"])
            )
            assert state_count == len(owned) < len(resumed_parameters)
            assert all(
                item.optimizer_state_local == (item.owner_rank == rank)
                for item in resumed.ownership
            )
            phases = {item.phase for item in resumed.progress}
            assert {
                "forward",
                "backward",
                "optimizer",
                "checkpoint",
                "teardown",
            } <= phases
            assert {
                "execution_segment",
                "gate_block",
                "collective",
                "backward_segment",
                "optimizer_step",
            } <= phases
            payload = {
                "summary": resumed.summary(),
                "partial_steps": partial.completed_steps,
                "owned_indices": owned,
                "optimizer_state_count": state_count,
            }
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    payload["cleanup_verified"] = not dist.is_initialized()
    if device.type == "cuda":
        if args.mode == "train":
            del uninterrupted, uninterrupted_parameters, full
            del partial_circuit, partial, resumed_circuit, resumed_parameters, resumed
            del checkpoint
        elif args.mode == "resume":
            del uninterrupted, uninterrupted_parameters, full
            del resumed_circuit, resumed_parameters, resumed
        else:
            del circuit
        gc.collect()
        torch.cuda.empty_cache()
        payload["post_cleanup_memory_bytes"] = int(torch.cuda.memory_allocated(device))
        payload["post_teardown_allocator_delta_bytes"] = max(
            0, payload["post_cleanup_memory_bytes"] - baseline_memory_bytes
        )
        payload["process_exit_releases_cuda_context"] = True
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
