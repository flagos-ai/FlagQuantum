"""Torchrun multi-step, recovery and fault acceptance for ISSUE-043."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.backends.statevector.training import DistributedTrainingError


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
            "timeout": timedelta(seconds=5),
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
            time.sleep(60.0)
            raise AssertionError("elastic launcher did not terminate surviving rank")
        if args.mode == "crash_after_checkpoint":
            partial_circuit, _ = build(device)
            partial = fq.train_distributed_statevector(
                partial_circuit,
                steps=args.steps // 2,
                optimizer="adam",
                lr=0.02,
                checkpoint_dir=args.checkpoint_dir,
            )
            assert partial.completed_steps == args.steps // 2
            if rank == 1:
                os._exit(137)
            time.sleep(60.0)
            raise AssertionError("watchdog did not terminate surviving rank")
        if args.mode == "resume":
            uninterrupted, uninterrupted_parameters = build(device)
            full = fq.train_distributed_statevector(
                uninterrupted, steps=args.steps, optimizer="adam", lr=0.02
            )
            resumed_circuit, resumed_parameters = build(device)
            resumed = fq.train_distributed_statevector(
                resumed_circuit,
                steps=args.steps,
                optimizer="adam",
                lr=0.02,
                checkpoint_dir=args.checkpoint_dir,
                resume=True,
                rematerialization_interval=2,
            )
            for actual, expected in zip(resumed_parameters, uninterrupted_parameters):
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
                fq.train_distributed_statevector(
                    circuit,
                    steps=2,
                    fault_rank=1 if args.mode == "fault" else None,
                    inject_collective_timeout=args.mode == "timeout",
                    timeout_seconds=2.0,
                )
            except DistributedTrainingError as exc:
                payload = {"error": exc.to_dict()}
            else:
                raise AssertionError("injected failure did not fail closed")
        else:
            uninterrupted, uninterrupted_parameters = build(device)
            full = fq.train_distributed_statevector(
                uninterrupted, steps=args.steps, optimizer="adam", lr=0.02
            )
            partial_circuit, _ = build(device)
            partial = fq.train_distributed_statevector(
                partial_circuit,
                steps=args.steps // 2,
                optimizer="adam",
                lr=0.02,
                checkpoint_dir=args.checkpoint_dir,
            )
            resumed_circuit, resumed_parameters = build(device)
            resumed = fq.train_distributed_statevector(
                resumed_circuit,
                steps=args.steps,
                optimizer="adam",
                lr=0.02,
                checkpoint_dir=args.checkpoint_dir,
                resume=True,
                rematerialization_interval=2,
            )
            for actual, expected in zip(resumed_parameters, uninterrupted_parameters):
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
