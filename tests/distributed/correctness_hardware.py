"""Torchrun correctness lane for 1/2/4-8 local GPUs."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; a skip is not correctness evidence")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.device_count() < world:
        raise SystemExit(
            f"required {world} visible GPUs, found {torch.cuda.device_count()}"
        )
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    try:
        shard = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
        reduced = shard.clone()
        if world > 1:
            dist.all_reduce(reduced)
        expected = world * (world + 1) / 2
        torch.testing.assert_close(
            reduced, torch.tensor([expected], device=reduced.device)
        )
        payload = {
            "world_size": world,
            "rank": rank,
            "device": str(reduced.device),
            "rank_owned_value": float(shard.item()),
            "collective_value": float(reduced.item()),
            "distribution_semantics": (
                "sharded_across_ranks" if world > 1 else "single_device_fast_path"
            ),
            "cleanup_verified": False,
        }
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()
    payload["cleanup_verified"] = not dist.is_initialized()
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
