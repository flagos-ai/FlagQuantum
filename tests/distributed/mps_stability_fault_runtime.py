from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq
import flagquantum.experimental.distributed as fqxd
from flagquantum.runtime.executors.mps.errors import MPSTrainingError


def circuit(theta):
    return fq.Circuit(4).ry(0, theta).rxx(0, 1, theta).rzz(1, 2, theta).ry(3, theta)


def committed_rank_path(root: Path, rank: int) -> Path:
    manifest = json.loads((root / "COMMITTED.json").read_text("utf-8"))
    shard = next(item for item in manifest["shards"] if item["rank"] == rank)
    return root / shard["file"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "topology",
            "generation",
            "manifest_missing",
            "manifest_corrupt",
            "committed_corruption",
            "unshared_checkpoint_root",
            "accidental_overwrite",
            "active_writer_lease",
            "premature_lease_break",
            "stale_lease_recovery",
            "interrupted",
            "rank_exception",
            "cuda_oom",
            "collective_timeout",
        ),
    )
    args = parser.parse_args()
    # Only the timeout fault needs a deliberately short collective deadline.
    # Checkpoint/lease tests must tolerate normal shared-runner scheduling delays.
    collective_timeout = 3 if args.mode == "collective_timeout" else 30
    dist.init_process_group(
        "gloo", timeout=datetime.timedelta(seconds=collective_timeout)
    )
    rank = dist.get_rank()
    root = Path(os.environ.get("FQ_TEST_CHECKPOINT", tempfile.mkdtemp()))
    theta = torch.tensor(0.2, requires_grad=True)
    active_root = (
        root / f"rank-{rank}-local" if args.mode == "unshared_checkpoint_root" else root
    )
    try:
        fqxd.train_distributed_mps(
            circuit(theta),
            steps=1,
            checkpoint_dir=active_root,
            initial_bond_dimension=2,
        )
    except MPSTrainingError as exc:
        if args.mode == "unshared_checkpoint_root":
            print(f"expected_failure:{args.mode}:{exc}", flush=True)
            dist.destroy_process_group()
            return
        raise
    dist.barrier()
    if args.mode == "accidental_overwrite":
        fresh_theta = torch.tensor(0.2, requires_grad=True)
        try:
            fqxd.train_distributed_mps(
                circuit(fresh_theta),
                steps=1,
                checkpoint_dir=root,
                initial_bond_dimension=2,
            )
        except MPSTrainingError as exc:
            print(f"expected_failure:{args.mode}:{exc}", flush=True)
            dist.destroy_process_group()
            return
        raise AssertionError("accidental checkpoint overwrite did not fail")
    if args.mode == "rank_exception" and rank == 0:
        raise RuntimeError("injected rank exception")
    if args.mode == "cuda_oom" and rank == 0:
        raise torch.OutOfMemoryError("injected CUDA OOM")
    if args.mode == "collective_timeout":
        if rank == 0:
            time.sleep(5)
        dist.barrier()
    if args.mode == "generation" and rank == 0:
        path = committed_rank_path(root, rank)
        payload = torch.load(path, weights_only=False)
        payload["completed_steps"] = 0
        torch.save(payload, path)
        path.with_suffix(".pt.sha256").write_text(
            hashlib.sha256(path.read_bytes()).hexdigest() + "\n",
            encoding="ascii",
        )
    if args.mode == "interrupted" and rank == 0:
        path = committed_rank_path(root, rank)
        path.replace(path.with_suffix(".pt.tmp"))
    if args.mode == "manifest_missing" and rank == 0:
        (root / "COMMITTED.json").unlink()
    if args.mode == "manifest_corrupt" and rank == 0:
        (root / "COMMITTED.json").write_text("{not-json", encoding="utf-8")
    if args.mode == "committed_corruption" and rank == 0:
        path = committed_rank_path(root, rank)
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0xFF
        path.write_bytes(data)
    if args.mode in {"active_writer_lease", "premature_lease_break"} and rank == 0:
        (root / "ACTIVE_WRITER.json").write_text(
            json.dumps(
                {
                    "schema": "sharded_mps_checkpoint_writer_lease_v1",
                    "hostname": __import__("socket").gethostname(),
                    "pid": os.getpid(),
                    "created_unix_seconds": time.time(),
                }
            )
            + "\n",
            encoding="utf-8",
        )
    if args.mode == "stale_lease_recovery" and rank == 0:
        (root / "ACTIVE_WRITER.json").write_text(
            json.dumps(
                {
                    "schema": "sharded_mps_checkpoint_writer_lease_v1",
                    "hostname": "retired-checkpoint-host",
                    "pid": 999999,
                    "created_unix_seconds": 1.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    dist.barrier()
    resumed_theta = torch.tensor(0.2, requires_grad=True)
    try:
        fqxd.train_distributed_mps(
            circuit(resumed_theta),
            steps=2,
            checkpoint_dir=root,
            resume=True,
            allow_checkpoint_writer_lease_break=(
                args.mode in {"premature_lease_break", "stale_lease_recovery"}
            ),
            checkpoint_writer_lease_stale_seconds=(
                1.0 if args.mode == "stale_lease_recovery" else 3600.0
            ),
            initial_bond_dimension=3 if args.mode == "topology" else 2,
        )
    except MPSTrainingError as exc:
        if args.mode in {
            "topology",
            "generation",
            "manifest_missing",
            "manifest_corrupt",
            "committed_corruption",
            "unshared_checkpoint_root",
            "accidental_overwrite",
            "active_writer_lease",
            "premature_lease_break",
        }:
            print(f"expected_failure:{args.mode}:{exc}", flush=True)
            dist.destroy_process_group()
            return
        raise
    if args.mode == "stale_lease_recovery":
        print(f"expected_success:{args.mode}", flush=True)
        dist.destroy_process_group()
        return
    raise AssertionError(f"fault mode {args.mode} did not fail")


if __name__ == "__main__":
    main()
