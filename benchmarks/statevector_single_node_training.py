"""Measured single-node capacity workload for sharded statevector training.

Run with, for example:
  torchrun --standalone --nproc-per-node=2 \
    benchmarks/statevector_single_node_training.py --n-wires 31 --steps 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist

import flagquantum as fq


def _workload(n_wires: int, device: torch.device) -> tuple[fq.Circuit, list[torch.Tensor]]:
    theta = torch.tensor(0.23, device=device, requires_grad=True)
    phi = torch.tensor(-0.37, device=device, requires_grad=True)
    circuit = fq.Circuit(n_wires, device=device)
    circuit.ry(n_wires - 1, theta)
    circuit.cx(n_wires - 1, n_wires - 2).ry(n_wires - 2, phi)
    return circuit, [theta, phi]


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned-worktree"


def _classify_capacity_outcome(*, observed_oom: bool, expect_oom: bool) -> dict[str, object]:
    """Return an explicit, fail-closed classification for a capacity probe."""
    expectation = "cuda_oom" if expect_oom else "completion"
    expectation_met = observed_oom == expect_oom
    if observed_oom:
        status = "expected_oom" if expect_oom else "unexpected_oom"
        classification = "measured_single_device_capacity_failure"
    else:
        status = "unexpected_completion" if expect_oom else "passed"
        classification = "measured_completion"
    return {
        "status": status,
        "expectation": expectation,
        "expectation_met": expectation_met,
        "capacity_classification": classification,
        "single_device_oom_observed": observed_oom,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, default=31)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="adam")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-oom", action="store_true")
    parser.add_argument(
        "--workload-manifest",
        type=Path,
        help="Immutable workload JSON used for a release-eligible rerun.",
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--crash-after-completion-rank", type=int)
    args = parser.parse_args()
    if args.resume and args.checkpoint_dir is None:
        parser.error("--resume requires --checkpoint-dir")

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    workload = {
        "name": "issue044_single_node_capacity_v1",
        "n_wires": args.n_wires,
        "steps": args.steps,
        "optimizer": args.optimizer,
        "dtype": "complex64",
        "gates": ["RY(n-1,theta)", "CX(n-1,n-2)", "RY(n-2,phi)"],
        "observable": "Z(n-2)",
    }
    if args.workload_manifest is not None:
        frozen_workload = json.loads(args.workload_manifest.read_text(encoding="utf-8"))
        if not isinstance(frozen_workload, dict):
            parser.error("--workload-manifest must contain a JSON object")
        requested = {
            "n_wires": args.n_wires,
            "steps": args.steps,
            "optimizer": args.optimizer,
        }
        mismatches = {
            key: (requested[key], frozen_workload.get(key))
            for key in requested
            if requested[key] != frozen_workload.get(key)
        }
        if mismatches:
            parser.error(f"arguments do not match frozen workload: {mismatches}")
        workload = frozen_workload
        workload_sha256 = hashlib.sha256(
            args.workload_manifest.read_bytes()
        ).hexdigest()
    else:
        workload_sha256 = hashlib.sha256(
            json.dumps(workload, sort_keys=True).encode()
        ).hexdigest()
    started = time.perf_counter()
    record: dict[str, object]
    pending_error: BaseException | None = None
    try:
        circuit, parameters = _workload(args.n_wires, device)
        result = fq.train_distributed_statevector(
            circuit,
            steps=args.steps,
            observable_wire=args.n_wires - 2,
            optimizer=args.optimizer,
            checkpoint_dir=args.checkpoint_dir,
            resume=args.resume,
            timeout_seconds=3600.0,
        )
        torch.cuda.synchronize(device)
        if args.crash_after_completion_rank == rank:
            os._exit(137)
        record = {
            **_classify_capacity_outcome(observed_oom=False, expect_oom=args.expect_oom),
            "training": result.summary(),
            "parameters": [float(item.detach().cpu()) for item in parameters],
        }
        if args.expect_oom:
            pending_error = RuntimeError("workload unexpectedly fit on one GPU")
    except torch.cuda.OutOfMemoryError as error:
        record = {
            **_classify_capacity_outcome(observed_oom=True, expect_oom=args.expect_oom),
            "oom_type": type(error).__name__,
            "oom_message": str(error),
        }
        if not args.expect_oom:
            pending_error = error
    finally:
        elapsed = time.perf_counter() - started
        properties = torch.cuda.get_device_properties(device)
        record = {
            **locals().get("record", {"status": "failed_before_record"}),
            "schema": "flagquantum.single_node_training_acceptance.v2",
            "measured": True,
            "release_evidence": False,
            "artifact_classification": "development_capacity_probe",
            "capacity_claim_allowed": False,
            "multi_node_claim": False,
            "rank": rank,
            "world_size": world,
            "local_rank": local_rank,
            "hostname": platform.node(),
            "commit": _commit(),
            "workload": workload,
            "workload_sha256": workload_sha256,
            "resume_requested": args.resume,
            "checkpoint_enabled": args.checkpoint_dir is not None,
            "elapsed_seconds": elapsed,
            "device": {
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "capability": list(properties.major_minor) if hasattr(properties, "major_minor") else [properties.major, properties.minor],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            },
            "software": {"torch": torch.__version__, "cuda": torch.version.cuda},
        }
        gathered: list[object] | None = [None] * world if rank == 0 else None
        if world > 1:
            dist.gather_object(record, gathered, dst=0)
        else:
            gathered = [record]
        if rank == 0:
            document = {"ranks": gathered, "workload_sha256": workload_sha256}
            payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
            if args.output is None:
                print(payload, end="", flush=True)
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(payload, encoding="utf-8")
                print(args.output, flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
    if pending_error is not None:
        raise pending_error


if __name__ == "__main__":
    main()
