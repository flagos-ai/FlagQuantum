#!/usr/bin/env python
"""Fail-closed preflight for the single-node MPS NCCL certification matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_WORLD_SIZES = (1, 2, 4, 8)


def readiness_errors(
    *,
    cuda_available: bool,
    nccl_available: bool,
    device_names: Sequence[str],
    source_tree_dirty: bool,
    signing_key_present: bool,
) -> tuple[str, ...]:
    """Return blockers for an auditable 1/2/4/8-GPU MPS certification run."""

    errors: list[str] = []
    if not cuda_available:
        errors.append("CUDA runtime is unavailable")
    if not nccl_available:
        errors.append("PyTorch NCCL backend is unavailable")
    if len(device_names) < max(REQUIRED_WORLD_SIZES):
        errors.append(
            "MPS certification requires 8 visible GPUs; "
            f"detected {len(device_names)}"
        )
    if device_names and len(set(device_names[:8])) != 1:
        errors.append("the first 8 visible GPUs must use one accelerator model")
    if source_tree_dirty:
        errors.append("MPS certification requires a clean source tree")
    if not signing_key_present:
        errors.append("FQ_EVIDENCE_SIGNING_KEY is not configured")
    return tuple(errors)


def certification_commands(
    *,
    output_root: str = "benchmarks/results/smoke/release_candidates/mps_single_node",
) -> tuple[str, ...]:
    """Return the frozen measured-training commands for every required world size."""

    runner = "benchmarks/internal/evidence/mps_training_acceptance.py"
    commands = []
    for world_size in REQUIRED_WORLD_SIZES:
        commands.append(
            " ".join(
                (
                    "python -m torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={world_size}",
                    runner,
                    f"--output {output_root}/{world_size}gpu-adam.json",
                    "--family variable_bond_training",
                    "--scaling-mode strong",
                    "--n-wires 16",
                    "--layers 4",
                    "--max-bond 64",
                    "--warmup 5",
                    "--repetitions 20",
                    "--gradient-policy exact",
                )
            )
        )
    return tuple(commands)


def build_payload(
    *,
    cuda_available: bool,
    nccl_available: bool,
    device_names: Sequence[str],
    source_tree_dirty: bool,
    signing_key_present: bool,
    source_commit: str,
) -> dict[str, Any]:
    blockers = readiness_errors(
        cuda_available=cuda_available,
        nccl_available=nccl_available,
        device_names=device_names,
        source_tree_dirty=source_tree_dirty,
        signing_key_present=signing_key_present,
    )
    return {
        "schema": "flagquantum.mps_single_node_certification_preflight.v1",
        "benchmark": "mps_single_node_certification_preflight",
        "source_commit": source_commit,
        "required_world_sizes": REQUIRED_WORLD_SIZES,
        "required_gpu_count": 8,
        "detected_gpu_count": len(device_names),
        "device_names": tuple(device_names),
        "cuda_available": cuda_available,
        "nccl_available": nccl_available,
        "source_tree_dirty": source_tree_dirty,
        "signing_key_present": signing_key_present,
        "distribution_semantics": "preflight_only",
        "claim_evidence_type": "plan_preflight",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "commands": certification_commands(),
        "blockers": blockers,
        "ready": not blockers,
    }


def _command_output(command: Sequence[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def environment_payload() -> Mapping[str, Any]:
    import torch

    device_names = tuple(
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    )
    return build_payload(
        cuda_available=torch.cuda.is_available(),
        nccl_available=torch.distributed.is_nccl_available(),
        device_names=device_names,
        source_tree_dirty=bool(
            _command_output(("git", "status", "--porcelain=v1"))
        ),
        signing_key_present=bool(os.environ.get("FQ_EVIDENCE_SIGNING_KEY")),
        source_commit=_command_output(("git", "rev-parse", "HEAD")) or "unavailable",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    payload = environment_payload()
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
