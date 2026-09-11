#!/usr/bin/env python
"""Capture reproducible local accelerator-run provenance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _output(command: Sequence[str]) -> str:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--log", type=Path, action="append", default=[])
    parser.add_argument("--skip-reason", default="")
    parser.add_argument("--world-size", type=int)
    parser.add_argument(
        "--evidence-scope",
        choices=(
            "one_gpu_local",
            "two_gpu_semantic_regression",
            "scheduled_4_8_gpu_scale",
        ),
        required=True,
    )
    args = parser.parse_args(argv)

    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": (
                ".".join(str(item) for item in torch.cuda.nccl.version())
                if torch.cuda.is_available()
                else "unavailable"
            ),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        }
    except (ImportError, RuntimeError) as exc:
        torch_info = {"unavailable": str(exc)}

    try:
        jax_version = importlib.metadata.version("jax")
    except importlib.metadata.PackageNotFoundError:
        jax_version = "not_installed"

    logs = [
        {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in args.log
        if path.is_file()
    ]
    payload = {
        "schema": "flagquantum_hardware_run_v1",
        "artifact_class": "development_run",
        "evidence_scope": args.evidence_scope,
        "commit": _output(("git", "rev-parse", "HEAD")),
        "command": args.command,
        "skip_reason": args.skip_reason or None,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch_info,
        "jax": jax_version,
        "rank_placement": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "RANK", "LOCAL_RANK", "WORLD_SIZE")
        },
        "gpus": _output(
            (
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            )
        ),
        "topology": _output(("nvidia-smi", "topo", "-m")),
        "logs": logs,
    }
    visible_devices = tuple(
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    )
    payload["rank_placement"]["expected_world_size"] = args.world_size
    payload["rank_placement"]["visible_device_mapping"] = (
        [
            {"local_rank": rank, "visible_device": device}
            for rank, device in enumerate(visible_devices[: args.world_size])
        ]
        if args.world_size
        else []
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
