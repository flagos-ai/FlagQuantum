"""Atomically seal MPS capacity logs and telemetry after collection stops."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from flagquantum.testing.mps_capacity_certification import (
    finalize_capacity_source_integrity,
    require_capacity_source_integrity,
    require_general_mps_capacity,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-log-destination", type=Path)
    parser.add_argument("--gpu-samples-destination", type=Path)
    parser.add_argument("--single-gpu-destination", type=Path)
    parser.add_argument("--cuda-allocator-policy")
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    payload = json.loads(artifact.read_text())
    for source in payload.get("source_artifacts", ()):
        destination = (
            args.raw_log_destination
            if source.get("kind") == "raw_log"
            else args.gpu_samples_destination
            if source.get("kind") == "gpu_samples"
            else args.single_gpu_destination
            if source.get("kind") == "single_gpu_failure"
            else None
        )
        if destination is not None:
            destination = destination.resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(str(source["path"])).resolve(), destination)
            source["path"] = str(destination.relative_to(REPO_ROOT))
        if source.get("kind") not in {
            "raw_log",
            "gpu_samples",
            "single_gpu_failure",
        }:
            continue
        path = Path(str(source["path"]))
        resolved = path if path.is_absolute() else REPO_ROOT / path
        source["sha256"] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if args.cuda_allocator_policy:
        payload["cuda_allocator_policy"] = args.cuda_allocator_policy
    finalized = finalize_capacity_source_integrity(payload, base_dir=REPO_ROOT)
    require_general_mps_capacity(finalized)
    require_capacity_source_integrity(finalized, base_dir=REPO_ROOT)
    output = (args.output or artifact).resolve()
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(finalized, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"artifact": str(output), "source_integrity": "finalized"}))


if __name__ == "__main__":
    main()
