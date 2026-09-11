#!/usr/bin/env python
"""Seal executor-emitted measurements into a release-verifiable envelope."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from importlib import import_module
from pathlib import Path
from typing import Sequence

from flagquantum.runtime.observability.evidence import (
    ArtifactClass,
    EvidenceScope,
    RuntimeProvenance,
    create_evidence_artifact,
    sha256_file,
)

environment_errors = import_module(
    (
        "tools.check_release_evidence_environment"
        if __package__
        else "check_release_evidence_environment"
    )
).environment_errors


def _output(command: Sequence[str]) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--raw-log", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=tuple(item.value for item in EvidenceScope), required=True
    )
    parser.add_argument("--command", action="append", required=True)
    parser.add_argument("--collective-backend", required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    args = parser.parse_args(argv)

    preflight_errors = environment_errors(world_size=args.world_size)
    if preflight_errors:
        raise SystemExit(
            "release evidence preflight failed: " + "; ".join(preflight_errors)
        )
    signing_key = os.environ.get("FQ_EVIDENCE_SIGNING_KEY", "").encode()
    if not signing_key:
        raise SystemExit("FQ_EVIDENCE_SIGNING_KEY is required")
    evidence = json.loads(args.measurements.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise SystemExit("measurements must be a JSON object emitted by the executor")
    detected_devices = tuple(
        line.strip()
        for line in _output(
            (
                "nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader",
            )
        ).splitlines()
        if line.strip()
    )
    devices = detected_devices[: args.world_size]
    if len(devices) != args.world_size:
        raise SystemExit(
            f"requested world size {args.world_size}, detected {len(detected_devices)} GPUs"
        )
    provenance = RuntimeProvenance(
        commit=_output(("git", "rev-parse", "HEAD")),
        workload_sha256=sha256_file(args.workload),
        command=tuple(args.command),
        devices=devices,
        topology=_output(("nvidia-smi", "topo", "-m")),
        rank_mapping=tuple(
            f"local_rank={rank}:device_uuid={device}"
            for rank, device in enumerate(devices)
        ),
        collective_backend=args.collective_backend,
        warmup=args.warmup,
        iterations=args.iterations,
        seeds=tuple(args.seed),
        raw_log_sha256=sha256_file(args.raw_log),
        fallback_events=tuple(
            str(item) for item in evidence.get("fallback_events", ())
        ),
    )
    artifact = create_evidence_artifact(
        artifact_class=ArtifactClass.MEASURED_PRODUCTION_RUN,
        evidence_scope=EvidenceScope(args.scope),
        provenance=provenance,
        evidence=evidence,
        signing_key=signing_key,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact.summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
