#!/usr/bin/env python
"""Seal executor-emitted measurements into a release-verifiable envelope."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path

from flagquantum.runtime.observability.evidence import (
    ArtifactClass,
    EvidenceScope,
    RuntimeProvenance,
    create_evidence_artifact,
    sha256_file,
)

environment_errors = import_module(
    "tools.check_release_evidence_environment"
    if __package__
    else "check_release_evidence_environment"
).environment_errors


def _output(command: Sequence[str]) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()


def _rank_devices(
    declared: Sequence[str] | None, *, world_size: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the device per rank and the rank-to-device mapping.

    A declared entry may name the host that ran the rank as ``UUID@NODE``. The
    node is recorded in the mapping rather than in the device list, because a
    device inventory that mixed UUIDs with hostnames could not be checked
    against ``nvidia-smi`` afterwards.
    """

    entries = list(declared or ())
    if not entries:
        detected = tuple(
            line.strip()
            for line in _output(
                ("nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader")
            ).splitlines()
            if line.strip()
        )
        if len(detected) < world_size:
            raise SystemExit(
                f"requested world size {world_size}, detected {len(detected)} GPUs"
            )
        entries = list(detected[:world_size])
    if len(entries) != world_size:
        raise SystemExit(
            f"requested world size {world_size}, received {len(entries)} device UUIDs"
        )
    devices: list[str] = []
    mapping: list[str] = []
    for rank, entry in enumerate(entries):
        uuid, _, node = entry.partition("@")
        devices.append(uuid.strip())
        scope = f"node={node.strip()}:" if node.strip() else ""
        mapping.append(f"rank={rank}:{scope}device_uuid={uuid.strip()}")
    return tuple(devices), tuple(mapping)


def _recorded_provenance(path: Path) -> RuntimeProvenance:
    """Return the provenance of a previously measured run, unchanged.

    The provenance states which run produced the measurements. Detecting a fresh
    one here would restamp a re-sealed payload with the commit, devices, topology
    and raw-log digest of the checkout doing the re-sealing, which is a different
    run and possibly a different revision. So a re-seal copies the whole record
    instead of completing one, and a record that is missing a field fails here
    rather than being filled in from outside.
    """

    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"{path} is not a JSON object")
    recorded = document.get("provenance", document)
    if not isinstance(recorded, dict):
        raise SystemExit(f"{path} carries no provenance block to reuse")
    try:
        return RuntimeProvenance(
            commit=str(recorded["commit"]),
            workload_sha256=str(recorded["workload_sha256"]),
            command=tuple(str(item) for item in recorded["command"]),
            devices=tuple(str(item) for item in recorded["devices"]),
            topology=str(recorded["topology"]),
            rank_mapping=tuple(str(item) for item in recorded["rank_mapping"]),
            collective_backend=str(recorded["collective_backend"]),
            warmup=int(recorded["warmup"]),
            iterations=int(recorded["iterations"]),
            seeds=tuple(int(item) for item in recorded["seeds"]),
            raw_log_sha256=str(recorded["raw_log_sha256"]),
            fallback_events=tuple(str(item) for item in recorded["fallback_events"]),
        )
    except KeyError as error:
        raise SystemExit(
            f"the recorded provenance in {path} is missing {error.args[0]!r}; a "
            "re-seal reuses a whole provenance record rather than completing one "
            "from the state of this checkout"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--raw-log", type=Path)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=tuple(item.value for item in EvidenceScope), required=True
    )
    parser.add_argument("--command", action="append")
    parser.add_argument("--collective-backend")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument(
        "--recorded-provenance",
        type=Path,
        default=None,
        help=(
            "reuse the provenance of the run that produced these measurements, "
            "read from a sealed envelope or a bare provenance object, instead of "
            "detecting one here. Re-sealing an already measured payload must not "
            "restamp its commit, devices, topology or raw-log digest with the "
            "state of the checkout doing the re-sealing, so this option replaces "
            "--raw-log, --workload, --command, --collective-backend, --warmup, "
            "--iterations, --seed and --device-uuid rather than supplementing them"
        ),
    )
    parser.add_argument(
        "--release-payload",
        action="store_true",
        help=(
            "seal the measurements as a scalability release payload. The claim "
            "flags are written here rather than by the measuring program, so a "
            "run cannot promote itself: this tool also holds the release "
            "preflight, and the strict audit still re-validates the payload."
        ),
    )
    parser.add_argument(
        "--device-uuid",
        action="append",
        default=None,
        metavar="UUID[@NODE]",
        help=(
            "device that ran each rank, in rank order. Supply it for a run whose "
            "ranks span hosts: this process can only see its own devices, so "
            "detection alone would attribute every rank to this host."
        ),
    )
    args = parser.parse_args(argv)

    recorded = (
        _recorded_provenance(args.recorded_provenance)
        if args.recorded_provenance is not None
        else None
    )
    if recorded is None and any(
        item is None
        for item in (
            args.raw_log,
            args.workload,
            args.command,
            args.collective_backend,
            args.warmup,
            args.iterations,
            args.seed,
        )
    ):
        # An incomplete invocation is an interface error, so it is reported
        # before the host preflight rather than behind it. A caller who supplied
        # neither provenance source should not be told about GPU UUIDs.
        raise SystemExit(
            "sealing a fresh run requires --raw-log, --workload, --command, "
            "--collective-backend, --warmup, --iterations and --seed; supply "
            "--recorded-provenance only to re-seal a payload that already has one"
        )
    preflight_errors = environment_errors(
        world_size=args.world_size,
        seal_destination=args.output,
        commit=recorded.commit if recorded is not None else None,
        device_uuids=recorded.devices if recorded is not None else None,
    )
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
    if args.release_payload:
        if int(evidence.get("world_size", 0) or 0) <= 1:
            raise SystemExit(
                "a scalability release payload must shard one workload across "
                "ranks; a single-device run can only be provenance evidence"
            )
        for key, value in (
            ("claim_evidence_type", "production_training_benchmark"),
            ("release_payload", True),
            ("scalability_claim_allowed", True),
            ("release_gate_allowed", True),
        ):
            evidence[key] = value
    if recorded is not None:
        provenance = recorded
    else:
        devices, ranks = _rank_devices(args.device_uuid, world_size=args.world_size)
        provenance = RuntimeProvenance(
            commit=_output(("git", "rev-parse", "HEAD")),
            workload_sha256=sha256_file(args.workload),
            command=tuple(args.command),
            devices=devices,
            topology=_output(("nvidia-smi", "topo", "-m")),
            rank_mapping=ranks,
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
