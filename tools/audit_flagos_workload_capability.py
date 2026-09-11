#!/usr/bin/env python
"""Aggregate checked-in FlagOS F1/F2/F3 evidence into the F4 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flagquantum.runtime.distributed.workload_capability import (  # noqa: E402
    build_flagos_workload_capability_matrix,
)

DEFAULT_CONFORMANCE = (
    ROOT / "artifacts/flagos_distributed_conformance_a800_f11_20260825.json"
)
DEFAULT_SCALE = ROOT / "artifacts/flagos_statevector_scale_f2_a800_20260826.json"
DEFAULT_TRAINING = ROOT / "artifacts/flagos_statevector_training_f3_a800_20260826.json"
DEFAULT_OUTPUT = ROOT / "artifacts/flagos_workload_capability_f4_20260826.json"


def load_evidence(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"evidence must be a JSON object: {path}")
    try:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        relative = str(path.resolve())
    environment = payload.get("environment", {})
    return payload, {
        "path": relative,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema": str(payload.get("schema", "")),
        "source_revision": str(environment.get("source_revision", "")),
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conformance", type=Path, default=DEFAULT_CONFORMANCE)
    parser.add_argument("--scale", type=Path, default=DEFAULT_SCALE)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    conformance, conformance_evidence = load_evidence(args.conformance)
    scale, scale_evidence = load_evidence(args.scale)
    training, training_evidence = load_evidence(args.training)
    matrix = build_flagos_workload_capability_matrix(
        conformance,
        scale,
        training,
        evidence_artifacts=(conformance_evidence, scale_evidence, training_evidence),
    )
    write_json_atomic(args.output, matrix.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
