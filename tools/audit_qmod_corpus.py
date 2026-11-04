"""Audit a pinned Qmod corpus on the current FlagQuantum checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.ecosystem import InteropConversionError
from flagquantum.ecosystem._qmod.source import load_qmod
from flagquantum.services import synthesis


def probe(
    path: Path,
    revision: str,
    memory_bytes: int,
    execute: bool,
    classical_inputs: Mapping[str, float] | None = None,
) -> dict[str, str | int]:
    try:
        load_qmod(path, source_version=revision, classical_inputs=classical_inputs)
    except InteropConversionError as error:
        return {"stage": "frontend_refused", "detail": str(error)}
    if not execute:
        return {"stage": "frontend_admitted"}

    from flagquantum import run

    try:
        program = synthesis.synthesize(
            path,
            source_version=revision,
            memory_limit_bytes=memory_bytes,
            classical_inputs=classical_inputs,
        )
    except (synthesis.SynthesisError, ValueError, RuntimeError) as error:
        return {"stage": "planning_refused", "detail": str(error)}
    resources = cast(ResourceEstimate, program.report["resources"])
    costs = {
        "qubits": resources.n_qubits,
        "operations": resources.n_operations,
        "depth": resources.depth,
    }
    try:
        state = run(program).statevector()
        norm = float(state.abs().square().sum())
        if not math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            return {
                "stage": "execution_failed",
                "detail": f"state norm {norm}",
                **costs,
            }
    except (ValueError, RuntimeError) as error:
        return {"stage": "execution_failed", "detail": str(error), **costs}
    return {"stage": "execution_smoke_passed", **costs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpus", type=Path, help="directory with tracker JSON and qmod/"
    )
    parser.add_argument("output", type=Path, help="path to write the result JSON")
    parser.add_argument(
        "--execute", action="store_true", help="plan and run admitted files"
    )
    parser.add_argument("--memory-mib", type=int, default=512)
    parser.add_argument(
        "--bindings", type=Path, help="JSON mapping of source paths to real inputs"
    )
    args = parser.parse_args()
    if args.memory_mib < 1:
        parser.error("--memory-mib must be positive")
    manifest = json.loads((args.corpus / "qmod_compatibility_tracker.json").read_text())
    bindings = {} if args.bindings is None else json.loads(args.bindings.read_text())
    if not isinstance(bindings, dict) or not set(bindings) <= {
        entry["path"] for entry in manifest["files"]
    }:
        parser.error("--bindings must map known corpus paths to real inputs")
    rows = []
    for entry in manifest["files"]:
        relative = entry["path"]
        path = args.corpus / "qmod" / relative
        raw = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob != entry["source_blob_sha"]:
            raise ValueError(f"corpus blob mismatch for {relative}")
        result = probe(
            path,
            entry["source_commit"],
            args.memory_mib * 1024**2,
            args.execute,
            bindings.get(relative),
        )
        row = {"path": relative, "blob": blob, **result}
        if relative in bindings:
            row["classical_inputs"] = bindings[relative]
        rows.append(row)
    counts = Counter(row["stage"] for row in rows)
    args.output.write_text(
        json.dumps({"counts": counts, "files": rows}, indent=2) + "\n"
    )
    print(dict(counts))


if __name__ == "__main__":
    main()
