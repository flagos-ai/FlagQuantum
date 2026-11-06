"""Audit a pinned Qmod corpus on the current FlagQuantum checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.ecosystem import InteropConversionError
from flagquantum.ecosystem._qmod.source import load_qmod
from flagquantum.services import synthesis
from flagquantum.services._qmod_workflow import run_qaoa_workflow
from tools._qmod_basis_mps import probe_basis_mps


def probe(
    path: Path,
    revision: str,
    memory_bytes: int,
    execute: bool,
    classical_inputs: Mapping[str, float | Sequence[float]] | None = None,
    *,
    basis_mps: bool = False,
) -> dict[str, str | int]:
    try:
        imported = load_qmod(
            path, source_version=revision, classical_inputs=classical_inputs
        )
    except InteropConversionError as error:
        return {"stage": "frontend_refused", "detail": str(error)}
    if not execute:
        return {"stage": "frontend_admitted"}

    if basis_mps:
        return probe_basis_mps(imported, memory_bytes)

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
        evidence: dict[str, str | int] = {}
        if imported.workflow is None:
            state = run(program).statevector()
        else:
            workflow = run_qaoa_workflow(
                imported.workflow, memory_limit_bytes=memory_bytes
            )
            state = workflow.vqe.statevector
            evidence = {
                "workflow_evaluations": len(workflow.vqe.evaluations),
                "workflow_termination": workflow.vqe.termination,
                "resource_parameters": (
                    "source_initial_point"
                    if classical_inputs is None
                    else "explicit_quantum_bindings"
                ),
            }
        norm = float(state.abs().square().sum())
        if not math.isclose(norm, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            return {
                "stage": "execution_failed",
                "detail": f"state norm {norm}",
                **costs,
            }
    except (ValueError, RuntimeError) as error:
        return {"stage": "execution_failed", "detail": str(error), **costs}
    return {"stage": "execution_smoke_passed", **costs, **evidence}


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
    parser.add_argument(
        "--basis-mps-paths",
        type=Path,
        help="JSON list of sources explicitly selected for exact basis MPS",
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
    basis_mps_paths = (
        []
        if args.basis_mps_paths is None
        else json.loads(args.basis_mps_paths.read_text())
    )
    known_paths = {entry["path"] for entry in manifest["files"]}
    if not isinstance(basis_mps_paths, list) or any(
        not isinstance(path, str) or path not in known_paths for path in basis_mps_paths
    ):
        parser.error("--basis-mps-paths must list known corpus paths")
    if basis_mps_paths and not args.execute:
        parser.error("--basis-mps-paths requires --execute")
    basis_mps_set = set(basis_mps_paths)
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
            basis_mps=relative in basis_mps_set,
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
