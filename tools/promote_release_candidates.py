#!/usr/bin/env python
"""Promote a passing candidate set into the release directory.

Sealing writes into a candidate directory because the strict promotion audit
refuses to seal while the release directory already holds promoted JSON, and
because a payload must be checked before it becomes release evidence. This tool
is the single step between the two, and it is fail-closed: it re-runs the same
gate over the candidates and moves nothing unless that gate passes with a
signing key, so promotion cannot happen on an unchecked or unsigned set.

The gate is selected by name because a release directory belongs to one
capability: a candidate set is checked against the gate of the capability it
claims, and no gate accepts another capability's artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path

# Each entry names the module that owns the gate and the evaluator that decides.
# The keys are capability names, not module names, so a caller states which
# release is being promoted.
_GATES = {
    "statevector": (
        "benchmarks.internal.evidence.statevector_release_gate",
        "evaluate_issue044_release",
    ),
    "tensor-network": (
        "benchmarks.internal.evidence.tensor_network_release_gate",
        "evaluate_tensor_network_release",
    ),
}
_DEFAULT_GATE = "statevector"


def _load_gate(name: str):
    """Return the gate module and its evaluator for the capability ``name``."""

    module_name, evaluator = _GATES[name]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        # Running the tool as a script leaves the repository root off the import
        # path, and every gate lives under it.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        module = import_module(module_name)
    return module, getattr(module, evaluator)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--gate",
        choices=sorted(_GATES),
        default=_DEFAULT_GATE,
        help="the capability release the candidate set must satisfy",
    )
    parser.add_argument(
        "--release-directory",
        type=Path,
        help="where promoted release payloads live; defaults to the gate's own",
    )
    args = parser.parse_args(argv)

    gate, evaluate = _load_gate(args.gate)
    release_directory = (
        args.release_directory if args.release_directory is not None else gate.RESULTS
    )
    manifest = gate.load_manifest()
    signing_key = os.environ.get("FQ_EVIDENCE_SIGNING_KEY", "").encode()
    if not signing_key:
        raise SystemExit(
            "FQ_EVIDENCE_SIGNING_KEY is required: an unsigned candidate set "
            "cannot be promoted"
        )
    candidates = sorted(args.candidate.glob("*.json"))
    if not candidates:
        raise SystemExit(f"no candidate artifacts under {args.candidate}")
    artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in candidates]
    passed, blockers = evaluate(artifacts, manifest, signing_key=signing_key)
    if not passed:
        raise SystemExit(
            "candidate set does not satisfy the release gate: " + "; ".join(blockers)
        )
    release_directory.mkdir(parents=True, exist_ok=True)
    for path in candidates:
        shutil.move(str(path), str(release_directory / path.name))
        print(f"promoted {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
