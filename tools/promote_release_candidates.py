#!/usr/bin/env python
"""Promote a passing ISSUE-044 candidate set into the release directory.

Sealing writes into a candidate directory because the strict promotion audit
refuses to seal while the release directory already holds promoted JSON, and
because a payload must be checked before it becomes release evidence. This tool
is the single step between the two, and it is fail-closed: it re-runs the same
gate over the candidates and moves nothing unless that gate passes with a
signing key, so promotion cannot happen on an unchecked or unsigned set.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path

if __package__:
    _GATE = import_module("benchmarks.internal.evidence.statevector_release_gate")
else:
    _GATE = import_module("statevector_release_gate")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--release-directory",
        type=Path,
        default=_GATE.RESULTS,
        help="where promoted release payloads live; defaults to the gate's own",
    )
    args = parser.parse_args(argv)

    manifest = _GATE.load_manifest()
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
    passed, blockers = _GATE.evaluate_issue044_release(
        artifacts, manifest, signing_key=signing_key
    )
    if not passed:
        raise SystemExit(
            "candidate set does not satisfy the release gate: " + "; ".join(blockers)
        )
    args.release_directory.mkdir(parents=True, exist_ok=True)
    for path in candidates:
        shutil.move(str(path), str(args.release_directory / path.name))
        print(f"promoted {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
