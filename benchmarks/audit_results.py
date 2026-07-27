"""Audit benchmark JSON files for FlagQuantum scalability claims.

Repro commands
--------------
Scan curated benchmark results:
  python benchmarks/audit_results.py --input benchmarks/results

Fail unless every JSON file is valid sharded scalability evidence:
  python benchmarks/audit_results.py --input benchmarks/results/scalability --require-scalability

Write a machine-readable audit summary:
  python benchmarks/audit_results.py --input benchmarks/results --json-output scalability_audit_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_audit_module = import_module("flagquantum.runtime.audit.engine")
_evidence_module = import_module("flagquantum.runtime.observability.evidence")


def _json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [] if _is_generated_audit_summary(path) else [path]
    if path.name == "results":
        curated = ("local", "comparison", "smoke", "scalability")
        return sorted(
            item
            for category in curated
            for item in (path / category).rglob("*.json")
            if item.is_file() and not _is_generated_audit_summary(item)
        )
    return sorted(
        item
        for item in path.rglob("*.json")
        if item.is_file() and not _is_generated_audit_summary(item)
    )


def _is_generated_audit_summary(path: Path) -> bool:
    return path.name in {
        "scalability_audit_summary.json",
        "audit_summary.json",
    } or path.stem.endswith("_audit_summary")


def _read_json(path: Path) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, {"type": type(exc).__name__, "message": str(exc)}
    if not isinstance(data, dict):
        return None, {
            "type": "TypeError",
            "message": "top-level JSON value is not an object",
        }
    return data, None


def audit_paths(
    paths: list[Path],
    *,
    require_scalability: bool = False,
    signing_key: bytes | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload, error = _read_json(path)
        if error is not None:
            records.append(
                {
                    "path": str(path),
                    "status": "unreadable",
                    "error": error,
                    "scalability_audit": {
                        "valid": False,
                        "scalability_claim_allowed": False,
                        "distribution_semantics": "unknown",
                        "errors": (error["message"],),
                        "warnings": (),
                    },
                }
            )
            continue
        assert payload is not None
        if payload.get("benchmark") == "scalability_results_audit":
            continue
        if require_scalability:
            provenance_valid, provenance_errors = _evidence_module.verify_evidence_artifact(
                payload, signing_key=signing_key or b""
            )
            audited_payload = payload.get("evidence", {})
            if not provenance_valid or not isinstance(audited_payload, dict):
                records.append(
                    {
                        "path": str(path),
                        "status": "provenance_rejected",
                        "benchmark": payload.get("benchmark"),
                        "scalability_audit": {
                            "valid": False,
                            "scalability_claim_allowed": False,
                            "release_gate_allowed": False,
                            "distribution_semantics": "unknown",
                            "errors": provenance_errors,
                            "warnings": (),
                        },
                    }
                )
                continue
            audit = _audit_module.validate_distributed_claim_evidence(audited_payload)
        else:
            audit = _audit_module.audit_distributed_scalability(payload)
        records.append(
            {
                "path": str(path),
                "status": "ok",
                "benchmark": payload.get("benchmark"),
                "distribution_semantics": audit.distribution_semantics,
                "scalability_claim_allowed": audit.scalability_claim_allowed,
                "claim_evidence_type": audit.claim_evidence_type,
                "release_gate_allowed": audit.release_gate_allowed,
                "scalability_audit": audit.summary(),
            }
        )

    claimable = [
        item for item in records if item["scalability_audit"]["release_gate_allowed"]
    ]
    invalid = [item for item in records if not item["scalability_audit"]["valid"]]
    return {
        "benchmark": "scalability_results_audit",
        "file_count": len(records),
        "claimable_count": len(claimable),
        "invalid_count": len(invalid),
        "claimable_paths": tuple(item["path"] for item in claimable),
        "invalid_paths": tuple(item["path"] for item in invalid),
        "records": records,
    }


def _write_json_output(path_text: str, text: str) -> None:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path("benchmarks") / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="JSON file or directory. Can be passed multiple times.",
    )
    parser.add_argument("--json-output", default="", help="Optional JSON output path.")
    parser.add_argument(
        "--require-scalability",
        action="store_true",
        help="Exit nonzero unless every audited file is release-grade sharded scalability evidence.",
    )
    args = parser.parse_args()

    if args.input:
        input_texts = args.input
    elif args.require_scalability:
        input_texts = ["benchmarks/results/scalability"]
    else:
        input_texts = ["benchmarks/results"]
    inputs = [Path(item) for item in input_texts]
    if args.require_scalability:
        inputs = [
            item / "scalability" if item.name == "results" else item for item in inputs
        ]
    paths: list[Path] = []
    for item in inputs:
        paths.extend(_json_files(item))
    signing_key = os.environ.get("FQ_EVIDENCE_SIGNING_KEY", "").encode()
    payload = audit_paths(
        paths,
        require_scalability=args.require_scalability,
        signing_key=signing_key,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        _write_json_output(args.json_output, text)
    if args.require_scalability and (
        payload["file_count"] == 0
        or payload["claimable_count"] != payload["file_count"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
