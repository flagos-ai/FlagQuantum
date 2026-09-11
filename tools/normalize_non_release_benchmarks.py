"""Normalize curated non-release benchmark evidence metadata.

This migration is intentionally fail-closed: it never upgrades an artifact to
release evidence or enables a scalability claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flagquantum.runtime.audit import audit_distributed_scalability

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "results"
EVIDENCE_CLASSES = {
    "local": "local_non_release",
    "comparison": "comparison_non_release",
    "smoke": "non_release_smoke",
}
BLOCKERS = {
    "local": "local benchmark is not release-certified scalability evidence",
    "comparison": "comparison benchmark is not release-certified scalability evidence",
    "smoke": "smoke benchmark is not release-certified scalability evidence",
}


def normalize_payload(
    payload: dict[str, Any], *, category: str, artifact_name: str = "benchmark"
) -> bool:
    """Apply the non-release evidence contract and report whether it changed."""

    before = json.dumps(payload, sort_keys=True)
    payload.setdefault("benchmark", artifact_name)
    payload["benchmark_evidence_class"] = EVIDENCE_CLASSES[category]
    payload["non_release_evidence"] = True
    payload["release_gate_allowed"] = False
    payload["scalability_claim_allowed"] = False
    blockers = payload.get("scalability_blockers")
    if not isinstance(blockers, list) or not blockers:
        payload["scalability_blockers"] = [BLOCKERS[category]]
    payload["claim_evidence_type"] = "development_smoke"
    if payload.get("distribution_semantics") in {None, "", "unknown", "unverified"}:
        execution_semantics = payload.get("execution_semantics")
        payload["distribution_semantics"] = (
            execution_semantics
            if execution_semantics == "single_device_fast_path"
            else (
                "rank_local_replicated_kernel"
                if int(payload.get("world_size", 1) or 1) > 1
                else "single_device_fast_path"
            )
        )
    audit = audit_distributed_scalability(payload)
    if not audit.valid:
        payload["intended_distribution_semantics"] = payload["distribution_semantics"]
        payload["distribution_semantics"] = "requires_runtime_summary"
    return before != json.dumps(payload, sort_keys=True)


def main() -> None:
    changed = 0
    for category in EVIDENCE_CLASSES:
        for path in sorted((ROOT / category).rglob("*.json")):
            if path.name.endswith("audit_summary.json"):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError(f"{path}: top-level JSON must be an object")
            if normalize_payload(payload, category=category, artifact_name=path.stem):
                path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                changed += 1
    print(f"normalized {changed} non-release benchmark artifacts")


if __name__ == "__main__":
    main()
