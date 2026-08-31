"""Summarize all completed stage-A FlagQuantum/CoTenGra gap reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", default="stage_a")
    return parser.parse_args()


def summarize(
    payloads: list[dict[str, Any]],
    *,
    stage: str = "stage_a",
) -> dict[str, Any]:
    comparable = [payload for payload in payloads if payload.get("comparable")]
    if not comparable:
        largest = None
    else:
        largest = max(
            comparable,
            key=lambda payload: max(
                float(value)
                for value in payload["ratios"].values()
                if value is not None
            ),
        )
    return {
        "schema_version": 1,
        "comparison_scope": "planning",
        "result_count": len(payloads),
        "comparable_result_count": len(comparable),
        "workloads": [
            {
                "name": payload["workload_name"],
                "identity": payload["workload_identity"],
                "comparable": payload["comparable"],
                "ratios": payload["ratios"],
                "largest_gap": payload["largest_gap"],
                "next_optimization_target": payload["next_optimization_target"],
            }
            for payload in sorted(payloads, key=lambda item: item["workload_name"])
        ],
        "largest_observed_gap": (
            None
            if largest is None
            else {
                "workload_name": largest["workload_name"],
                "workload_identity": largest["workload_identity"],
                "largest_gap": largest["largest_gap"],
                "ratios": largest["ratios"],
            }
        ),
        "stage": str(stage),
        "stage_status": (
            "completed" if len(comparable) == len(payloads) else "incomplete"
        ),
        "next_optimization_target": (
            (
                "reduce native planning overhead or bypass reconfiguration"
                if largest is not None
                and largest["largest_gap"] == "search_time"
                else "improve native contraction path search before further scale work"
            )
            if largest is not None
            else "establish comparable baselines"
        ),
        "claim_boundary": (
            "Planning-only development evidence. Cost estimates share the same "
            "framework-neutral topology but are not execution measurements."
        ),
    }


def main() -> None:
    arguments = _arguments()
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(arguments.input.glob("*.gap.json"))
    ]
    if not payloads:
        raise RuntimeError("no TN gap reports found")
    payload = summarize(payloads, stage=arguments.stage)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
