"""Compare explicitly selected complete cohorts; retain the original failed attempt."""

from __future__ import annotations

import gzip
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict


class Metric(TypedDict):
    before_median: float
    after_median: float
    before_range: list[int]
    after_range: list[int]
    reduction_percent: float
    absolute_reduction: float


root = Path(__file__).parent
base_path = root / "baseline.json.gz"
retry_path = root / "qdrift-corrective.json.gz"
after_path = root / "candidate.json.gz"
base, retry, after = (
    json.loads(gzip.decompress(p.read_bytes()))
    for p in (base_path, retry_path, after_path)
)
assert retry["complete"] and after["complete"]
assert (
    not base["source_changes"]
    and not retry["source_changes"]
    and not after["source_changes"]
)
result: dict[str, Any] = {
    "scope": "local CPU single-thread sequential before/after cohorts; no randomized-pair or release claim",
    "raw_receipts": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (base_path, retry_path, after_path)
    },
    "before_selection": "Five complete original cohorts plus the full corrective qDRIFT cohort. Original qDRIFT cohort retained, including one process-cleanup failure; no individual sample replaced.",
    "original_attempt_statuses": dict(
        Counter(s["status"] for r in base["rows"].values() for s in r["samples"])
    ),
    "rows": {},
}


def cohort(receipt: dict[str, Any], name: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = receipt["rows"][name]["samples"]
    assert len(samples) == 9
    assert Counter(s["mode"] for s in samples) == {
        "timing": 5,
        "memory": 3,
        "diagnostic": 1,
    }
    assert all(s["status"] == "verified" and s["process_group_reaped"] for s in samples)
    assert (
        len(
            {
                (s["circuit_hash"], json.dumps(s["resources"], sort_keys=True))
                for s in samples
            }
        )
        == 1
    )
    return samples


def measure(a: list[int], b: list[int]) -> Metric:
    x, y = statistics.median(a), statistics.median(b)
    return {
        "before_median": x,
        "after_median": y,
        "before_range": [min(a), max(a)],
        "after_range": [min(b), max(b)],
        "reduction_percent": 100 * (x - y) / x,
        "absolute_reduction": x - y,
    }


for name in base["rows"]:
    before_receipt = retry if name == "qdrift" else base
    left, right = cohort(before_receipt, name), cohort(after, name)
    assert left[0]["circuit_hash"] == right[0]["circuit_hash"]
    assert left[0]["resources"] == right[0]["resources"]
    changed = sorted(
        k
        for k, v in before_receipt["source_hashes"].items()
        if after["source_hashes"].get(k) != v
    )
    allowed = {
        "benchmarks/synthesis/performance.py",
        "flagquantum/synth/_workflow.py",
        "flagquantum/runtime/planner/synthesis.py",
    }
    assert set(changed) <= allowed, changed
    lt, rt = ([s for s in xs if s["mode"] == "timing"] for xs in (left, right))
    metrics = {
        key: measure([s[key] for s in lt], [s[key] for s in rt])
        for key in ("e0_ns", "execution_ns")
    }
    metrics["compile_plus_execution_ns"] = measure(
        [s["e0_ns"] + s["execution_ns"] for s in lt],
        [s["e0_ns"] + s["execution_ns"] for s in rt],
    )
    for key, field in [
        ("compile_rss", "memory"),
        ("execution_rss", "execution_memory"),
    ]:
        metrics[key] = measure(
            [
                s[field]["window_sampled_peak_rss_bytes"]
                for s in left
                if s["mode"] == "memory"
            ],
            [
                s[field]["window_sampled_peak_rss_bytes"]
                for s in right
                if s["mode"] == "memory"
            ],
        )

    def inspections(xs: list[dict[str, Any]]) -> int:
        sample = next(s for s in xs if s["mode"] == "diagnostic")
        return sum(
            s["calls"]
            for s in sample["diagnostic"]["spans"]
            if s["function"] == "flagquantum.services._synthesis_execution._program"
        )

    result["rows"][name] = {
        "before_receipt": (
            "qdrift-corrective.json.gz" if name == "qdrift" else "baseline.json.gz"
        ),
        "source_changes": changed,
        "resources": left[0]["resources"],
        "circuit_hash": left[0]["circuit_hash"],
        "metrics": metrics,
        "inspection_calls": [inspections(left), inspections(right)],
        "meets_time_review_threshold": any(
            m["reduction_percent"] >= 10 and m["absolute_reduction"] >= 20_000_000
            for k, m in metrics.items()
            if k.endswith("_ns")
        ),
        "maximum_amplitude_error": max(
            s["verification"]["maximum_error"] for s in left + right
        ),
    }
(root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
for name, row in result["rows"].items():
    m = row["metrics"]
    print(
        name,
        [
            (
                k,
                round(v["before_median"] / 1e6, 3),
                round(v["after_median"] / 1e6, 3),
                round(v["reduction_percent"], 2),
            )
            for k, v in m.items()
            if k.endswith("_ns")
        ],
        row["meets_time_review_threshold"],
    )
