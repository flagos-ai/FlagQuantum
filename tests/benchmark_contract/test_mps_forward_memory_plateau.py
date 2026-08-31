from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from benchmarks.internal.evidence.mps_forward_memory_plateau import audit_plateau

ROOT = Path(__file__).resolve().parents[2]


def test_mps_benchmark_entrypoint_resolves_repository_imports() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/internal/evidence/mps_forward_memory_plateau.py"),
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "Certify bounded layer-local memory" in completed.stdout


def _records(values):
    return [
        {
            "remaining_precomputed_entries": 0,
            "allocated_memory_bytes": value,
            "local_tensor_bytes": value - 10,
        }
        for value in values
    ]


def test_mps_plateau_audit_passes_bounded_rank_records() -> None:
    ranks = [
        {
            "rank": rank,
            "runs": [
                {
                    "depth": depth,
                    "layer_records": _records((100, 110, 120, 130, 140)),
                    "peak_allocated_memory_bytes": 200,
                    "layer_cache_empty_at_return": True,
                }
                for depth in (1, 2, 4, 8, 12, 20)
            ],
        }
        for rank in range(8)
    ]
    audit = audit_plateau(ranks, target_depth=20, memory_budget_bytes=10_000)
    assert audit["passed"] is True
    assert audit["blockers"] == ()


def test_mps_plateau_audit_rejects_leak_peak_and_missing_target() -> None:
    ranks = [
        {
            "rank": 0,
            "runs": [
                {
                    "depth": 12,
                    "layer_records": [
                        {
                            "remaining_precomputed_entries": 1,
                            "allocated_memory_bytes": value,
                            "local_tensor_bytes": 0,
                        }
                        for value in (0, 100, 200, 300, 400)
                    ],
                    "peak_allocated_memory_bytes": 20_000,
                    "layer_cache_empty_at_return": False,
                }
            ],
        }
    ]
    audit = audit_plateau(ranks, target_depth=20, memory_budget_bytes=1_000)
    assert audit["passed"] is False
    assert "rank_0_depth_12_layer_cache_not_drained" in audit["blockers"]
    assert "rank_0_depth_12_residual_memory_growth" in audit["blockers"]
    assert "rank_0_target_depth_missing" in audit["blockers"]
    assert "rank_0_return_cache_not_empty" in audit["blockers"]
