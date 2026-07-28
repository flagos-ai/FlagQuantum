from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path("benchmarks/statevector_full_width_capacity_report.py")
SPEC = importlib.util.spec_from_file_location(
    "statevector_full_width_capacity_report", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _distributed(*, semantics: str = "sharded_across_ranks") -> dict:
    return {
        "world_size": 16,
        "node_count": 2,
        "distribution_semantics": semantics,
        "forward_distribution_semantics": semantics,
        "backward_distribution_semantics": semantics,
        "benchmark_protocol": {"full_state_materialization": False},
        "correctness": {"passed": True},
        "validation_blockers": [],
    }


def test_capacity_completion_requires_full_lifecycle_sharding() -> None:
    assert MODULE.is_sharded_capacity_completion(_distributed()) is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("distribution_semantics", "rank_local_replicated_kernel"),
        ("forward_distribution_semantics", "rank_local_replicated_kernel"),
        ("backward_distribution_semantics", "rank_local_replicated_kernel"),
    ],
)
def test_replicated_phase_cannot_support_capacity_completion(
    field: str, value: str
) -> None:
    distributed = _distributed()
    distributed[field] = value
    assert MODULE.is_sharded_capacity_completion(distributed) is False
