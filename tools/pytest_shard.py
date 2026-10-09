"""Deterministically partition one pytest selection across CI jobs.

Each shard still performs normal pytest collection and marker filtering. This
plugin runs last and uses measured durations to make a deterministic
least-loaded partition of the remaining node IDs, so the union of all shards
is exactly the selection the unsharded command would run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DURATION_HINTS = ROOT / "contracts" / "pytest-duration-hints.json"
EXPECTED_HINT_SCHEMA = "flagquantum_pytest_duration_hints_v1"


def shard_for(nodeid: str, count: int) -> int:
    """Return the stable zero-based shard for ``nodeid``."""

    if count < 1:
        raise ValueError("shard count must be positive")
    digest = hashlib.sha256(nodeid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def load_duration_hints(
    path: Path = DURATION_HINTS,
) -> tuple[float, dict[str, float]]:
    """Load measured durations used to balance the coverage critical path."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != EXPECTED_HINT_SCHEMA:
        raise ValueError(f"duration hints must use {EXPECTED_HINT_SCHEMA}")
    default = float(payload.get("default_seconds", 0.0))
    if default <= 0:
        raise ValueError("duration hint default_seconds must be positive")
    raw = payload.get("durations")
    if not isinstance(raw, dict):
        raise ValueError("duration hints must contain a durations object")
    durations = {str(nodeid): float(seconds) for nodeid, seconds in raw.items()}
    if any(seconds <= 0 for seconds in durations.values()):
        raise ValueError("every duration hint must be positive")
    return default, durations


def balanced_assignments(
    nodeids: tuple[str, ...],
    count: int,
    *,
    default_seconds: float,
    durations: dict[str, float],
) -> dict[str, int]:
    """Assign tests with deterministic longest-processing-time scheduling."""

    if count < 1:
        raise ValueError("shard count must be positive")
    if default_seconds <= 0:
        raise ValueError("default duration must be positive")
    if len(nodeids) != len(set(nodeids)):
        raise ValueError("node IDs must be unique")

    def order_key(nodeid: str) -> tuple[float, bytes]:
        weight = durations.get(nodeid, default_seconds)
        return (-weight, hashlib.sha256(nodeid.encode("utf-8")).digest())

    loads = [0.0] * count
    assignments: dict[str, int] = {}
    for nodeid in sorted(nodeids, key=order_key):
        shard = min(range(count), key=lambda candidate: (loads[candidate], candidate))
        assignments[nodeid] = shard
        loads[shard] += durations.get(nodeid, default_seconds)
    return assignments


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("flagquantum-sharding")
    group.addoption("--fq-shard-count", type=int, default=1)
    group.addoption("--fq-shard-index", type=int, default=0)


def pytest_configure(config: pytest.Config) -> None:
    count = int(config.getoption("--fq-shard-count"))
    index = int(config.getoption("--fq-shard-index"))
    if count < 1:
        raise pytest.UsageError("--fq-shard-count must be positive")
    if not 0 <= index < count:
        raise pytest.UsageError(
            f"--fq-shard-index must be between 0 and {count - 1}, got {index}"
        )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    count = int(config.getoption("--fq-shard-count"))
    if count == 1:
        return
    index = int(config.getoption("--fq-shard-index"))
    default_seconds, durations = load_duration_hints()
    assignments = balanced_assignments(
        tuple(item.nodeid for item in items),
        count,
        default_seconds=default_seconds,
        durations=durations,
    )
    selected = [item for item in items if assignments[item.nodeid] == index]
    deselected = [item for item in items if assignments[item.nodeid] != index]
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)
