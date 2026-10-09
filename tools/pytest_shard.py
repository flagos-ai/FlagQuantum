"""Deterministically partition one pytest selection across CI jobs.

Each shard still performs normal pytest collection and marker filtering. This
plugin runs last and keeps a stable subset of the remaining node IDs, so the
union of all shards is exactly the selection the unsharded command would run.
"""

from __future__ import annotations

import hashlib

import pytest


def shard_for(nodeid: str, count: int) -> int:
    """Return the stable zero-based shard for ``nodeid``."""

    if count < 1:
        raise ValueError("shard count must be positive")
    digest = hashlib.sha256(nodeid.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


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
    selected = [item for item in items if shard_for(item.nodeid, count) == index]
    deselected = [item for item in items if shard_for(item.nodeid, count) != index]
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)
