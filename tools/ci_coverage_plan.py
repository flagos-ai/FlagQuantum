"""Partition the collected coverage selection without collecting it on every slot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from tools.pytest_shard import balanced_assignments, load_duration_hints

SCHEMA = "flagquantum_coverage_plan_v1"
MAX_SHARDS = 192
MAX_FILE_BATCH_TESTS = 16
MAX_FILE_BATCH_SECONDS = 60.0


def write_plan(nodeids: tuple[str, ...], directory: Path, requested: int) -> int:
    """Bound file batches and preserve every collected node exactly once."""
    if not nodeids or len(nodeids) != len(set(nodeids)):
        raise ValueError("Coverage collection must be nonempty and unique")
    if not 1 <= requested <= MAX_SHARDS:
        raise ValueError("Coverage shard count must be between 1 and 192")
    files: dict[str, list[str]] = defaultdict(list)
    for nodeid in nodeids:
        filename = nodeid.split("::", 1)[0]
        if not filename.startswith("tests/") or ".." in Path(filename).parts:
            raise ValueError(f"Invalid collected test path: {filename}")
        files[filename].append(nodeid)
    default, hints = load_duration_hints()
    # Keep untimed files bounded by test count. Measured heavy tests must also
    # be separable: three QLBM cases in one 16-test batch exhausted a job while
    # other slots were idle. A single long test remains indivisible.
    batches: dict[str, list[str]] = {}
    for filename, nodes in sorted(files.items()):
        batch: list[str] = []
        seconds = 0.0
        batch_index = 0
        for node in nodes:
            weight = hints.get(node, default)
            if batch and (
                len(batch) >= MAX_FILE_BATCH_TESTS
                or seconds + weight > MAX_FILE_BATCH_SECONDS
            ):
                batches[f"{filename}::batch-{batch_index}"] = batch
                batch_index += 1
                batch = []
                seconds = 0.0
            batch.append(node)
            seconds += weight
        if batch:
            batches[f"{filename}::batch-{batch_index}"] = batch
    weights = {
        name: sum(hints.get(node, default) for node in nodes)
        for name, nodes in batches.items()
    }
    count = min(requested, len(batches))
    assignments = balanced_assignments(
        tuple(batches), count, default_seconds=default, durations=weights
    )
    shards: list[list[str]] = [[] for _ in range(count)]
    for batch in sorted(batches):
        shards[assignments[batch]].extend(batches[batch])
    assert all(shards)
    directory.mkdir(parents=True, exist_ok=False)
    for index, nodes in enumerate(shards):
        (directory / f"shard-{index}.txt").write_text("\n".join(nodes) + "\n")
    payload = {
        "schema": SCHEMA,
        "source_sha": os.environ["GITHUB_SHA"],
        "count": count,
        "tests": len(nodeids),
        "shards": [
            {
                "tests": len(nodes),
                "sha256": hashlib.sha256(
                    (directory / f"shard-{i}.txt").read_bytes()
                ).hexdigest(),
            }
            for i, nodes in enumerate(shards)
        ],
    }
    (directory / "plan.json").write_text(json.dumps(payload, indent=2) + "\n")
    matrix = {"include": [{"shard": i, "hosted-bucket": i % 8} for i in range(count)]}
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a") as stream:
            stream.write(
                f"count={count}\nmatrix={json.dumps(matrix, separators=(',', ':'))}\n"
            )
    print(
        f"Planned {len(nodeids)} tests in {len(files)} files across {count} nonempty shards",
        flush=True,
    )
    return count


def verify_plan(directory: Path, index: int) -> Path:
    """Validate the source binding and shard contents before passing them to pytest."""
    payload = json.loads((directory / "plan.json").read_text())
    if payload["schema"] != SCHEMA or payload["source_sha"] != os.environ["GITHUB_SHA"]:
        raise ValueError("Coverage plan does not match this source commit")
    if not 1 <= payload["count"] <= MAX_SHARDS or not 0 <= index < payload["count"]:
        raise ValueError("Invalid coverage shard index or count")
    path = directory / f"shard-{index}.txt"
    nodes = path.read_text().splitlines()
    metadata = payload["shards"][index]
    if (
        not nodes
        or len(nodes) != len(set(nodes))
        or len(nodes) != metadata["tests"]
        or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]
    ):
        raise ValueError("Coverage shard contents differ from the collected plan")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    print(verify_plan(args.directory, args.index))


if __name__ == "__main__":
    main()
