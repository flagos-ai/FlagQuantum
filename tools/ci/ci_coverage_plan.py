"""Partition the collected coverage selection without collecting it on every slot."""

from __future__ import annotations

import sys as _entry_sys
from pathlib import Path as _EntryPath

if __package__ in (None, ""):
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from tools.ci.pytest_shard import balanced_assignments, load_duration_hints

SCHEMA = "flagquantum_coverage_plan_v1"
MAX_SHARDS = 256
EXCLUSIVE_TEST_SECONDS = 120.0
MAX_FILE_BATCH_TESTS = 16
MAX_FILE_BATCH_SECONDS = 60.0
DEFAULT_TIMEOUT_MINUTES = 60
# Original embedding reports used 1.09 CPU cores out of twelve admitted.
# Map only independently verified complete workloads to existing BW services;
# keep Maxwell on A800 because its original run actually used seven cores.
_EMBEDDING_TEST = "tests/integration/test_synthesis_projection_embedding_notebook.py::"
_SERIAL_BW_RUNNERS: dict[str, str] = {
    _EMBEDDING_TEST
    + "test_original_chemistry_and_four_embedding_validations": "fq-bw1100-10-cpu16-validation",
    _EMBEDDING_TEST
    + "test_native_frame_and_all_complete_nonzero_state_amplitudes": "fq-bw1100-11-cpu16-validation",
    _EMBEDDING_TEST
    + "test_actual_full_optimizer_budget_and_unrepaired_estimates": "fq-bw1100-12-cpu16-validation",
    _EMBEDDING_TEST
    + "test_invalid_singles_angles_fail_before_execution": "fq-bw1100-2-cpu16-validation",
    # Keep the complete memory-intensive state oracle on a qualified BW slot.
    "tests/integration/test_synthesis_glued_trees_notebook.py::"
    "test_wide_full_state_at_original_first_time": "fq-bw1100-17-cpu16-validation",
    "tests/integration/test_synthesis_native_autocallable.py::"
    "test_complete_native_source_and_circuit[2-0.22]": "fq-bw1100-15-cpu16-validation",
    "tests/integration/test_synthesis_native_autocallable.py::"
    "test_complete_native_source_and_circuit[2-0.1274075]": "fq-bw1100-16-cpu16-validation",
}
_CPU_QUALIFICATION = (
    Path(__file__).resolve().parents[2]
    / "docs/development/migration/ci-cpu-cache-qualification.json"
)
# Original optimization workloads remain indivisible. These are finite job
# budgets, not measured duration hints; reserve separate slots when available.
LONG_TEST_TIMEOUT_MINUTES = {
    "tests/integration/test_synthesis_induced_color_notebook.py::"
    "test_actual_fifty_iteration_optimization_and_sampling": 120,
    "tests/integration/test_synthesis_min_coloring_notebook.py::"
    "test_actual_hundred_iteration_optimization_and_sampling": 120,
}


def write_plan(nodeids: tuple[str, ...], directory: Path, requested: int) -> int:
    """Bound file batches and preserve every collected node exactly once."""
    if not nodeids or len(nodeids) != len(set(nodeids)):
        raise ValueError("Coverage collection must be nonempty and unique")
    if not 1 <= requested <= MAX_SHARDS:
        raise ValueError("Coverage shard count must be between 1 and 256")
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
            weight = max(
                hints.get(node, default), LONG_TEST_TIMEOUT_MINUTES.get(node, 0) * 60
            )
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
        name: sum(
            max(hints.get(node, default), LONG_TEST_TIMEOUT_MINUTES.get(node, 0) * 60)
            for node in nodes
        )
        for name, nodes in batches.items()
    }
    count = min(requested, len(batches))
    # A long measured case must not inherit a tail of untimed neighbors. Keep
    # a separate slot when the requested fan-out has room for the other cases.
    heavy = sorted(
        (
            name
            for name, nodes in batches.items()
            if len(nodes) == 1 and weights[name] >= EXCLUSIVE_TEST_SECONDS
        ),
        key=lambda name: (-weights[name], name),
    )
    reserve = min(len(heavy), count - 1) if count < len(batches) else len(heavy)
    exclusive = heavy[:reserve]
    assignments = {name: index for index, name in enumerate(exclusive)}
    remaining = tuple(name for name in batches if name not in assignments)
    if remaining:
        balanced = balanced_assignments(
            remaining, count - reserve, default_seconds=default, durations=weights
        )
        assignments.update({name: index + reserve for name, index in balanced.items()})
    shards: list[list[str]] = [[] for _ in range(count)]
    for batch_key in sorted(batches):
        shards[assignments[batch_key]].extend(batches[batch_key])
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
    sole_nodes = {nodes[0] for nodes in shards if len(nodes) == 1}
    bw_runners = {
        node: runner
        for node, runner in _SERIAL_BW_RUNNERS.items()
        if node in sole_nodes
    }
    if bw_runners:
        qualification = json.loads(_CPU_QUALIFICATION.read_text())
        qualified = {row["selector"] for row in qualification["qualified_slots"]}
        if (
            "coverage-py3.12" not in qualification["profiles"]
            or not set(bw_runners.values()) <= qualified
        ):
            raise ValueError(
                "Serial coverage runners need the verified locked coverage profile"
            )
    entries = []
    for i, nodes in enumerate(shards):
        runner = bw_runners.get(nodes[0], "") if len(nodes) == 1 else ""
        a800 = (
            len(nodes) == 1
            and hints.get(nodes[0], default) >= 300.0
            and nodes[0] not in bw_runners
        )
        if not runner and not a800:
            # Keep ordinary work off the pool reserved for the expensive cases.
            # Existing BW and MC slots both admit four CPUs; their online pool
            # sizes are approximately two to one.
            runner = "flagquantum-mc550-cpu" if i % 3 == 0 else "flagquantum-bw1100-cpu"
        entries.append(
            {
                "shard": i,
                "hosted-bucket": i % 8,
                "exclusive": len(nodes) == 1
                and max(
                    hints.get(nodes[0], default),
                    LONG_TEST_TIMEOUT_MINUTES.get(nodes[0], 0) * 60,
                )
                >= EXCLUSIVE_TEST_SECONDS,
                "a800": a800,
                "runner": runner,
                "timeout-minutes": max(
                    DEFAULT_TIMEOUT_MINUTES,
                    sum(LONG_TEST_TIMEOUT_MINUTES.get(node, 0) for node in nodes),
                ),
            }
        )
    matrix = {"include": entries}
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
