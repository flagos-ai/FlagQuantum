#!/usr/bin/env python3
"""Protect the lazy core import-time budget."""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def forbidden_imports() -> tuple[str, ...]:
    policy = tomllib.loads(
        (ROOT / "dependency-policy.toml").read_text(encoding="utf-8")
    )
    return tuple(policy["import_policy"]["core_forbidden_imports"])


def probe_source(forbidden: tuple[str, ...]) -> str:
    """Build the child program that times one import and reports module leaks.

    The child times its own import rather than letting the parent time the
    subprocess. On this host CPython startup costs about 0.070s while the import
    itself costs 0.018-0.089s, so a wall-clock measurement spends most of its
    budget on a constant that no change to this repository can move, and that
    moves with machine load instead. The child reports the import through
    ``stdout`` and any dependency-policy violation through ``stderr`` and a
    distinct exit code, so a leak cannot be mistaken for a slow import.
    """

    return f"""
import sys
import time

forbidden = {forbidden!r}

started = time.perf_counter()
import flagquantum
elapsed = time.perf_counter() - started

loaded = sorted(
    name
    for name in sys.modules
    if any(name == root or name.startswith(root + ".") for root in forbidden)
)
if loaded:
    print("loaded: " + ", ".join(loaded), file=sys.stderr)
    raise SystemExit(2)
print(f"{{elapsed!r}}")
"""


def sample_imports(repetitions: int) -> tuple[float, ...]:
    """Return how long ``import flagquantum`` took, once per repetition."""

    code = probe_source(forbidden_imports())
    values = []
    for _ in range(repetitions):
        completed = subprocess.run(
            (sys.executable, "-c", code), capture_output=True, text=True
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "importing flagquantum failed or loaded a module the dependency "
                f"policy forbids:\n{completed.stderr.strip()}"
            )
        values.append(float(completed.stdout.strip()))
    return tuple(values)


def gated_statistic(samples: tuple[float, ...]) -> float:
    """Return the fastest import, which is the number the budget is compared to.

    The budget answers whether this repository can still be imported inside it.
    A measurement of that is bounded below by the code being measured and
    inflated above it by whatever else the machine is doing, so the floor is the
    estimate that survives a loaded host, and the rest of the distribution is
    reported for a human rather than gated on.
    """

    return min(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.20,
        help=(
            "Budget for the fastest observed import. The floor is the gated "
            "statistic because the budget answers whether this repository can "
            "still be imported inside it, and that stays true under load while "
            "the mean and the median do not."
        ),
    )
    args = parser.parse_args()
    samples = sample_imports(args.repetitions)
    fastest = gated_statistic(samples)
    print(
        f"flagquantum import fastest={fastest:.6f}s "
        f"median={statistics.median(samples):.6f}s "
        f"slowest={max(samples):.6f}s "
        f"over {len(samples)} runs"
    )
    if fastest <= args.max_seconds:
        return 0
    print(
        f"import budget exceeded: the fastest of {len(samples)} runs took "
        f"{fastest:.3f}s against a {args.max_seconds:.2f}s budget",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
