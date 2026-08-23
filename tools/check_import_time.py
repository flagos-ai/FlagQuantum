#!/usr/bin/env python3
"""Protect the lazy core import-time budget."""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
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


def sample_imports(repetitions: int) -> tuple[float, ...]:
    values = []
    forbidden = forbidden_imports()
    code = f"""
import sys
import flagquantum
forbidden = {forbidden!r}
loaded = sorted(
    name for name in sys.modules
    if any(name == root or name.startswith(root + '.') for root in forbidden)
)
assert not loaded, loaded
"""
    for _ in range(repetitions):
        started = time.perf_counter()
        subprocess.run((sys.executable, "-c", code), check=True)
        values.append(time.perf_counter() - started)
    return tuple(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--max-median-seconds", type=float, default=0.20)
    args = parser.parse_args()
    samples = sample_imports(args.repetitions)
    median = statistics.median(samples)
    print(f"flagquantum import median={median:.6f}s samples={samples}")
    return 0 if median <= args.max_median_seconds else 1


if __name__ == "__main__":
    raise SystemExit(main())
