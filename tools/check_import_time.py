#!/usr/bin/env python3
"""Protect the lazy core import-time budget."""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time


def sample_imports(repetitions: int) -> tuple[float, ...]:
    values = []
    code = "import flagquantum; assert 'jax' not in __import__('sys').modules"
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
