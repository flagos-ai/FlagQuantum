# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Kickmix py/perf/run_all_perf.py at
# 9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453.

"""Discover and run the native physical Python benchmark suites."""

import glob
import importlib.util
from pathlib import Path

from flagquantum.benchmarking._timing_suite import TimingSuite


def run_directory(directory: Path) -> None:
    for path in glob.glob(str(directory) + "/physical_*_perf.py", recursive=True):
        spec = importlib.util.spec_from_file_location("_physical_perf", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        benchmark = getattr(module, "benchmark", None)
        if isinstance(benchmark, TimingSuite):
            benchmark.main()


def main() -> None:
    run_directory(Path(__file__).parent)


if __name__ == "__main__":
    main()
