"""Measure what bind-time angle validation costs a public ``Module`` forward.

A compiled builder is traced, so its gate angles are tracers when the canonical
finite-angle rule runs and the magnitude check is deferred. The values arrive
when the reusable circuit is rebound, and applying the rule there costs
something on every call. This producer measures that cost against the same code
with the check removed, in the same process, so the difference is the check and
not a different build.

Examples
--------
Write a development artifact payload::

    OMP_NUM_THREADS=1 python benchmarks/module_bind_validation.py \
        --output artifacts/development/module_bind_validation_cpu_20261027/summary.json

This is a single-process CPU cost measurement of one validation step. It
establishes no accelerator, distributed, or universal performance claim.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

import flagquantum as fq
from flagquantum.runtime.module import Module

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_ansatz(width: int, depth: int):
    """Return a depth-``depth`` ladder ansatz over ``width`` qubits."""

    def build(parameters: torch.Tensor) -> fq.Circuit:
        angles = parameters.reshape(depth, width, 2)
        circuit = fq.Circuit(width, dtype=torch.complex128)
        for layer in range(depth):
            for qubit in range(width):
                circuit.ry(qubit, angles[layer, qubit, 0])
                circuit.rz(qubit, angles[layer, qubit, 1])
            for qubit in range(width - 1):
                circuit.cx(qubit, qubit + 1)
        return circuit

    return build


def build_module(width: int, depth: int, mode: str) -> Module:
    count = depth * width * 2
    return fq.Module(
        build_ansatz(width, depth),
        count,
        init=torch.linspace(-0.3, 0.3, count),
        dtype=torch.float64,
        policy=fq.RuntimePolicy(execution_options=fq.ExecutionOptions(mode=mode)),
    )


def timed(module: Module, repeats: int) -> list[float]:
    """Return repeated elapsed seconds for one module forward."""

    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        module()
        samples.append(time.perf_counter() - started)
    return samples


def summarize(samples: list[float]) -> dict[str, Any]:
    median = statistics.median(samples)
    return {
        "samples_seconds": [round(value, 7) for value in samples],
        "median_seconds": round(median, 7),
        "median_absolute_deviation_seconds": round(
            statistics.median([abs(value - median) for value in samples]), 7
        ),
    }


def measure(
    width: int, depth: int, mode: str, repeats: int, blocks: int
) -> dict[str, Any]:
    """Time one shape with the check and without it, alternating whole blocks.

    The two arms are interleaved rather than run one after the other: a single
    ordered pair puts process warm-up and cache drift into the difference, which
    showed up as a negative overhead when the second arm happened to be faster.
    Alternating blocks and comparing medians of block medians leaves only the
    check.
    """

    module = build_module(width, depth, mode)
    module()

    original = Module._refuse_non_finite_bound_parameters
    # Attribute lookup on a staticmethod returns the plain function, so both the
    # no-op and the restore have to re-wrap it, or the class would start binding
    # ``self`` into a one-argument function.
    unchecked_body = staticmethod(lambda circuit: None)

    checked_blocks: list[float] = []
    unchecked_blocks: list[float] = []
    for _ in range(blocks):
        checked_blocks.append(statistics.median(timed(module, repeats)))
        Module._refuse_non_finite_bound_parameters = unchecked_body
        try:
            unchecked_blocks.append(statistics.median(timed(module, repeats)))
        finally:
            Module._refuse_non_finite_bound_parameters = staticmethod(original)

    with_check = statistics.median(checked_blocks)
    without_check = statistics.median(unchecked_blocks)
    gate_count = len(module._builder_bound_circuit._instructions)
    return {
        "width": width,
        "depth": depth,
        "mode": mode,
        "gate_count": gate_count,
        "with_validation": {
            "block_medians_seconds": [round(value, 7) for value in checked_blocks],
            "median_seconds": round(with_check, 7),
        },
        "without_validation": {
            "block_medians_seconds": [round(value, 7) for value in unchecked_blocks],
            "median_seconds": round(without_check, 7),
        },
        "overhead_seconds": round(with_check - without_check, 7),
        "overhead_fraction": (
            round((with_check - without_check) / without_check, 6)
            if without_check
            else None
        ),
        "nanoseconds_per_gate": round(
            (with_check - without_check) / max(gate_count, 1) * 1e9, 1
        ),
    }


def git_revision(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, text=True, timeout=15
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="2:1,8:4,12:4")
    parser.add_argument("--modes", default="statevector")
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument(
        "--blocks",
        type=int,
        default=7,
        help="number of alternating checked/unchecked blocks per shape",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    shapes = tuple(
        tuple(int(part) for part in item.split(":"))
        for item in arguments.shapes.split(",")
    )
    modes = tuple(item.strip() for item in arguments.modes.split(","))

    started = time.perf_counter()
    payload = {
        "schema": "flagquantum.module_bind_validation.v1",
        "benchmark": "module_bind_validation",
        "artifact_class": "measured_local_run",
        "claim": (
            "the bind-time finite-angle check costs a measured fraction of one "
            "public module forward, and refuses the values the canonical rule "
            "defers during tracing"
        ),
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "non_release_evidence": True,
        "benchmark_evidence_class": "local_non_release",
        "scalability_blockers": [
            "single-process CPU cost measurement of one validation step; it "
            "shards no workload and is not distributed or accelerator evidence"
        ],
        "method": (
            "the same module is timed twice in one process: once as shipped, and "
            "once with Module._refuse_non_finite_bound_parameters replaced by a "
            "no-op, so the difference is the check rather than a different build"
        ),
        "cells": [
            measure(width, depth, mode, arguments.repeats, arguments.blocks)
            for mode in modes
            for width, depth in shapes
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "command": " ".join(sys.argv),
        "threads": {
            name: os.environ.get(name, "")
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "base_commit": git_revision("rev-parse", "origin/main"),
        "source_commit": git_revision("rev-parse", "HEAD"),
        "measuring_source_committed": not git_revision(
            "status", "--porcelain", "--", str(Path(__file__).resolve())
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "overhead_fraction": {
                    f"{cell['width']}x{cell['depth']}": cell["overhead_fraction"]
                    for cell in payload["cells"]
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
