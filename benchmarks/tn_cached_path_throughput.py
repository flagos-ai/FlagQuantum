"""Measure a warm tensor-network contraction before and after a bookkeeping fix.

A planned contraction path is cached per network profile, so the steady-state
cost of contracting the same network shape twice is what a training loop pays.
This producer measures that cost on operator networks of two sizes, and records
the profile of the contraction so a reader can see which share of the call is
tensor arithmetic and which is planning bookkeeping.

Examples
--------
Write a development artifact payload::

    OMP_NUM_THREADS=1 python benchmarks/tn_cached_path_throughput.py \
        --output artifacts/development/tn_cached_path_bookkeeping_cpu_20261027/summary.json

This is a single-process CPU cost measurement of one internal contraction path.
It establishes no accelerator, distributed, or universal performance claim.
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
from flagquantum.algorithms import Hamiltonian, pauli_term
from flagquantum.simulation.tensor_network.entrypoints import run_tensor_network
from flagquantum.simulation.tensor_network.observables import (
    build_tensor_network_hamiltonian_expectation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

WIDTHS = (8, 12, 14)
DEPTH = 2


def build_ansatz(width: int, depth: int, dtype: torch.dtype):
    """Return one parameterized ansatz builder over ``width`` qubits."""

    def build(parameters: torch.Tensor) -> fq.Circuit:
        angles = parameters.reshape(depth, width, 2)
        circuit = fq.Circuit(width, dtype=dtype)
        for layer in range(depth):
            for qubit in range(width):
                circuit.ry(qubit, angles[layer, qubit, 0])
                circuit.rz(qubit, angles[layer, qubit, 1])
            for qubit in range(width - 1):
                circuit.cx(qubit, qubit + 1)
        return circuit

    return build


def operator_plan(width: int, depth: int):
    """Return a warm operator network whose path is already cached."""

    generator = torch.Generator().manual_seed(260720)
    initial = 0.1 * torch.randn(depth, width, 2, generator=generator)
    circuit = build_ansatz(width, depth, torch.complex128)(initial.reshape(-1))
    state = run_tensor_network(circuit, dense_observable_qubits=12)
    hamiltonian = Hamiltonian(
        [pauli_term(-0.5, "Z", (qubit,)) for qubit in range(width)]
        + [pauli_term(-0.7, "X", (qubit,)) for qubit in range(width)]
    )
    plan = build_tensor_network_hamiltonian_expectation(state.plan, hamiltonian)
    plan.contract()
    return plan


def summarize(samples: list[float]) -> dict[str, Any]:
    """Report the raw samples beside a median and its absolute deviation."""

    median = statistics.median(samples)
    return {
        "samples_seconds": [round(value, 6) for value in samples],
        "median_seconds": round(median, 6),
        "median_absolute_deviation_seconds": round(
            statistics.median([abs(value - median) for value in samples]), 6
        ),
        "minimum_seconds": round(min(samples), 6),
    }


def measure(width: int, repeats: int) -> dict[str, Any]:
    """Measure one warm operator network and record its plan shape."""

    plan = operator_plan(width, DEPTH)
    value = torch.real(plan.contract())
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        torch.real(plan.contract())
        samples.append(time.perf_counter() - started)
    profile = plan.contraction_profile("greedy")
    return {
        "width": width,
        "nodes": len(plan.nodes),
        "terms": 2 * width,
        "contraction_steps": profile.n_steps,
        "peak_size": profile.peak_size,
        "value": float(value),
        "warm": summarize(samples),
        "seconds_per_pair_step": round(
            statistics.median(samples) / max(profile.n_steps, 1), 9
        ),
    }


def git_revision(*arguments: str) -> str:
    """Return one git revision or an empty string when it is unavailable."""

    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, text=True, timeout=15
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=list(WIDTHS))
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.manual_seed(260720)

    started = time.perf_counter()
    payload = {
        "schema": "flagquantum.tn_cached_path_throughput.v1",
        "benchmark": "tn_cached_path_throughput",
        "artifact_class": "measured_local_run",
        "claim": (
            "a warm tensor-network contraction pays its label bookkeeping once "
            "per call rather than once per pair step"
        ),
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "non_release_evidence": True,
        "benchmark_evidence_class": "local_non_release",
        "scalability_blockers": [
            "single-process CPU measurement of one internal contraction path; it "
            "shards no workload and is not distributed or accelerator evidence"
        ],
        "workload": {
            "network": "operator network for a transverse-field Pauli sum, ket and bra copies",
            "hamiltonian": "Z on every qubit at -0.5, X on every qubit at -0.7",
            "ansatz": "depth x (ry, rz on every qubit, cx chain)",
            "dtype": "complex128 amplitudes",
            "depth": DEPTH,
            "path": "greedy contraction order, warmed before timing",
        },
        "cells": [measure(width, arguments.repeats) for width in arguments.widths],
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
                "medians_ms": {
                    cell["width"]: round(cell["warm"]["median_seconds"] * 1e3, 3)
                    for cell in payload["cells"]
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
