"""Observe the existing Surface memory pipeline without performance gates.

One invocation measures one fixed-shot case in one process. RSS includes imports,
warmups and all repetitions; it is not exclusive sampler memory. Each repetition
restarts seed 53, so repeated counts are not independent statistical trials.
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
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict, TypeVar

import torch

from flagquantum import __version__
from flagquantum.qec import (
    DetectorErrorModel,
    PhenomenologicalNoise,
    PyMatchingDecoder,
    RotatedSurfaceCode,
    build_memory_circuit,
    compile_memory_sampler,
)

from .contract import runtime_metadata, write_json_atomic

_T = TypeVar("_T")


class _Observation(TypedDict):
    warmup: bool
    completed_shots: int
    logical_failures: int
    observables: int
    detectors: int
    batches: int
    phases_seconds: dict[str, float]
    pipeline_seconds_excluding_imports_and_report_io: float
    pipeline_shots_per_second: float
    visible_batch_tensor_bytes_max: int


def _timed(function: Callable[[], _T]) -> tuple[_T, float]:
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    scale = 1 if sys.platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale


def _run_once(*, shots: int, batch_size: int) -> _Observation:
    started = time.perf_counter()
    phases: dict[str, float] = {}
    memory, phases["memory_construction"] = _timed(
        lambda: build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=3)
    )
    noise = PhenomenologicalNoise(
        data_flip=0.005, phase_flip=0.007, measurement_flip=0.009
    )
    model, phases["dem_construction"] = _timed(
        lambda: DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    )
    decoder, phases["decoder_setup"] = _timed(
        lambda: PyMatchingDecoder.from_detector_error_model(
            model.merge_duplicate_mechanisms()
        )
    )
    sampler, phases["sampler_compile"] = _timed(
        lambda: compile_memory_sampler(memory, noise=noise, seed=53)
    )
    phases.update(sampling=0.0, decoding=0.0, count_reduction=0.0)
    completed = failures = batches = visible_bytes = 0
    with sampler as active_sampler:
        while completed < shots:
            requested = min(batch_size, shots - completed)
            sample_start = time.perf_counter()
            sample = active_sampler.sample(shots=requested)
            phases["sampling"] += time.perf_counter() - sample_start
            decode_start = time.perf_counter()
            predicted = decoder.decode_batch(sample.detectors)
            phases["decoding"] += time.perf_counter() - decode_start
            count_start = time.perf_counter()
            if sample.shots != requested or predicted.shape != sample.observables.shape:
                raise ValueError(
                    "sampling/decoding returned an incomplete logical batch"
                )
            failures += int((predicted != sample.observables).any(dim=1).sum())
            completed += sample.shots
            phases["count_reduction"] += time.perf_counter() - count_start
            visible_bytes = max(
                visible_bytes,
                sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in (sample.detectors, sample.observables, predicted)
                ),
            )
            batches += 1
            del sample, predicted
    wall = time.perf_counter() - started
    return {
        "warmup": False,
        "completed_shots": completed,
        "logical_failures": failures,
        "observables": model.num_observables,
        "detectors": model.num_detectors,
        "batches": batches,
        "phases_seconds": phases,
        "pipeline_seconds_excluding_imports_and_report_io": wall,
        "pipeline_shots_per_second": completed / wall,
        "visible_batch_tensor_bytes_max": visible_bytes,
    }


def run_benchmark(
    *, shots: int = 128, batch_size: int = 32, warmup: int = 3, iterations: int = 10
) -> dict[str, object]:
    """Measure a bounded CPU baseline; propagate failures without a partial report."""
    for name, value, lower, upper in (
        ("shots", shots, 1, 4096),
        ("batch_size", batch_size, 1, 128),
        ("warmup", warmup, 0, 3),
        ("iterations", iterations, 1, 10),
    ):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("this runner requires one local process")
    before = _peak_rss_bytes()
    observations: list[_Observation] = []
    for index in range(warmup + iterations):
        row = _run_once(shots=shots, batch_size=batch_size)
        row["warmup"] = index < warmup
        observations.append(row)
    after = _peak_rss_bytes()
    root = Path(__file__).resolve().parents[2]
    source: dict[str, object] = {"source_revision": None, "tracked_tree_clean": None}
    if (root / ".git").exists():
        source = {
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=10
            ).strip(),
            "tracked_tree_clean": not subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root,
                text=True,
                timeout=10,
            ).strip(),
        }
    return {
        **runtime_metadata(
            runner="qec_memory_sampling",
            schema="flagquantum.qec_memory_sampling.local_observations.v1",
        ),
        **source,
        "versions": {
            "flagquantum_source": __version__,
            "flagquantum_distribution": version("flagquantum"),
            "torch": torch.__version__,
            "stim": version("stim"),
            "pymatching": version("pymatching"),
        },
        "torch_threads": torch.get_num_threads(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "precision": "CPU int8 measurement records and bool logical predictions",
        "device": "cpu",
        "distribution_semantics": "single_process",
        "local_world_size": 1,
        "node_count": 1,
        "rank_ownership": {"0": "whole_workload"},
        "communication_bytes": 0,
        "non_release_evidence": True,
        "benchmark_evidence_class": "local_non_release",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "blockers": ["local CPU observation; no hardware or capacity certification"],
        "workload": {
            "code": "RotatedSurfaceCode",
            "distance": 3,
            "rounds": 3,
            "readout_basis": "z",
            "noise": {
                "data_flip": 0.005,
                "phase_flip": 0.007,
                "measurement_flip": 0.009,
            },
            "seed": 53,
            "shots": shots,
            "batch_size": batch_size,
        },
        "methodology": {
            "warmup": warmup,
            "iterations": iterations,
            "seed_restarted_each_repetition": True,
            "repetitions_are_independent_statistical_trials": False,
            "pipeline_includes_handle_close": True,
        },
        "process_max_rss_before_bytes": before,
        "process_max_rss_after_bytes": after,
        "rss_available": after is not None,
        "observations": observations,
        "pipeline_median_seconds": statistics.median(
            row["pipeline_seconds_excluding_imports_and_report_io"]
            for row in observations[warmup:]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(
        shots=args.shots,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    if args.json_output is not None:
        write_json_atomic(args.json_output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
