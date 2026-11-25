"""Observe native compilation and execution separately, using PF-00 workers.

Run ``python -m benchmarks.synthesis.native_performance --output /tmp/native.json``.
This is local diagnostic evidence, not a release or hardware performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .native_workloads import EXAMPLES, NativeWorkload
from .performance import (
    Mode,
    barrier,
    deadline,
    emit,
    environment,
    high_water,
    integer,
    observe_worker,
)
from .performance_probe import Probe


def worker(name: str, mode: Mode, warmups: int, seconds: float) -> None:
    """Emit the existing observer protocol; verification never enters timers."""
    start = time.perf_counter_ns()
    import torch

    import flagquantum as fq
    from flagquantum.compiler.resource_estimation import estimate_resources

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    emit("imported", import_ns=time.perf_counter_ns() - start)
    payload: dict[str, object] = {}
    try:
        with deadline(seconds):
            workload = NativeWorkload(name)
        first_ns = None
        compile_warmups = []
        for _ in range(warmups):
            barrier("warmup")
            with deadline(seconds):
                start = time.perf_counter_ns()
                warm = workload.build()
                elapsed = time.perf_counter_ns() - start
                compile_warmups.append(elapsed)
                if first_ns is None:
                    first_ns = elapsed
                del warm
        barrier("compile")
        probe = Probe(lambda stage: emit("stage", stage=stage))
        with deadline(seconds):
            start = time.perf_counter_ns()
            if mode == "diagnostic":
                with probe:
                    program = workload.build()
            else:
                program = workload.build()
            elapsed = time.perf_counter_ns() - start
        payload.update(
            e0_ns=elapsed, first_call_ns=first_ns, compile_warmup_ns=compile_warmups
        )
        barrier("compile_end")
        payload["compile_lifetime_high_water"] = high_water()
        if mode == "diagnostic":
            payload["diagnostic"] = probe.report(elapsed)
        # Warm execution separately, with the same retained compiled program.
        execution_warmups = []
        for _ in range(warmups):
            barrier("warmup")
            with deadline(seconds):
                start = time.perf_counter_ns()
                fq.run(program)
                execution_warmups.append(time.perf_counter_ns() - start)
        payload["execution_warmup_ns"] = execution_warmups
        barrier("execution")
        with deadline(seconds):
            start = time.perf_counter_ns()
            result = fq.run(program)
            state = result.statevector().flatten()
            payload["execution_ns"] = time.perf_counter_ns() - start
        barrier("execution_end")
        payload["execution_lifetime_high_water"] = high_water()
        barrier("verification")
        with deadline(60):
            expected = workload.expected(program)
            torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
            payload["verification"] = {
                "scope": "complete complex state including absolute phase and all work wires",
                "amplitudes": state.numel(),
                "maximum_error": float((state - expected).abs().max()),
                "norm": float(state.abs().square().sum()),
                "phase_alignment": False,
            }
        payload.update(
            status="verified",
            circuit_hash=program.circuit.content_hash,
            resources=estimate_resources(program.circuit).to_dict(),
            workspace_qubits=program.report["workspace_qubits"],
        )
    except Exception as error:
        # Process boundary: preserve timeout, numerical and compiler failures.
        payload.update(status="failed", error=f"{type(error).__name__}: {error}")
    emit("result", **payload)


def run(
    output: Path,
    selected: list[str],
    *,
    samples: int = 5,
    warmups: int = 2,
    memory_trials: int = 3,
    seconds: float = 60,
    diagnostic_seconds: float = 180,
) -> None:
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(selected) - EXAMPLES.keys()
    ):
        raise ValueError("select distinct fixed native workloads")
    if (
        samples < 1
        or warmups < 0
        or memory_trials < 1
        or not math.isfinite(seconds)
        or seconds <= 0
        or not math.isfinite(diagnostic_seconds)
        or diagnostic_seconds <= 0
    ):
        raise ValueError(
            "sample counts and deadline must be positive; warmups nonnegative"
        )
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=10
    ).strip()
    # Hash the working sources as well as the base revision; an uncommitted
    # observer change must never be described as the unchanged Git revision.
    sources = sorted(
        p
        for directory in ("flagquantum", "benchmarks/synthesis", "examples/synthesis")
        for p in (root / directory).rglob("*")
        if p.suffix in {".py", ".json"}
    )
    source_hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }
    payload: dict[str, object] = {
        "complete": False,
        "schema": "flagquantum.synthesis.native_performance.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": revision,
        "source_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, timeout=10
            ).strip()
        ),
        "source_hashes": source_hashes,
        "environment": environment(),
        "configuration": {
            "samples": samples,
            "warmups": warmups,
            "memory_trials": memory_trials,
            "deadline_seconds": seconds,
            "diagnostic_deadline_seconds": diagnostic_seconds,
            "scope": "fixed native programs; fresh worker per sample; one thread; CPU complex128",
            "timing": "uninstrumented compilation and warm execution measured separately",
            "memory": "separate sampled compile/execution RSS; lifetime peaks also retained",
            "diagnostics": "intrusive C1-C6 exclusive spans; never substituted for primary timing",
            "claims": "local development baseline; no speedup, scalability or release claim",
        },
        "rows": {},
    }
    rows: dict[str, object] = {}
    failures: list[str] = []
    payload["rows"] = rows
    for name in selected:
        observed = []
        modes: list[Mode] = ["timing" for _ in range(samples)]
        modes.extend("memory" for _ in range(memory_trials))
        modes.append("diagnostic")
        for mode in modes:
            limit = diagnostic_seconds if mode == "diagnostic" else seconds
            command = [
                sys.executable,
                "-m",
                "benchmarks.synthesis.native_performance",
                "--worker",
                name,
                "--mode",
                mode,
                "--warmups",
                str(warmups),
                "--deadline",
                str(limit),
            ]
            result = observe_worker(command, mode, warmups=warmups, seconds=limit)
            observed.append(result)
            print(f"{name}: {mode}: {result.get('status')}", flush=True)
            rows[name] = {"samples": observed}
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2) + "\n")
        good = [r for r in observed if r.get("status") == "verified"]
        signatures = {
            (str(r["circuit_hash"]), json.dumps(r["resources"], sort_keys=True))
            for r in good
        }
        if len(good) != len(observed) or len(signatures) != 1:
            failures.append(name)
            continue
        timings = [r for r in good if r["mode"] == "timing"]
        rows[name] = {
            "samples": observed,
            "summary": {
                key: {
                    "median_ns": statistics.median(integer(r[key]) for r in timings),
                    "min_ns": min(integer(r[key]) for r in timings),
                    "max_ns": max(integer(r[key]) for r in timings),
                }
                for key in ("e0_ns", "execution_ns")
            },
        }
        output.write_text(json.dumps(payload, indent=2) + "\n")

    changed = [
        name
        for name, digest in source_hashes.items()
        if not (root / name).is_file()
        or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
    ]
    payload.update(complete=not failures and not changed, source_changes=changed)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    if changed:
        raise ValueError(
            "source changed during measurement; results are not a frozen baseline"
        )
    if failures:
        raise ValueError(
            f"incomplete verification or non-deterministic circuit: {failures}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/native-performance.json")
    )
    parser.add_argument("--case", action="append", choices=EXAMPLES)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--memory-trials", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--deadline", type=float, default=60)
    parser.add_argument("--diagnostic-deadline", type=float, default=180)
    parser.add_argument("--worker", choices=EXAMPLES)
    parser.add_argument(
        "--mode", choices=("timing", "memory", "diagnostic"), default="timing"
    )
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.mode, args.warmups, args.deadline)
    else:
        run(
            args.output,
            args.case or list(EXAMPLES),
            samples=args.samples,
            memory_trials=args.memory_trials,
            warmups=args.warmups,
            seconds=args.deadline,
            diagnostic_seconds=args.diagnostic_deadline,
        )


if __name__ == "__main__":
    main()
