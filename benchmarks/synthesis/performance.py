"""PF-00 fresh-worker E0, diagnostic spans, cold import and OS RSS baseline.

Run ``python -m benchmarks.synthesis.performance --output /tmp/pf00.json``.
All instrumentation lives in benchmark workers; production behavior is unchanged.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import select
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from flagquantum.services.synthesis import Program

from .performance_probe import Probe, process_rss
from .performance_workloads import Prepared, Workload, workloads

Mode = Literal["timing", "diagnostic", "memory", "cold"]
THREADS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class CallTimeoutError(TimeoutError):
    pass


def integer(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("expected nonnegative integer measurement")
    return value


def emit(event: str, **values: object) -> None:
    print(json.dumps({"event": event, **values}), flush=True)


def barrier(phase: str) -> None:
    emit("phase", phase=phase)
    if sys.stdin.readline() != "continue\n":
        raise RuntimeError("measurement controller disconnected")


@contextmanager
def deadline(seconds: float) -> Iterator[None]:
    def expired(signum: int, frame: object) -> None:
        raise CallTimeoutError("per-call wall deadline exceeded")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def high_water() -> dict[str, object]:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit = "bytes" if sys.platform == "darwin" else "KiB"
    return {
        "raw": raw,
        "raw_unit": unit,
        "bytes": int(raw if unit == "bytes" else raw * 1024),
        "scope": "worker lifetime including import/preparation/warmups; never subtracted",
    }


def worker(workload: Workload, mode: Mode, warmups: int, seconds: float) -> None:
    # The parent times interpreter startup; this span measures dependency imports.
    start = time.perf_counter_ns()
    import torch

    from flagquantum import synth

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    emit("imported", import_ns=time.perf_counter_ns() - start)
    if mode == "cold":
        emit("result", status="cold", lifetime_high_water=high_water())
        return
    prepared: Prepared | None = None
    payload: dict[str, object] = {"mode": mode, "gc_enabled": gc.isenabled()}
    try:
        setup_start = time.perf_counter_ns()
        with deadline(seconds):
            prepared = Prepared(workload)
        payload["source_preparation_ns"] = time.perf_counter_ns() - setup_start
        payload.update(source=prepared.source, source_sha256=prepared.source_sha256)
        first_ns = None
        warmup_ns = []
        refused_codes: list[str] = []

        def compile_once() -> Program | None:
            assert prepared is not None
            nonlocal refused_codes
            try:
                with deadline(seconds):
                    return prepared.compile()
            except synth.SynthesisError as error:
                refused_codes = [d.code for d in error.diagnostics]
                if not workload.refused:
                    raise
                return None

        for index in range(warmups):
            barrier("warmup")
            start = time.perf_counter_ns()
            program = compile_once()
            warmup_ns.append(prepared.last_e0_ns)
            if index == 0:
                first_ns = warmup_ns[-1]
            del program
        barrier("compile")
        start = time.perf_counter_ns()
        if mode == "diagnostic":
            probe = Probe(lambda stage: emit("stage", stage=stage))
            with probe:
                program = compile_once()
            elapsed = prepared.last_e0_ns
        else:
            program = compile_once()
            elapsed = prepared.last_e0_ns
        payload["observer_wall_ns"] = time.perf_counter_ns() - start
        payload.update(
            e0_ns=elapsed,
            first_call_ns=first_ns if first_ns is not None else elapsed,
            warmup_ns=warmup_ns,
        )
        barrier("compile_end")
        # Capture before verification/simulation can raise the OS high-water mark.
        payload["compile_lifetime_high_water"] = high_water()
        if mode == "diagnostic":
            barrier("diagnostic_report")
            report_start = time.perf_counter_ns()
            payload["diagnostic"] = probe.report(elapsed)
            payload["diagnostic_report_ns"] = time.perf_counter_ns() - report_start
        barrier("verification")
        if program is None:
            if not workload.refused or not refused_codes:
                raise AssertionError("missing expected budget refusal")
            payload.update(
                status="resource_refused",
                diagnostic_codes=refused_codes,
                verification={"expected_refusal": True},
            )
        else:
            if workload.refused:
                raise AssertionError(
                    "request exceeded one-qubit budget without refusal"
                )
            with deadline(60):
                verified = prepared.verify(program)
            e1 = verified.get("e1")
            if isinstance(e1, dict) and e1.get("status") == "verified":
                e1["e0_plus_run_plus_read_ns"] = (
                    elapsed + int(e1["run_ns"]) + int(e1["named_read_ns"])
                )
            payload.update(status="verified", verification=verified)
    except CallTimeoutError as error:
        payload.update(status="timeout", error=str(error))
    except NotImplementedError as error:
        payload.update(status="unsupported", error=repr(error))
    except KeyboardInterrupt:
        payload.update(status="cancelled")
    except (AssertionError, ValueError) as error:
        payload.update(status="validation_failed", error=repr(error))
    except Exception as error:
        # Worker isolation boundary: preserve a failed sample, never claim success.
        payload.update(status="measurement_failed", error=repr(error))
    finally:
        if prepared is not None:
            prepared.close()
    emit("result", **payload)


def terminate_group(process: subprocess.Popen[str]) -> bool:
    """Kill and reap the entire isolated worker group, including descendants."""
    try:
        process.wait(timeout=0.2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
    # ps also handles the macOS race where killpg returns EPERM for an exited group.
    groups = subprocess.run(
        ["ps", "-axo", "pgid="], capture_output=True, text=True, check=True, timeout=2
    )
    remaining = process.pid in {
        int(line) for line in groups.stdout.splitlines() if line.strip()
    }
    if remaining:
        os.killpg(process.pid, signal.SIGKILL)
        for _ in range(20):
            groups = subprocess.run(
                ["ps", "-axo", "pgid="],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            if process.pid not in {
                int(line) for line in groups.stdout.splitlines() if line.strip()
            }:
                return True
            time.sleep(0.01)
    return not remaining


def trial(
    workload: Workload, mode: Mode, *, warmups: int = 2, seconds: float = 30
) -> dict[str, object]:
    env = os.environ.copy()
    env.update(dict.fromkeys(THREADS, "1"))
    env["PYTHONHASHSEED"] = "0"
    start = time.perf_counter_ns()
    points: list[dict[str, object]] = []
    samples: dict[str, object] = {}
    phase = "startup"
    stage = "unattributed"
    phase_start = time.monotonic()
    overhead_ns = 0
    transitions: list[dict[str, object]] = []
    with tempfile.TemporaryFile(mode="w+t") as errors:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                __spec__.name if __spec__ else "benchmarks.synthesis.performance",
                "--worker",
                json.dumps(asdict(workload)),
                "--mode",
                mode,
                "--warmups",
                str(warmups),
                "--deadline",
                str(seconds),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            env=env,
            start_new_session=True,
        )
        assert process.stdin is not None and process.stdout is not None
        try:
            # Requests + source setup + import + verification/cleanup allowances.
            overall = time.monotonic() + (warmups + 3) * seconds + 80
            buffer = ""
            next_sample_ns = 0
            last_heartbeat = time.monotonic()
            while time.monotonic() < overall:
                readable, _, _ = select.select(
                    [process.stdout], [], [], 0 if "\n" in buffer else 0.015
                )
                if readable:
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk and not buffer:
                        break
                    buffer += chunk.decode("utf-8")
                if "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    data = json.loads(line)
                    event = data.pop("event")
                    if event == "imported":
                        phase = "source_preparation"
                        phase_start = time.monotonic()
                        samples.update(
                            cold_start_to_import_ns=time.perf_counter_ns() - start,
                            **data,
                        )
                    elif event == "stage":
                        stage = data["stage"]
                    elif event == "phase":
                        phase = data["phase"]
                        transitions.append(
                            {"phase": phase, "timestamp_ns": time.perf_counter_ns()}
                        )
                        phase_start = time.monotonic()
                        stage = "unattributed"
                        if phase == "compile" and mode in {"memory", "diagnostic"}:
                            point = process_rss(process.pid)
                            samples["window_start_rss_bytes"] = point[
                                "parent_rss_bytes"
                            ]
                            points.append(
                                {
                                    "timestamp_ns": time.perf_counter_ns(),
                                    "phase": "compile",
                                    "stage": "unattributed",
                                    **point,
                                }
                            )
                        if phase == "compile_end" and mode in {"memory", "diagnostic"}:
                            point = process_rss(process.pid)
                            samples["window_end_rss_bytes"] = point["parent_rss_bytes"]
                            points.append(
                                {
                                    "timestamp_ns": time.perf_counter_ns(),
                                    "phase": "compile",
                                    "stage": stage,
                                    **point,
                                }
                            )
                        process.stdin.write("continue\n")
                        process.stdin.flush()
                    elif event == "result":
                        samples.update(data)
                        break
                if (
                    mode in {"memory", "diagnostic"}
                    and phase in {"startup", "warmup", "compile"}
                    and process.poll() is None
                    and time.perf_counter_ns() >= next_sample_ns
                ):
                    tick = time.perf_counter_ns()
                    point = process_rss(process.pid)
                    points.append(
                        {"timestamp_ns": tick, "phase": phase, "stage": stage, **point}
                    )
                    overhead_ns += time.perf_counter_ns() - tick
                    next_sample_ns = tick + 15_000_000
                if time.monotonic() - last_heartbeat >= 5:
                    print(
                        f'worker={process.pid} cohort={mode} phase={phase} last_stage={stage} rss={points[-1]["parent_rss_bytes"] if points else "not_sampled"}',
                        flush=True,
                    )
                    last_heartbeat = time.monotonic()
                if time.monotonic() - phase_start > (
                    60 if phase == "verification" else seconds + 5
                ):
                    samples.update(
                        status="timeout", error="controller phase deadline: " + phase
                    )
                    break
            if "status" not in samples:
                samples.update(
                    status="measurement_failed", error="worker exited without a receipt"
                )
        except KeyboardInterrupt:
            samples.update(status="cancelled")
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            samples.update(status="measurement_failed", error=repr(error))
        finally:
            try:
                cleanup = terminate_group(process)
            except (OSError, subprocess.SubprocessError) as error:
                cleanup = False
                samples.update(
                    status="measurement_failed", error="cleanup: " + repr(error)
                )
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
            process.stdin.close()
            process.stdout.close()
            errors.seek(0)
            samples["stderr"] = errors.read()[-4000:]
    samples.update(
        mode=mode,
        worker_pid=process.pid,
        process_group_reaped=cleanup,
        wall_ns=time.perf_counter_ns() - start,
        phase_transitions=transitions,
    )
    if not cleanup:
        samples.update(status="measurement_failed", error="worker group remains alive")
    if mode in {"memory", "diagnostic"}:
        window = [p for p in points if p["phase"] == "compile"]
        samples["memory"] = {
            "scope": "externally sampled worker/tree RSS; verification and simulation excluded from compile window",
            "points": points,
            "requested_interval_ns": 15_000_000,
            "sampler_wall_ns": overhead_ns,
            "window_sampled_peak_rss_bytes": max(
                (integer(p["parent_rss_bytes"]) for p in window), default=None
            ),
            "window_sampled_peak_tree_rss_sum_bytes": max(
                (integer(p["process_tree_rss_sum_bytes"]) for p in window), default=None
            ),
            "stage_sampled_peaks_bytes": {
                s: max(
                    integer(p["parent_rss_bytes"]) for p in window if p["stage"] == s
                )
                for s in {str(p["stage"]) for p in window}
            },
            "resolution_ns": [
                integer(b["timestamp_ns"]) - integer(a["timestamp_ns"])
                for a, b in zip(points, points[1:], strict=False)
            ],
            "limit": "sampled peaks are lower-bound observations; tree RSS can double-count shared pages; diagnostic stage peaks include profiler overhead; primary memory mode is unprofiled",
        }
    return samples


def summary(samples: list[dict[str, object]], mode: Mode) -> dict[str, object]:
    accepted = [
        s
        for s in samples
        if s["mode"] == mode and s["status"] in {"verified", "resource_refused", "cold"}
    ]
    key = "cold_start_to_import_ns" if mode == "cold" else "e0_ns"
    values = [integer(s[key]) for s in accepted if key in s]
    return {
        "statuses": dict(
            Counter(str(s["status"]) for s in samples if s["mode"] == mode)
        ),
        "sample_count": len(values),
        "median_ns": statistics.median(values) if values else None,
        "min_ns": min(values) if values else None,
        "max_ns": max(values) if values else None,
    }


def environment() -> dict[str, object]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_model": (
            subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
            ).strip()
            if sys.platform == "darwin"
            else next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "unknown",
            )
        ),
        "logical_cpus": os.cpu_count(),
        "load_average": os.getloadavg(),
        "affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "threads": dict.fromkeys(THREADS, "1"),
        "compile_workers": 1,
        "gc_enabled": gc.isenabled(),
        "gc_thresholds": gc.get_threshold(),
        "host_isolation": "not exclusive; load average recorded; CPU frequency and thermal policy unmeasured",
        "dependencies": dict(
            sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            )
        ),
        "quota": (
            Path("/sys/fs/cgroup/cpu.max").read_text().strip()
            if Path("/sys/fs/cgroup/cpu.max").exists()
            else "not exposed; no cgroup quota measured"
        ),
    }


def run(
    output: Path,
    selected: list[str],
    samples: int,
    warmups: int,
    seconds: float,
    memory_trials: int,
    *,
    production_revision: str = "103250df54a0cf7204a8057baae5316ae0a4144f",
) -> None:
    rows = workloads()
    valid_names = {w.identifier for w in rows} | {w.case for w in rows}
    if set(selected) - valid_names:
        raise ValueError("unknown fixed workload selection")
    if selected:
        rows = tuple(w for w in rows if w.identifier in selected or w.case in selected)
        if not rows:
            raise ValueError("no matching fixed workload")
    root = Path(__file__).resolve().parents[2]
    if len(production_revision) != 40 or any(
        c not in "0123456789abcdef" for c in production_revision
    ):
        raise ValueError("pin the production reference with a full commit SHA")
    production_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD:flagquantum"], cwd=root, text=True, timeout=15
    ).strip()
    accepted_tree = subprocess.check_output(
        ["git", "rev-parse", production_revision + ":flagquantum"],
        cwd=root,
        text=True,
        timeout=15,
    ).strip()
    if production_tree != accepted_tree:
        raise ValueError(
            "measurement requires the explicitly pinned production tree; label algorithm changes separately"
        )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=15
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True, timeout=15
    )
    payload: dict[str, object] = {
        "schema": "flagquantum.synthesis.python_performance.v1",
        "source_revision": revision,
        "production_baseline_revision": production_revision,
        "production_tree": production_tree,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dirty": bool(dirty),
        "environment": environment(),
        "configuration": {
            "warmups_per_fresh_worker": warmups,
            "formal_samples": samples,
            "memory_trials": memory_trials,
            "diagnostic_trials": 1,
            "cold_trials": samples,
            "per_compile_deadline_seconds": seconds,
            "verification_deadline_seconds": 60,
            "worker_deadline_seconds": (warmups + 3) * seconds + 80,
            "max_qubits": 256,
            "max_operations": 100000,
            "memory_limit_bytes": 16 << 256,
            "objective": "width",
            "seed": 20261102,
            "target_profiles": "logical, PHASE_BASIS, PHASE_BASIS plus line; per workload inventory",
            "profile_enabled_in_primary_timing": False,
            "profile_enabled_in_primary_memory": False,
            "cache_scope": "fresh process per formal trial; two identical warmups; no persistent-worker cohort",
            "claims": "Python P0 observation only; no speedup, Rust, hardware or release claim",
        },
        "source_hashes": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "benchmarks/synthesis").glob("*.py"))
        },
        "original_resource_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "benchmarks/synthesis/baselines").glob("ao*.json"))
        },
        "fixed_workload_inventory": [asdict(w) for w in workloads()],
        "not_selected": [w.identifier for w in workloads() if w not in rows],
        "cold": [],
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)

    cold = [
        trial(rows[0], "cold", warmups=warmups, seconds=seconds) for _ in range(samples)
    ]
    payload["cold"] = cold
    recorded: list[dict[str, object]] = []
    payload["rows"] = recorded
    for position, workload in enumerate(rows):
        raw: list[dict[str, object]] = []
        row: dict[str, object] = {
            "workload": asdict(workload),
            "id": workload.identifier,
            "raw": raw,
        }
        recorded.append(row)
        cohorts: tuple[tuple[Mode, int], ...] = (
            ("timing", samples),
            ("memory", memory_trials),
            ("diagnostic", 1),
        )
        for mode, count in cohorts:
            for repetition in range(count):
                print(
                    f"trial {position+1}/{len(rows)} {workload.identifier} {mode} {repetition+1}/{count}",
                    flush=True,
                )
                raw.append(trial(workload, mode, warmups=warmups, seconds=seconds))
                row.update(
                    timing=summary(raw, "timing"),
                    memory_statuses=dict(
                        Counter(str(s["status"]) for s in raw if s["mode"] == "memory")
                    ),
                )
                save()
                if raw[-1]["status"] == "cancelled":
                    raise KeyboardInterrupt
                if raw[-1]["process_group_reaped"] is not True:
                    raise RuntimeError(
                        "worker cleanup failed; stop before another trial"
                    )
        print(
            f'{position+1}/{len(rows)} {workload.identifier} {row["timing"]}',
            flush=True,
        )
    save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--memory-trials", type=int, default=1)
    parser.add_argument("--deadline", type=float, default=30)
    parser.add_argument(
        "--production-revision", default="103250df54a0cf7204a8057baae5316ae0a4144f"
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode", choices=("timing", "diagnostic", "memory", "cold"), default="timing"
    )
    args = parser.parse_args()
    if (
        args.samples < 1
        or args.warmups < 0
        or args.memory_trials < 1
        or args.deadline <= 0
        or not math.isfinite(args.deadline)
    ):
        parser.error("invalid sample counts or deadline")
    if args.worker:
        worker(
            Workload(**json.loads(args.worker)), args.mode, args.warmups, args.deadline
        )
    elif args.output is not None:
        run(
            args.output,
            args.case,
            args.samples,
            args.warmups,
            args.deadline,
            args.memory_trials,
            production_revision=args.production_revision,
        )
    else:
        parser.error("--output required")


if __name__ == "__main__":
    main()
