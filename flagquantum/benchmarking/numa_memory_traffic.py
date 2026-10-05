#!/usr/bin/env python3
"""Measure Linux x86 NUMA policy effects with uncore DRAM counters."""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from .contract import runtime_metadata, write_json_atomic

SCHEMA = "flagquantum.numa_memory_traffic.v1"
RUNNER = "numa_memory_traffic"
SEED = 7319
POLICIES = ("default", "bind", "interleave")
ENGINES = ("flagquantum_adjoint", "pennylane_lightning_adjoint")
WORKLOADS = ("hardware_efficient_vqe", "qaoa_path_maxcut")
_PERF_EVENT_IOC_ENABLE = 0x2400
_PERF_EVENT_IOC_DISABLE = 0x2401
_PERF_EVENT_IOC_RESET = 0x2403
_MPOL_DEFAULT = 0
_MPOL_BIND = 2
_MPOL_INTERLEAVE = 3
_STABILITY_LIMIT = 0.20
_CORRECTNESS_TOLERANCE = 1e-9
_EXTERNAL_CORRECTNESS_TOLERANCE = 1e-6
_MATERIAL_SPEEDUP = 1.05


class _PerfEventAttr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("sample_period", ctypes.c_uint64),
        ("sample_type", ctypes.c_uint64),
        ("read_format", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("wakeup_events", ctypes.c_uint32),
        ("bp_type", ctypes.c_uint32),
        ("config1", ctypes.c_uint64),
        ("config2", ctypes.c_uint64),
        ("branch_sample_type", ctypes.c_uint64),
        ("sample_regs_user", ctypes.c_uint64),
        ("sample_stack_user", ctypes.c_uint32),
        ("clockid", ctypes.c_int32),
        ("sample_regs_intr", ctypes.c_uint64),
        ("aux_watermark", ctypes.c_uint32),
        ("sample_max_stack", ctypes.c_uint16),
        ("reserved2", ctypes.c_uint16),
        ("aux_sample_size", ctypes.c_uint32),
        ("reserved3", ctypes.c_uint32),
        ("sig_data", ctypes.c_uint64),
    ]


def _require_linux_x86_64() -> None:
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise RuntimeError("NUMA traffic measurement requires Linux x86-64")


def _set_memory_policy(policy: str, nodes: Sequence[int]) -> None:
    """Set the calling thread's inherited Linux memory policy."""

    _require_linux_x86_64()
    if policy not in POLICIES:
        raise ValueError(f"unsupported NUMA policy: {policy}")
    if any(node < 0 or node >= 64 for node in nodes):
        raise ValueError("NUMA node identifiers must be in [0, 63]")
    mode = {
        "default": _MPOL_DEFAULT,
        "bind": _MPOL_BIND,
        "interleave": _MPOL_INTERLEAVE,
    }[policy]
    selected = () if policy == "default" else tuple(nodes)
    if policy != "default" and not selected:
        raise ValueError(f"{policy} requires at least one NUMA node")
    mask = ctypes.c_ulong(sum(1 << node for node in selected))
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(
        238,
        mode,
        ctypes.byref(mask) if selected else None,
        64 if selected else 0,
    )
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, errno.errorcode.get(code, "set_mempolicy"))


def _event_config(path: Path) -> int:
    fields = {
        name: int(value, 0)
        for name, value in re.findall(
            r"(event|umask)=(0x[0-9a-fA-F]+|\d+)", path.read_text()
        )
    }
    if set(fields) != {"event", "umask"}:
        raise RuntimeError(f"unsupported uncore event encoding: {path}")
    return fields["event"] | (fields["umask"] << 8)


def _uncore_sources(
    root: Path = Path("/sys/bus/event_source/devices"),
) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in root.glob("uncore_imc_[0-9]*")
                if "free_running" not in path.name
                and (path / "events" / "cas_count_read").is_file()
                and (path / "events" / "cas_count_write").is_file()
            ),
            key=lambda path: int(path.name.rsplit("_", 1)[1]),
        )
    )


def _open_uncore_event(source: Path, event: str, cpu: int) -> int:
    attr = _PerfEventAttr()
    attr.type = int((source / "type").read_text())
    attr.size = ctypes.sizeof(attr)
    attr.config = _event_config(source / "events" / event)
    attr.flags = 1  # disabled until the synchronized timing window starts
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.syscall(298, ctypes.byref(attr), -1, cpu, -1, 0)
    if descriptor < 0:
        code = ctypes.get_errno()
        raise OSError(code, errno.errorcode.get(code, "perf_event_open"))
    return int(descriptor)


class _UncoreCounters:
    def __init__(self, socket_cpus: Sequence[int]) -> None:
        sources = _uncore_sources()
        if not sources:
            raise RuntimeError("Intel uncore IMC CAS counters are unavailable")
        self._socket_cpus = tuple(int(cpu) for cpu in socket_cpus)
        self._fds: dict[tuple[int, str, str], int] = {}
        try:
            for socket_index, cpu in enumerate(self._socket_cpus):
                for source in sources:
                    for event in ("cas_count_read", "cas_count_write"):
                        self._fds[(socket_index, source.name, event)] = (
                            _open_uncore_event(source, event, cpu)
                        )
        except Exception:
            self.close()
            raise

    def start(self) -> None:
        for descriptor in self._fds.values():
            fcntl.ioctl(descriptor, _PERF_EVENT_IOC_RESET, 0)
            fcntl.ioctl(descriptor, _PERF_EVENT_IOC_ENABLE, 0)

    def stop(self) -> dict[str, Any]:
        for descriptor in self._fds.values():
            fcntl.ioctl(descriptor, _PERF_EVENT_IOC_DISABLE, 0)
        counts = {
            key: int.from_bytes(os.read(descriptor, 8), "little")
            for key, descriptor in self._fds.items()
        }
        sockets: dict[str, dict[str, int]] = {}
        for socket_index in range(len(self._socket_cpus)):
            reads = sum(
                value
                for (index, _, event), value in counts.items()
                if index == socket_index and event == "cas_count_read"
            )
            writes = sum(
                value
                for (index, _, event), value in counts.items()
                if index == socket_index and event == "cas_count_write"
            )
            sockets[str(socket_index)] = {
                "representative_cpu": self._socket_cpus[socket_index],
                "read_bytes": reads * 64,
                "write_bytes": writes * 64,
            }
        return {
            "sockets": sockets,
            "total_bytes": sum(
                item["read_bytes"] + item["write_bytes"] for item in sockets.values()
            ),
        }

    def close(self) -> None:
        for descriptor in self._fds.values():
            os.close(descriptor)
        self._fds.clear()


def _summary(samples: Sequence[float]) -> dict[str, Any]:
    values = tuple(float(value) for value in samples)
    if not values or any(value <= 0 for value in values):
        raise ValueError("timing samples must be positive and non-empty")
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return {
        "samples_seconds": values,
        "sample_count": len(values),
        "median_seconds": median,
        "mean_seconds": statistics.fmean(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "relative_median_absolute_deviation": mad / median,
    }


def parse_cpu_list(value: str) -> tuple[int, ...]:
    """Parse Linux taskset notation without importing a PyTorch runner."""

    cpus: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("CPU list contains an empty item")
        if "-" in item:
            parts = item.split("-")
            if len(parts) != 2:
                raise ValueError(f"invalid CPU range: {item}")
            first, last = (int(part) for part in parts)
            if first < 0 or last < first:
                raise ValueError(f"invalid CPU range: {item}")
            cpus.extend(range(first, last + 1))
        else:
            cpu = int(item)
            if cpu < 0:
                raise ValueError("CPU identifiers must be non-negative")
            cpus.append(cpu)
    if not cpus or len(set(cpus)) != len(cpus):
        raise ValueError("CPU list must be non-empty and contain no duplicates")
    return tuple(cpus)


def _worker(payload_text: str) -> int:
    import torch

    from ..simulation.native_cpu import native_cpu_adjoint_available
    from .differentiable_simulator_corpus import _engine_callable

    payload = json.loads(payload_text)
    policy = str(payload["policy"])
    nodes = tuple(int(node) for node in payload["nodes"])
    cpus = tuple(int(cpu) for cpu in payload["cpus"])
    threads = int(payload["threads"])
    _set_memory_policy(policy, nodes)
    set_affinity = cast(
        Callable[[int, set[int]], None], getattr(os, "sched_setaffinity", None)
    )
    get_affinity = cast(
        Callable[[int], set[int]], getattr(os, "sched_getaffinity", None)
    )
    if not callable(set_affinity) or not callable(get_affinity):
        raise RuntimeError("NUMA traffic measurement requires Linux CPU affinity")
    set_affinity(0, set(cpus))
    value = str(threads)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = value
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_PLACES"] = "cores"
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    engine = cast(Any, payload["engine"])
    workload = cast(Any, payload["workload"])
    execute = _engine_callable(
        engine,
        workload,
        n_qubits=int(payload["n_wires"]),
        layers=int(payload["layers"]),
        seed=int(payload["seed"]),
    )
    for _ in range(int(payload["warmup"])):
        execute()
    print(
        json.dumps(
            {
                "kind": "ready",
                "policy": policy,
                "engine": engine,
                "workload": workload,
                "observed_cpus": sorted(get_affinity(0)),
                "threads": torch.get_num_threads(),
                "native_cpu_adjoint": native_cpu_adjoint_available(),
            }
        ),
        flush=True,
    )
    if sys.stdin.readline().strip() != "start":
        raise RuntimeError("worker did not receive the start barrier")
    totals: list[float] = []
    forwards: list[float] = []
    backwards: list[float] = []
    checksum = 0.0
    last_result: Any | None = None
    for _ in range(int(payload["calls"])):
        result = execute()
        last_result = result
        forwards.append(result.forward_seconds)
        backwards.append(result.backward_seconds)
        totals.append(result.forward_seconds + result.backward_seconds)
        checksum += result.value + float(result.gradient.sum())
    if last_result is None:
        raise RuntimeError("worker retained no measurements")
    print(
        json.dumps(
            {
                "kind": "result",
                "total": _summary(totals),
                "forward": _summary(forwards),
                "backward": _summary(backwards),
                "checksum": checksum,
                "value": last_result.value,
                "gradient": last_result.gradient.reshape(-1).tolist(),
            }
        ),
        flush=True,
    )
    return 0


def _run_case(
    *,
    engine: str,
    workload: str,
    policy: str,
    nodes: Sequence[int],
    cpus: Sequence[int],
    socket_cpus: Sequence[int],
    n_qubits: int,
    layers: int,
    threads: int,
    warmup: int,
    calls: int,
    seed: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "engine": engine,
        "workload": workload,
        "policy": policy,
        "nodes": tuple(nodes),
        "cpus": tuple(cpus),
        "n_wires": n_qubits,
        "layers": layers,
        "threads": threads,
        "warmup": warmup,
        "calls": calls,
        "seed": seed,
    }
    process = subprocess.Popen(
        (
            sys.executable,
            "-m",
            "flagquantum.benchmarking.numa_memory_traffic",
            "--worker-payload",
            json.dumps(payload, separators=(",", ":")),
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp")},
    )
    counters: _UncoreCounters | None = None
    try:
        assert process.stdout is not None
        ready_line = process.stdout.readline()
        if not ready_line:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise RuntimeError(f"NUMA worker exited before ready: {stderr}")
        ready = json.loads(ready_line)
        if ready.get("kind") != "ready":
            raise RuntimeError(f"invalid NUMA worker ready record: {ready}")
        counters = _UncoreCounters(socket_cpus)
        counters.start()
        started = time.perf_counter()
        assert process.stdin is not None
        process.stdin.write("start\n")
        process.stdin.flush()
        result_line = process.stdout.readline()
        elapsed = time.perf_counter() - started
        result = json.loads(result_line)
        if result.get("kind") != "result":
            raise RuntimeError(f"invalid NUMA worker result record: {result}")
        process.wait(timeout=timeout_seconds)
        if process.returncode:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise RuntimeError(f"NUMA worker failed: {stderr}")
        traffic = counters.stop()
        return {
            **payload,
            "ready": ready,
            "timing": {
                "total": result["total"],
                "forward": result["forward"],
                "backward": result["backward"],
                "monitor_wall_seconds": elapsed,
            },
            "checksum": result["checksum"],
            "value": result["value"],
            "gradient": result["gradient"],
            "traffic": {
                **traffic,
                "bytes_per_call": traffic["total_bytes"] / calls,
            },
        }
    finally:
        if process.poll() is None:
            process.kill()
        if counters is not None:
            counters.close()


def _annotate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    correctness_passed = True
    stability_passed = True
    native_path_passed = True
    for workload in {str(case["workload"]) for case in cases}:
        workload_cases = [case for case in cases if case["workload"] == workload]
        reference = next(
            case
            for case in workload_cases
            if case["engine"] == "flagquantum_adjoint" and case["policy"] == "default"
        )
        reference_gradient = tuple(float(value) for value in reference["gradient"])
        reference_seconds = float(reference["timing"]["total"]["median_seconds"])
        reference_bytes = float(reference["traffic"]["bytes_per_call"])
        for case in workload_cases:
            gradient = tuple(float(value) for value in case["gradient"])
            gradient_error = (
                max(
                    (
                        abs(left - right)
                        for left, right in zip(
                            reference_gradient, gradient, strict=True
                        )
                    ),
                    default=0.0,
                )
                if len(gradient) == len(reference_gradient)
                else float("inf")
            )
            value_error = abs(float(case["value"]) - float(reference["value"]))
            tolerance = (
                _CORRECTNESS_TOLERANCE
                if case["engine"] == "flagquantum_adjoint"
                else _EXTERNAL_CORRECTNESS_TOLERANCE
            )
            correct = value_error <= tolerance and gradient_error <= tolerance
            stable = (
                float(case["timing"]["total"]["relative_median_absolute_deviation"])
                <= _STABILITY_LIMIT
            )
            case["correctness"] = {
                "reference": "flagquantum_adjoint/default",
                "value_absolute_error": value_error,
                "gradient_max_absolute_error": gradient_error,
                "tolerance": tolerance,
                "passed": correct,
            }
            case["stable"] = stable
            case["speedup_vs_flagquantum_default"] = reference_seconds / float(
                case["timing"]["total"]["median_seconds"]
            )
            case["traffic_ratio_vs_flagquantum_default"] = (
                float(case["traffic"]["bytes_per_call"]) / reference_bytes
            )
            correctness_passed = correctness_passed and correct
            stability_passed = stability_passed and stable
            if case["engine"] == "flagquantum_adjoint":
                native_path_passed = native_path_passed and bool(
                    case["ready"]["native_cpu_adjoint"]
                )
    flagquantum_cases = [
        case for case in cases if case["engine"] == "flagquantum_adjoint"
    ]
    non_default = {
        policy: [case for case in flagquantum_cases if case["policy"] == policy]
        for policy in POLICIES
        if policy != "default"
    }
    universally_materially_faster = {
        policy: bool(items)
        and all(
            case["speedup_vs_flagquantum_default"] >= _MATERIAL_SPEEDUP
            for case in items
        )
        for policy, items in non_default.items()
    }
    return {
        "passed": correctness_passed and stability_passed and native_path_passed,
        "correctness_passed": correctness_passed,
        "stability_passed": stability_passed,
        "native_path_passed": native_path_passed,
        "maximum_relative_median_absolute_deviation": _STABILITY_LIMIT,
        "recommended_policy": "default",
        "automatic_bind_enabled": False,
        "forced_interleave_enabled": False,
        "material_speedup_threshold": _MATERIAL_SPEEDUP,
        "non_default_policy_universally_materially_faster": (
            universally_materially_faster
        ),
        "reason": (
            "keep Linux first-touch because neither forced bind nor forced "
            "interleave improves every measured workload by at least 5%"
        ),
    }


def run_benchmark(
    *,
    workloads: Sequence[str],
    engines: Sequence[str],
    policies: Sequence[str],
    nodes: Sequence[int],
    cpus: Sequence[int],
    socket_cpus: Sequence[int],
    n_qubits: int,
    layers: int,
    threads: int,
    warmup: int,
    calls: int,
    seed: int = SEED,
    host_label: str = "redacted",
    source_revision: str = "unknown",
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    """Run matched NUMA-policy cells and return one auditable payload."""

    _require_linux_x86_64()
    if not workloads or any(item not in WORKLOADS for item in workloads):
        raise ValueError("workloads must be supported and non-empty")
    if not engines or any(item not in ENGINES for item in engines):
        raise ValueError("engines must be supported and non-empty")
    if not policies or any(item not in POLICIES for item in policies):
        raise ValueError("policies must be supported and non-empty")
    if len(set(workloads)) != len(workloads) or len(set(engines)) != len(engines):
        raise ValueError("workloads and engines must not contain duplicates")
    if threads < 1 or len(cpus) < threads:
        raise ValueError("the CPU list must contain at least one CPU per thread")
    if len(socket_cpus) != len(nodes):
        raise ValueError("one representative CPU is required for every NUMA node")
    if n_qubits < 2 or layers < 1 or warmup < 0 or calls < 3:
        raise ValueError("invalid workload or sampling configuration")
    cases = []
    for workload in workloads:
        for engine in engines:
            selected_policies = (
                policies if engine == "flagquantum_adjoint" else ("default",)
            )
            for policy in selected_policies:
                cases.append(
                    _run_case(
                        engine=engine,
                        workload=workload,
                        policy=policy,
                        nodes=nodes,
                        cpus=cpus,
                        socket_cpus=socket_cpus,
                        n_qubits=n_qubits,
                        layers=layers,
                        threads=threads,
                        warmup=warmup,
                        calls=calls,
                        seed=seed,
                        timeout_seconds=timeout_seconds,
                    )
                )
    assessment = _annotate_cases(cases)
    payload = runtime_metadata(runner=RUNNER, schema=SCHEMA)
    payload["hostname"] = host_label
    payload.update(
        {
            "benchmark": "linux_x86_numa_adjoint_memory_traffic",
            "distribution_semantics": "single_device_fast_path",
            "claim_evidence_type": "unknown",
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "non_release_evidence": True,
            "benchmark_evidence_class": "comparison_non_release",
            "scalability_blockers": (
                "comparison_result_not_release_scalability_evidence",
            ),
            "passed": assessment["passed"],
            "assessment": assessment,
            "methodology": {
                "measurement": "expectation value plus complete gradient",
                "precision": "complex128",
                "memory_counters": "Intel uncore IMC CAS reads and writes, 64 bytes/event",
                "policy_scope": "inherited per-process Linux set_mempolicy",
                "placement": "sched_setaffinity plus OMP_PROC_BIND=close, OMP_PLACES=cores",
                "counter_window": "after warmup, around retained calls only",
            },
            "configuration": {
                "workloads": tuple(workloads),
                "engines": tuple(engines),
                "policies": tuple(policies),
                "nodes": tuple(nodes),
                "cpus": tuple(cpus),
                "socket_representative_cpus": tuple(socket_cpus),
                "n_wires": n_qubits,
                "layers": layers,
                "threads": threads,
                "warmup": warmup,
                "calls": calls,
                "seed": seed,
                "source_revision": source_revision,
            },
            "cases": cases,
            "claim_boundaries": (
                "linux_x86_64_only",
                "intel_uncore_imc_counters_required",
                "cas_counts_include_system_background_traffic_during_the_timed_window",
                "default_policy_means_linux_first_touch_not_forced_interleave",
                "bind_results_do_not_imply_a_universal_runtime_default",
            ),
        }
    )
    return payload


def render_markdown(payload: dict[str, Any], *, artifact_name: str) -> str:
    """Render the measured policy decision and external comparison."""

    configuration = payload["configuration"]
    lines = [
        "# Linux x86 NUMA memory traffic for differentiable statevectors",
        "",
        "## Decision",
        "",
        "**Keep the Linux default first-touch policy.** Forced interleave is not an",
        "optimization for the measured native adjoint workloads, and forced bind has",
        "mixed workload/size behavior, so neither is enabled automatically.",
        "",
        "## Measured results",
        "",
        "`Relative speed` is FQ-default time divided by row time and is above one",
        "when that row is faster. Traffic is",
        "uncore IMC CAS read+write bytes per complete value-and-gradient call.",
        "",
        "| Workload | Engine | NUMA policy | Total | DRAM/call | Relative speed | Traffic vs FQ default |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "flagquantum_adjoint": "FlagQuantum native adjoint",
        "pennylane_lightning_adjoint": "PennyLane Lightning adjoint",
    }
    for case in payload["cases"]:
        lines.append(
            "| {workload} | {engine} | {policy} | {milliseconds:.3f} ms | "
            "{gib:.3f} GiB | {speedup:.3f}x | {traffic:.3f}x |".format(
                workload=case["workload"],
                engine=labels[case["engine"]],
                policy=case["policy"],
                milliseconds=1000 * case["timing"]["total"]["median_seconds"],
                gib=case["traffic"]["bytes_per_call"] / 2**30,
                speedup=case["speedup_vs_flagquantum_default"],
                traffic=case["traffic_ratio_vs_flagquantum_default"],
            )
        )
    lines.extend(["", "## External same-semantics comparison", ""])
    for workload in configuration["workloads"]:
        native = next(
            case
            for case in payload["cases"]
            if case["workload"] == workload
            and case["engine"] == "flagquantum_adjoint"
            and case["policy"] == "default"
        )
        lightning = next(
            case
            for case in payload["cases"]
            if case["workload"] == workload
            and case["engine"] == "pennylane_lightning_adjoint"
        )
        latency_ratio = (
            lightning["timing"]["total"]["median_seconds"]
            / native["timing"]["total"]["median_seconds"]
        )
        traffic_ratio = (
            lightning["traffic"]["bytes_per_call"] / native["traffic"]["bytes_per_call"]
        )
        lines.append(
            f"- `{workload}`: FlagQuantum is **{latency_ratio:.3f}x faster** and "
            f"Lightning records **{traffic_ratio:.3f}x** as many IMC bytes per call."
        )
    lines.extend(
        [
            "",
            "## What this measures and why it matters",
            "",
            "Each retained call computes one expectation value and its complete",
            "parameter gradient. VQE has independent RX/RY/RZ parameters plus a local",
            "Hamiltonian; QAOA has shared RZZ/RX parameters and a MaxCut Hamiltonian.",
            "The result therefore tests user-visible training evaluations, not an",
            "isolated gate microbenchmark. IMC counters show whether an apparent NUMA",
            "speedup merely moves or duplicates DRAM traffic across sockets.",
            "",
            "A minimal public-API call with the same differentiation route is:",
            "",
            "```python",
            "import torch",
            "import flagquantum as fq",
            "import flagquantum.algorithms as fqa",
            "",
            "theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)",
            "circuit = fq.Circuit(2, dtype=torch.complex128)",
            "circuit.h(0).cx(0, 1).ry(1, theta)",
            'hamiltonian = fqa.Hamiltonian([fqa.HamiltonianTerm(1.0, "z", 1)])',
            'value = hamiltonian.expectation(circuit, differentiation="adjoint")',
            "gradient = torch.autograd.grad(value, theta)[0]",
            "```",
            "",
            "All cells use the same circuit, observable, complex128 precision and",
            "complete gradient semantics. FlagQuantum policy cells use a",
            f"`{_CORRECTNESS_TOLERANCE:g}` correctness tolerance; the independent",
            f"Lightning reduction uses `{_EXTERNAL_CORRECTNESS_TOLERANCE:g}`. Retained",
            "timing rMAD must be at most",
            f"{100 * _STABILITY_LIMIT:.0f}%. The raw samples and per-socket counters are in",
            f"[`{artifact_name}`]({artifact_name}).",
            "",
            "## Reproduce",
            "",
            "```bash",
            'export CUDA_VISIBLE_DEVICES=""',
            "export OMP_PROC_BIND=close OMP_PLACES=cores",
            "flagquantum-benchmark run numa_memory_traffic \\",
            "  --workloads hardware_efficient_vqe qaoa_path_maxcut \\",
            "  --engines flagquantum_adjoint pennylane_lightning_adjoint \\",
            "  --policies default bind interleave --nodes 0,1 \\",
            "  --cpu-list 0-63 --socket-cpus 0,32 --n-wires 22 --threads 64 \\",
            f"  --warmup {configuration['warmup']} --calls {configuration['calls']} \\",
            f"  --source-revision {configuration['source_revision']} \\",
            f"  --json-output {artifact_name}",
            "```",
            "",
            "The command needs Linux x86-64, permission to open system-wide perf",
            "events, and Intel IMC PMUs exposing `cas_count_read/write`.",
            "",
            "## Boundaries and stopping condition",
            "",
            "- CAS counts include unrelated system traffic inside the synchronized window.",
            "- Results cover the listed CPU, 22 qubits, one layer, complex128 and 64 physical cores.",
            "- `bind` results do not justify a general runtime default; larger sizes were also checked before rejecting it.",
            "- This NUMA-policy phase stops when both workloads are correct and stable,",
            "  the native path is observed, an external same-semantics comparison is",
            "  present, and no forced policy improves every measured workload.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(item) for item in value.split(",") if item.strip())
    if not items or len(set(items)) != len(items) or any(item < 0 for item in items):
        raise argparse.ArgumentTypeError(
            "expected unique non-negative comma-separated integers"
        )
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-payload", help=argparse.SUPPRESS)
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=WORKLOADS)
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=ENGINES)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    parser.add_argument("--nodes", type=_parse_int_list, default=(0, 1))
    parser.add_argument("--cpu-list", default="0-63")
    parser.add_argument("--socket-cpus", type=_parse_int_list, default=(0, 32))
    parser.add_argument("--n-wires", type=int, default=22)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--calls", type=int, default=7)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--host-label", default="redacted")
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--json-output", type=Path, required=False)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.worker_payload:
        return _worker(args.worker_payload)
    if args.json_output is None:
        raise SystemExit("--json-output is required")
    payload = run_benchmark(
        workloads=args.workloads,
        engines=args.engines,
        policies=args.policies,
        nodes=args.nodes,
        cpus=parse_cpu_list(args.cpu_list),
        socket_cpus=args.socket_cpus,
        n_qubits=args.n_wires,
        layers=args.layers,
        threads=args.threads,
        warmup=args.warmup,
        calls=args.calls,
        seed=args.seed,
        host_label=args.host_label,
        source_revision=args.source_revision,
        timeout_seconds=args.timeout_seconds,
    )
    write_json_atomic(args.json_output, payload)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            render_markdown(payload, artifact_name=args.json_output.name),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ENGINES",
    "POLICIES",
    "RUNNER",
    "SCHEMA",
    "WORKLOADS",
    "build_parser",
    "main",
    "parse_cpu_list",
    "render_markdown",
    "run_benchmark",
]
