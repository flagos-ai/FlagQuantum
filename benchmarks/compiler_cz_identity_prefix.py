"""Compare bounded native CZ collectors with unchanged independent acceptance."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import platform
import subprocess
import sys
import time
import traceback
import tracemalloc
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import flagquantum
import flagquantum.compiler.artifacts.artifact_compilation as artifact_compilation
import flagquantum.compiler.lowering.native_block_optimization as native_block_optimization

_BASE = "2d78133a1f5a7d234acbb128724ccf2674e2eeb8"
_HARNESS = "9aa9260920984e2ca05bcb293abb885fccd9c144"
_COLLECTOR = "flagquantum/compiler/optimization/_cz_blocks.py"
_COMPILE = artifact_compilation.compile_circuit_artifact_for_target
Case = tuple[int, str, str]


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, timeout=15
    ).strip()


def _control(root: Path) -> tuple[ModuleType, str]:
    source = subprocess.check_output(
        ["git", "show", f"{_BASE}:{_COLLECTOR}"], cwd=root, timeout=15
    )
    module = ModuleType("flagquantum.compiler.optimization._cz_blocks_control")
    # Load the exact historical collector with its original relative imports.
    # Everything else remains the candidate's complete native target workflow.
    exec(compile(source, f"{_BASE}:{_COLLECTOR}", "exec"), module.__dict__)
    return module, hashlib.sha256(source).hexdigest()


def _observe(case: Case, harness: Path, memory: bool = False) -> dict[str, Any]:
    from sdkbench.candidates.adapters import load_adapter, workflow_configuration
    from sdkbench.candidates.applications import fixture
    from sdkbench.p0 import Bundle
    from sdkbench.p0.evaluators import routing

    width, family, topology = case
    bundle = Bundle(harness / "docs/implementation")
    acceptance = bundle.cases[
        "ROUTE-GRID-08" if width == 8 else f"ROUTE-LINE-{width:02d}"
    ]
    config = workflow_configuration("flagquantum-target-artifact")
    adapter = load_adapter(
        SimpleNamespace(
            configuration={
                "options": {
                    "sdk": config["sdk"],
                    "workflow": "flagquantum-target-artifact",
                    "compiler_options": config,
                }
            }
        )
    )
    captured: list[artifact_compilation.ArtifactCompilationResult] = []

    def compile_capture(*args: Any, **kwargs: Any) -> object:
        result = _COMPILE(*args, **kwargs)
        captured.append(result)
        return result

    selected = fixture(width, family, topology, "01" * 32)
    seed = bundle.plan["execution_profile"]["compiler_seeds"][0]
    with patch.object(
        artifact_compilation, "compile_circuit_artifact_for_target", compile_capture
    ):
        if memory:
            tracemalloc.start()
        try:
            start = time.perf_counter()
            payload = adapter.execute("route", selected, seed)
            seconds = time.perf_counter() - start
            peak = tracemalloc.get_traced_memory()[1] if memory else None
        finally:
            if memory:
                tracemalloc.stop()
    metrics = routing(payload, selected, None, acceptance, 0)
    if len(captured) != 1:
        raise AssertionError("Expected one complete native target compilation")
    result = captured[0]
    route = result.legalization.topology_legalization
    if route is None or (
        route.initial_logical_to_physical != tuple(range(width))
        or route.final_logical_to_physical != tuple(range(width))
        or payload["final_layout"] != list(range(width))
    ):
        raise AssertionError("Endpoint identity placement was not preserved")
    stage = result.legalization.native_block_optimization
    if stage is None or any(
        len(rewrite.input_instruction_indices) > 64 for rewrite in stage.rewrites
    ):
        raise AssertionError("Bounded native rewrite evidence was not preserved")
    normalized = json.dumps(
        {key: payload[key] for key in ("operations", "final_layout")},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        "metrics": metrics,
        "selected_strategy": route.strategy,
        "restored_swap_count": route.inserted_swap_count,
        "rewrite_count": len(stage.rewrites),
        "maximum_rewrite_inputs": max(
            (len(r.input_instruction_indices) for r in stage.rewrites), default=0
        ),
        "normalized_output_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "seconds": seconds,
        "python_peak_bytes": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdkbench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(flagquantum.__file__).resolve().parents[1]
    harness = args.sdkbench_root.resolve()
    if _git(harness, "rev-parse", "HEAD") != _HARNESS or _git(
        harness, "status", "--porcelain"
    ):
        parser.error("The pinned SDKBench checkout must be unchanged")
    if _git(root, "diff", "HEAD", "--", "flagquantum"):
        parser.error("Compiler source must match the recorded commit")
    sys.path.insert(0, str(harness / "src"))
    control, control_hash = _control(root)

    def greedy(
        *positional: Any, identity_reader: object = None, **keywords: Any
    ) -> Any:
        return control._collect_blocks(*positional, **keywords)

    def mode_context(mode: str) -> Any:
        return (
            patch.object(native_block_optimization, "_collect_blocks", greedy)
            if mode == "greedy_control"
            else contextlib.nullcontext()
        )

    report: dict[str, Any] = {
        "evidence_class": "development",
        "implementation_commit": _git(root, "rev-parse", "HEAD"),
        "base_commit": _BASE,
        "source_commit": _HARNESS,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "control_collector_sha256": control_hash,
        "python": sys.version,
        "platform": platform.platform(),
        "limits": "Single-device CPU development control on shared macOS, one numerical thread. Historical collector replaces only the private native collection boundary; this is not a pristine historical compiler or a formal participant profile. Unchanged SDKBench inputs, evaluator and caps; identity endpoint layouts; native verification and serialization included. Six-qubit full operators, three probes at eight/twelve. Cases inspected during development; no held-out qualification, stable speedup, process RSS, hardware, scalability or SDK ranking claim. Separate tracemalloc observations cover Python allocations only and are excluded from timing samples. Failures retained, all cases attempted, nonzero final exit on any failure.",
        "rows": [],
        "latency": [],
        "memory": [],
        "failures": [],
        "completed": False,
    }

    def checkpoint() -> None:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def observe(case: Case, mode: str, memory: bool = False) -> dict[str, Any] | None:
        try:
            with mode_context(mode):
                return _observe(case, harness, memory)
        except Exception as error:
            # Diagnostic boundary: retain every exception, never compiler fallback.
            report["failures"].append(
                {
                    "case": case,
                    "mode": mode,
                    "memory": memory,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            checkpoint()
            return None

    modes = ("greedy_control", "identity_prefix")
    for index, case in enumerate(
        itertools.product(
            (6, 8, 12),
            ("qaoa", "ising", "hopping", "qft", "repetition"),
            ("line", "ring", "grid", "sparse"),
        )
    ):
        observations = {}
        for mode in modes if index % 2 == 0 else modes[::-1]:
            observations[mode] = observe(case, mode)
        report["rows"].append({"case": case, "observations": observations})
        checkpoint()
        print(index + 1, case, "inspected", flush=True)
    for case in ((8, "ising", "line"), (12, "qft", "grid")):
        for index in range(23):
            for mode in modes if index % 2 == 0 else modes[::-1]:
                observation = observe(case, mode)
                report["latency"].append(
                    {
                        "case": case,
                        "mode": mode,
                        "warmup": index < 2,
                        "sample": index,
                        "observation": observation,
                    }
                )
            checkpoint()
        for mode in modes:
            report["memory"].append(
                {"case": case, "mode": mode, "observation": observe(case, mode, True)}
            )
            checkpoint()
        print(case, "latency and Python memory inspected", flush=True)
    report["completed"] = True
    checkpoint()
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
