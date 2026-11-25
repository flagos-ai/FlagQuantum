"""Reproduce a rejected decay experiment after applying its archived source patch."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

from flagquantum.compiler import CouplingMap
from flagquantum.compiler.topology import routing_strategy, sabre
from flagquantum.core.ir import CircuitIR

_HARNESS_COMMIT = "9aa9260920984e2ca05bcb293abb885fccd9c144"


def _generic(width: int, topology: str, seed: int) -> dict[str, object]:
    from benchmarks.compiler_sabre_trials import _metrics, _program

    program = _program(width, seed)
    coupling = (
        CouplingMap.line(width)
        if topology == "line"
        else CouplingMap.grid(2, width // 2)
    )
    start = time.perf_counter()
    baseline = sabre.plan_sabre_swaps(program, coupling)
    baseline_seconds = time.perf_counter() - start
    before = _metrics(program, coupling, baseline)
    start = time.perf_counter()
    candidate = sabre._plan_sabre_trial(
        program, coupling, decay=True, committed_swap_limit=before["total_swaps"]
    )
    extra_seconds = time.perf_counter() - start
    after = None if candidate is None else _metrics(program, coupling, candidate)
    selected = (
        after
        if after is not None and after["total_swaps"] < before["total_swaps"]
        else before
    )
    return {
        "baseline": before,
        "decay": after,
        "selected": selected,
        "baseline_seconds": baseline_seconds,
        "additional_trial_seconds": extra_seconds,
    }


def _application(
    width: int, family: str, topology: str, workflow: str, harness: Path
) -> dict[str, object]:
    from types import SimpleNamespace

    sys.path.insert(0, str(harness / "src"))
    from sdkbench.candidates.adapters import load_adapter, workflow_configuration
    from sdkbench.candidates.applications import fixture
    from sdkbench.p0 import Bundle
    from sdkbench.p0.evaluators import routing

    bundle = Bundle(harness / "docs/implementation")
    case = bundle.cases["ROUTE-GRID-08" if width == 8 else f"ROUTE-LINE-{width:02d}"]
    config = workflow_configuration(workflow)
    registration = SimpleNamespace(
        configuration={
            "options": {
                "sdk": config["sdk"],
                "workflow": workflow,
                "compiler_options": config,
            }
        }
    )
    source = fixture(width, family, topology, "01" * 32)
    seed = bundle.plan["execution_profile"]["compiler_seeds"][0]
    choices: list[dict[str, int | bool | None]] = []
    original = sabre._select_sabre_trials

    def select(
        program: CircuitIR,
        coupling: CouplingMap,
        baseline: sabre.SabrePlan,
        *,
        cost: Callable[[sabre.SabrePlan], int] | None = None,
    ) -> sabre.SabrePlan:
        best = original(program, coupling, baseline, cost=cost)
        score = cost or (lambda plan: sabre._restored_swap_count(plan, coupling))
        candidate = sabre._plan_sabre_trial(
            program,
            coupling,
            initial_layout=baseline.initial_logical_to_physical,
            decay=True,
            committed_swap_limit=score(best),
        )
        adopted = candidate is not None and score(candidate) < score(best)
        choices.append(
            {
                "before": score(best),
                "decay": (None if candidate is None else score(candidate)),
                "selected": adopted,
            }
        )
        return candidate if candidate is not None and adopted else best

    adapter = load_adapter(registration)
    before = adapter.execute("route", source, seed)
    before_metrics = routing(before, source, None, case, 0)
    adapter = load_adapter(registration)
    with (
        patch.object(sabre, "_select_sabre_trials", select),
        patch.object(routing_strategy, "_select_sabre_trials", select),
    ):
        after = adapter.execute("route", source, seed)
    after_metrics = routing(after, source, None, case, 0)
    if any(p["final_layout"] != list(range(width)) for p in (before, after)):
        raise AssertionError("The experiment did not restore the identity layout")
    return {
        "baseline_metrics": before_metrics,
        "metrics": after_metrics,
        "choices": choices,
        "audit": adapter.audit,
        "acceptance": case["acceptance"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("generic", "applications"), required=True)
    parser.add_argument("--sdkbench-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if "decay" not in inspect.signature(sabre._plan_sabre_trial).parameters:
        parser.error(
            "Apply docs/development/evidence/sabre-decay.patch in a disposable checkout first"
        )
    if args.suite == "applications":
        if args.sdkbench_root is None:
            parser.error("--sdkbench-root is required for applications")
        args.sdkbench_root = args.sdkbench_root.resolve()
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=args.sdkbench_root,
            text=True,
            timeout=10,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=args.sdkbench_root,
            text=True,
            timeout=10,
        ).strip()
        if revision != _HARNESS_COMMIT or dirty:
            parser.error("Applications require the unchanged pinned SDKBench checkout")
    if args.worker is not None:
        case = json.loads(args.worker)
        result = (
            _generic(*case)
            if args.suite == "generic"
            else _application(*case, args.sdkbench_root)
        )
        print(json.dumps(result, allow_nan=False))
        return
    if args.output is None:
        parser.error("--output is required")
    cases = (
        itertools.product((6, 8, 12, 16), ("line", "grid"), range(10))
        if args.suite == "generic"
        else itertools.product(
            (6, 8, 12),
            ("qaoa", "ising", "hopping", "qft", "repetition"),
            ("line", "ring", "grid", "sparse"),
            ("flagquantum-ordinary-restored", "flagquantum-target-artifact"),
        )
    )
    outcomes: list[dict[str, object]] = []
    report = {
        "evidence_class": "development",
        "suite": args.suite,
        "source_sha256": hashlib.sha256(Path(sabre.__file__).read_bytes()).hexdigest(),
        "source_commit": _HARNESS_COMMIT if args.suite == "applications" else None,
        "limits": "Inspected inputs, shared CPU host; no held-out, memory, ranking or qualified timing claims. Generic depths count logical CX/SWAP. Applications at n=8,12 verify only three probes.",
        "outcomes": outcomes,
    }
    environment = dict(os.environ)
    environment.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    for case in cases:
        command = [
            sys.executable,
            "-m",
            "benchmarks.compiler_sabre_decay",
            "--suite",
            args.suite,
            "--worker",
            json.dumps(case),
        ]
        if args.sdkbench_root is not None:
            command.extend(("--sdkbench-root", str(args.sdkbench_root)))
        start = time.monotonic()
        child = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = child.communicate(timeout=150)
            row = {
                "case": case,
                "status": "passed" if child.returncode == 0 else "failed",
                "returncode": child.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
            if child.returncode == 0:
                row["result"] = json.loads(stdout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                stdout, stderr = child.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                stdout, stderr = child.communicate(timeout=10)
            row = {
                "case": case,
                "status": "timeout",
                "stdout": stdout,
                "stderr": stderr,
            }
        row["development_wall_seconds"] = time.monotonic() - start
        outcomes.append(row)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(len(outcomes), case, row["status"], flush=True)
    if any(row["status"] != "passed" for row in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
