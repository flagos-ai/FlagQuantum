"""Inspect a rejected routing tie policy without changing production compilation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import random
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict
from unittest.mock import patch

from flagquantum.compiler import CouplingMap
from flagquantum.compiler.topology import sabre
from flagquantum.core.ir import CircuitIR

_HARNESS_COMMIT = "9aa9260920984e2ca05bcb293abb885fccd9c144"
_ORIGINAL = sabre._select_sabre_trials


class DevelopmentReport(TypedDict):
    evidence_class: str
    implementation_commit: str
    suite: str
    input_nonce: str | None
    completed: bool
    failures: list[dict[str, object]]
    compiler_sha256: str
    runner_sha256: str
    source_commit: str | None
    python: str
    platform: str
    limits: str
    rows: list[dict[str, object]]
    failure: dict[str, str] | None


def _depth(program: CircuitIR, coupling: CouplingMap, plan: sabre.SabrePlan) -> int:
    levels = [0] * program.n_wires

    def advance(wires: tuple[int, ...], weight: int) -> None:
        value = max((levels[q] for q in wires), default=0) + weight
        for q in wires:
            levels[q] = value

    swap_index = 0
    for count, index in enumerate(plan.sequence):
        while (
            swap_index < len(plan.swaps) and plan.swap_placements[swap_index] == count
        ):
            advance(plan.swaps[swap_index], 3)
            swap_index += 1
        advance(
            plan.placements[index], int(len(program.instructions[index].wires) == 2)
        )
    if swap_index != len(plan.swaps):
        raise AssertionError("Plan contains unemitted SWAPs")
    for left, right, _ in sabre.plan_restore_swaps(plan, coupling):
        advance((left, right), 3)
    return max(levels, default=0)


def _select(
    program: CircuitIR,
    coupling: CouplingMap,
    baseline: sabre.SabrePlan,
    *,
    cost: Callable[[sabre.SabrePlan], int] | None = None,
) -> sabre.SabrePlan:
    if cost is not None:
        return _ORIGINAL(program, coupling, baseline, cost=cost)
    if not baseline.swaps:
        return baseline
    best = baseline
    best_cost = sabre._restored_swap_count(best, coupling)
    best_depth = _depth(program, coupling, best)
    for seed, layers in ((0, False), (1, False), (2, False), (None, True)):
        candidate = sabre._plan_sabre_trial(
            program,
            coupling,
            initial_layout=baseline.initial_logical_to_physical,
            tie_rng=random.Random(seed) if seed is not None else None,
            committed_swap_limit=best_cost + 1,
            layer_lookahead=layers,
        )
        if candidate is None:
            continue
        candidate_cost = sabre._restored_swap_count(candidate, coupling)
        if candidate_cost > best_cost:
            continue
        candidate_depth = _depth(program, coupling, candidate)
        if (candidate_cost, candidate_depth) < (best_cost, best_depth):
            best, best_cost, best_depth = candidate, candidate_cost, candidate_depth
    return best


def _generic(width: int, topology: str, seed: int) -> dict[str, object]:
    from benchmarks.compiler_sabre_trials import _metrics, _program

    program = _program(width, seed)
    coupling = (
        CouplingMap.line(width)
        if topology == "line"
        else CouplingMap.grid(2, width // 2)
    )
    baseline = sabre.plan_sabre_swaps(program, coupling)
    with patch.object(sabre, "_select_sabre_trials", _select):
        candidate = sabre.plan_sabre_swaps(program, coupling)
    return {
        "baseline": _metrics(program, coupling, baseline),
        "candidate": _metrics(program, coupling, candidate),
    }


def _application(
    width: int,
    family: str,
    topology: str,
    workflow: str,
    harness: Path,
    *,
    nonce: str = "01" * 32,
) -> dict[str, object]:
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
    source = fixture(width, family, topology, nonce)
    seed = bundle.plan["execution_profile"]["compiler_seeds"][0]
    adapter = load_adapter(registration)
    before = adapter.execute("route", source, seed)
    before_audit = dict(adapter.audit)
    before_metrics = routing(before, source, None, case, 0)
    adapter = load_adapter(registration)
    with patch.object(sabre, "_select_sabre_trials", _select):
        after = adapter.execute("route", source, seed)
    after_metrics = routing(after, source, None, case, 0)
    if any(
        payload["final_layout"] != list(range(width)) for payload in (before, after)
    ):
        raise AssertionError("The experiment did not restore identity output layout")
    return {
        "baseline": before_metrics,
        "candidate": after_metrics,
        "baseline_audit": before_audit,
        "candidate_audit": dict(adapter.audit),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("generic", "applications"), required=True)
    parser.add_argument("--sdkbench-root", type=Path)
    parser.add_argument("--nonce", default="01" * 32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.nonce) != 64 or any(c not in "0123456789abcdef" for c in args.nonce):
        parser.error("--nonce must contain exactly 64 lowercase hexadecimal characters")
    if args.suite == "applications":
        if args.sdkbench_root is None:
            parser.error("--sdkbench-root is required for applications")
        args.sdkbench_root = args.sdkbench_root.resolve()
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.sdkbench_root, text=True, timeout=10
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=args.sdkbench_root,
            text=True,
            timeout=10,
        ).strip()
        if revision != _HARNESS_COMMIT or dirty:
            parser.error("Applications require the unchanged pinned SDKBench checkout")
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
    rows: list[dict[str, object]] = []
    report: DevelopmentReport = {
        "evidence_class": "development",
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(sabre.__file__).resolve().parents[3],
            text=True,
            timeout=10,
        ).strip(),
        "suite": args.suite,
        "input_nonce": args.nonce if args.suite == "applications" else None,
        "completed": False,
        "failures": [],
        "compiler_sha256": hashlib.sha256(
            Path(sabre.__file__).read_bytes()
        ).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_commit": _HARNESS_COMMIT if args.suite == "applications" else None,
        "python": sys.version,
        "platform": platform.platform(),
        "limits": "Local development inputs; a new nonce alone does not establish unseen programs or qualify resources. No qualified timing, memory, held-out generalization or ranking claim. Baseline-first order. Generic depth counts CX/SWAP equally. Applications at n=8,12 verify only three probes.",
        "rows": rows,
        "failure": None,
    }
    for case in cases:
        try:
            result = (
                _generic(*case)
                if args.suite == "generic"
                else _application(*case, args.sdkbench_root, nonce=args.nonce)
            )
            rows.append({"case": case, "result": result})
            print(len(rows), case, "passed", flush=True)
        except Exception as error:
            # Preserve each failed diagnostic attempt and finish the frozen case set.
            failure = {"type": type(error).__name__, "message": str(error)}
            if report["failure"] is None:
                report["failure"] = failure
            report["failures"].append({"case": case, **failure})
            print(case, "failed", failure, flush=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
