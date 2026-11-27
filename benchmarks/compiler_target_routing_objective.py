"""Inspect restored route choices against delivered native CZ cost, without promotion."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict
from unittest.mock import patch

import flagquantum
from flagquantum.compiler.artifacts import artifact_compilation

Case = tuple[int, str, str]
_HARNESS = "9aa9260920984e2ca05bcb293abb885fccd9c144"
_COMPILE = artifact_compilation.compile_circuit_artifact_for_target
_STRATEGIES = ("restore_after_each_gate", "persistent_layout", "sabre")


class Observation(TypedDict):
    requested_strategy: str
    selected_strategy: str
    restored_swap_count: int
    native_baseline_cz_count: int
    block_cz_savings: list[int]
    metrics: dict[str, Any]
    normalized_output_sha256: str
    seconds: float


class CaseResult(TypedDict):
    case: Case
    observations: list[Observation]
    selected_index: int | None
    portfolio_seconds: float
    failures: list[dict[str, str]]


class Report(TypedDict):
    evidence_class: str
    implementation_commit: str
    source_commit: str
    runner_sha256: str
    python: str
    platform: str
    limits: str
    rows: list[CaseResult]
    completed: bool


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=root, text=True, timeout=15
    ).strip()


def _observe(case: Case, harness: Path, strategy: str) -> Observation:
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

    # This is the explicitly varied diagnostic boundary. The pinned adapter and
    # evaluator are unmodified; this override is not its registered formal profile.
    def compile_with_strategy(*args: Any, **kwargs: Any) -> object:
        if "routing_strategy" in kwargs:
            raise AssertionError("The inspected adapter now specifies routing")
        result = _COMPILE(*args, **kwargs, routing_strategy=strategy)
        captured.append(result)
        return result

    selected = fixture(width, family, topology, "01" * 32)
    seed = bundle.plan["execution_profile"]["compiler_seeds"][0]
    with patch.object(
        artifact_compilation,
        "compile_circuit_artifact_for_target",
        compile_with_strategy,
    ):
        start = time.perf_counter()
        payload = adapter.execute("route", selected, seed)
        elapsed = time.perf_counter() - start
    metrics = routing(payload, selected, None, acceptance, 0)
    if len(captured) != 1:
        raise AssertionError("Expected one native artifact compilation")
    result = captured[0]
    route = result.legalization.topology_legalization
    if route is None or (
        route.initial_logical_to_physical != tuple(range(width))
        or route.final_logical_to_physical != tuple(range(width))
        or payload["final_layout"] != list(range(width))
    ):
        raise AssertionError("The inspected route changed endpoint layout contracts")
    stage = result.legalization.native_block_optimization
    normalized = json.dumps(
        {key: payload[key] for key in ("operations", "final_layout")},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        "requested_strategy": strategy,
        "selected_strategy": route.strategy,
        "restored_swap_count": route.inserted_swap_count,
        "native_baseline_cz_count": adapter.audit["native_baseline_cz_count"],
        "block_cz_savings": (
            [r.baseline_cz_count - r.replacement_cz_count for r in stage.rewrites]
            if stage is not None
            else []
        ),
        "metrics": metrics,
        "normalized_output_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "seconds": elapsed,
    }


def _quality(observation: Observation) -> tuple[int, int]:
    return (
        observation["metrics"]["full_cz_count"],
        observation["metrics"]["two_body_dependency_depth"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdkbench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(flagquantum.__file__).resolve().parents[1]
    harness = args.sdkbench_root.resolve()
    if _git(harness, "rev-parse", "HEAD") != _HARNESS or _git(
        harness, "status", "--porcelain"
    ):
        parser.error("The pinned SDKBench checkout must be unchanged")
    if _git(source, "diff", "HEAD", "--", "flagquantum"):
        parser.error("Compiler source must match its recorded commit")
    sys.path.insert(0, str(harness / "src"))
    report: Report = {
        "evidence_class": "development",
        "implementation_commit": _git(source, "rev-parse", "HEAD"),
        "source_commit": _HARNESS,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "limits": "Inspected development inputs, shared macOS CPU, one numerical thread. Diagnostic wrapper varies only the existing native routing keyword; this is not the registered formal participant workflow. Fixed initial and final identity layouts, unchanged independent evaluator and caps. Default then other deployable strategies; three attempts when default succeeds, four if it fails. Failures are retained and cause nonzero exit after the full inspection. One timed sample each, no warmups, no significance, memory, held-out, stable speedup, hardware, scalability or SDK ranking claim. Six-qubit complete operator; three probes at eight/twelve. Accepted timing includes native artifact verification and serialization, excludes independent evaluation, setup and startup. Portfolio seconds sum only successful intervals; a portfolio with failures is incomplete.",
        "rows": [],
        "completed": False,
    }

    def checkpoint() -> None:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    for case in itertools.product(
        (6, 8, 12),
        ("qaoa", "ising", "hopping", "qft", "repetition"),
        ("line", "ring", "grid", "sparse"),
    ):
        observations: list[Observation] = []
        failures: list[dict[str, str]] = []
        default_selected: str | None = None
        for strategy in ("auto", *_STRATEGIES):
            if strategy == default_selected:
                continue
            try:
                observation = _observe(case, harness, strategy)
            except Exception as error:
                # This is the diagnostic job boundary, not compiler fallback.
                # Retain each failure and exit nonzero after independent controls.
                failures.append(
                    {
                        "strategy": strategy,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(case, strategy, repr(error), flush=True)
                continue
            if strategy == "auto":
                default_selected = observation["selected_strategy"]
                if default_selected not in _STRATEGIES:
                    raise AssertionError("Default selected an undeclared strategy")
            observations.append(observation)
        # Stable min preserves the incumbent for exactly equal delivered cost.
        selected = (
            min(range(len(observations)), key=lambda i: _quality(observations[i]))
            if observations
            else None
        )
        report["rows"].append(
            {
                "case": case,
                "observations": observations,
                "selected_index": selected,
                "portfolio_seconds": sum(row["seconds"] for row in observations),
                "failures": failures,
            }
        )
        checkpoint()
        print(len(report["rows"]), case, "inspected", flush=True)
    report["completed"] = True
    checkpoint()
    if any(row["failures"] for row in report["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
