"""Compare exact IR encoding against a local Git baseline on inspected inputs.

The optional SDKBench probe runs existing adapters and independent acceptance.
It is a shared-host development diagnostic, not a resource-qualified campaign.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TypedDict, cast

import flagquantum.core.ir as core
from flagquantum.core.ir import CircuitIR, Instruction

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = "flagquantum/core/ir/__init__.py"
_Encode = Callable[[object], object]


class Measurement(TypedDict):
    samples_seconds: dict[str, list[float]]
    median_seconds: dict[str, float]
    output_sha256: str


class DevelopmentReport(TypedDict):
    scope: str
    base_commit: str
    source_sha256: dict[str, str]
    python: str
    platform: str
    cases: dict[str, Measurement]


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=_ROOT, text=True, timeout=30
    ).strip()


def _baseline(revision: str) -> tuple[str, _Encode]:
    commit = _git("rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}")
    tree = ast.parse(_git("show", f"{commit}:{_SOURCE}"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_encode_value"
    ]
    if len(selected) != 1:
        raise ValueError("Baseline must contain exactly one IR value encoder")
    module = ModuleType("_scalar_encoding_baseline")
    module.__dict__.update(vars(core))
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]), f"{commit}:{_SOURCE}", "exec"
        ),
        module.__dict__,
    )
    return commit, cast(_Encode, module._encode_value)


def _measure(
    run: Callable[[], object],
    baseline: _Encode,
    *,
    count: int,
    validate: Callable[[object], None] | None = None,
    normalize: Callable[[object], object] | None = None,
) -> Measurement:
    candidate = core._encode_value
    implementations = {"baseline": baseline, "candidate": candidate}
    samples: dict[str, list[float]] = {name: [] for name in implementations}
    expected: object = None
    try:
        for index in range(count + 2):
            order = list(implementations)
            if index % 2:
                order.reverse()
            for name in order:
                core._encode_value = implementations[name]
                start = time.perf_counter()
                actual = run()
                elapsed = time.perf_counter() - start
                comparable = actual if normalize is None else normalize(actual)
                if index == 0 and name == "baseline":
                    expected = comparable
                if comparable != expected:
                    raise AssertionError("Complete inspected output differs")
                if index == 0 and validate is not None:
                    validate(actual)
                if index >= 2:
                    samples[name].append(elapsed)
    finally:
        core._encode_value = candidate
    return {
        "samples_seconds": samples,
        "median_seconds": {
            name: statistics.median(items) for name, items in samples.items()
        },
        "output_sha256": hashlib.sha256(
            json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _sdkbench_probe(workflow: str, baseline: _Encode) -> Measurement:
    # Optional diagnostic dependency; no SDKBench object enters Core.
    import sdkbench
    from sdkbench.candidates.adapters import load_adapter, workflow_configuration
    from sdkbench.candidates.applications import fixture
    from sdkbench.p0 import Bundle
    from sdkbench.p0.evaluators import routing

    bundle = Bundle(
        Path(sdkbench.__file__).resolve().parents[2] / "docs/implementation"
    )
    configuration = workflow_configuration(workflow)
    adapter = load_adapter(
        SimpleNamespace(
            configuration={
                "options": {
                    "sdk": configuration["sdk"],
                    "workflow": workflow,
                    "compiler_options": configuration,
                }
            }
        )
    )
    selected = fixture(12, "qft", "line", "01" * 32)

    def run() -> object:
        return adapter.execute("route", selected, 0)

    def normalize(payload: object) -> object:
        if not isinstance(payload, dict):
            raise TypeError("Adapter output must be a mapping")
        return {
            "operations": payload["operations"],
            "final_layout": payload["final_layout"],
        }

    def validate(payload: object) -> None:
        if not isinstance(payload, dict):
            raise TypeError("Adapter output must be a mapping")
        metrics = routing(payload, selected, None, bundle.cases["ROUTE-LINE-12"], 0)
        if (
            metrics["max_semantic_error"]
            > bundle.cases["ROUTE-LINE-12"]["acceptance"]["semantic_atol"]
        ):
            raise AssertionError("Independent development acceptance failed")
        if payload["final_layout"] != list(range(12)):
            raise AssertionError("Restored output placement is required")

    return _measure(run, baseline, count=21, validate=validate, normalize=normalize)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", required=True, help="Locally available Git revision"
    )
    parser.add_argument(
        "--sdkbench",
        action="store_true",
        help="Run optional existing SDKBench adapters",
    )
    arguments = parser.parse_args()
    commit, baseline = _baseline(arguments.baseline)
    native = CircuitIR(
        12,
        tuple(
            Instruction("rz", (i % 12,), {"theta": 0.17 + i / 10000})
            for i in range(768)
        ),
    )
    annotated = CircuitIR(
        12,
        tuple(
            Instruction(
                "cx",
                (i % 12, (i + 1) % 12),
                metadata={
                    "layout_mapped": True,
                    "source_instruction_index": i,
                    "note": "routing",
                },
            )
            for i in range(768)
        ),
    )
    cases = {
        "native_serialization": _measure(native.to_json, baseline, count=41),
        "annotated_serialization": _measure(annotated.to_json, baseline, count=41),
    }
    if arguments.sdkbench:
        for workflow in (
            "flagquantum-ordinary-restored",
            "flagquantum-target-artifact",
        ):
            cases[workflow] = _sdkbench_probe(workflow, baseline)
    report: DevelopmentReport = {
        "scope": (
            "Inspected local CPU diagnostic, two warmups then alternating variant order. "
            "Exact IR JSON or adapter operations/final-layout equality; timing metadata excluded. Optional "
            "adapters use clock-dependent snapshots, so serialized artifacts are not "
            "compared across calls. No qualified campaign, memory, cold-start, held-out, "
            "cross-SDK ranking, statistical significance or scalability claim."
        ),
        "base_commit": commit,
        "source_sha256": {
            _SOURCE: hashlib.sha256((_ROOT / _SOURCE).read_bytes()).hexdigest()
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cases": cases,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
