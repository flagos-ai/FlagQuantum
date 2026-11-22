"""Prospective local application diagnosis with unchanged SDKBench acceptance."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import flagquantum

RUNTIME = "2ea5b78b6449b837db151e0d29c90895f6b51a4e"
HARNESS = "9aa9260920984e2ca05bcb293abb885fccd9c144"
WORKFLOWS = (
    "flagquantum-ordinary-restored",
    "flagquantum-target-artifact",
    "qiskit-transpile-restored",
    "cirq-route-cz-restored",
)
PROTOCOL = "docs/development/COMPILER_APPLICATION_GAPS.md"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, timeout=15
    ).strip()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _save(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdkbench-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    harness = args.sdkbench_root.resolve()
    manifest = json.loads(args.manifest.read_text())
    if Path(flagquantum.__file__).resolve().parents[1] != root:
        parser.error("Imported FlagQuantum must be this source checkout")
    if _git(root, "rev-parse", "HEAD") != manifest["implementation_commit"]:
        parser.error("Use the exact published freeze checkout")
    if _git(root, "diff", RUNTIME, "--", "flagquantum"):
        parser.error("Compiler implementation must equal the integrated runtime pin")
    if _git(harness, "rev-parse", "HEAD") != HARNESS or _git(
        harness, "status", "--porcelain"
    ):
        parser.error("SDKBench must match the clean frozen harness")
    if args.output.exists():
        parser.error("Do not overwrite an existing observation")
    for path, field in (
        (Path(__file__), "runner_sha256"),
        (root / PROTOCOL, "protocol_sha256"),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[field]:
            parser.error("Runner and protocol must match the pre-draw manifest")
    if (
        manifest["base_commit"] != RUNTIME
        or manifest["source_commit"] != HARNESS
        or manifest["draws"] != 1
    ):
        parser.error("Manifest source pins or entropy accounting differ")
    installed = {
        name: version(name) for name in ("qiskit", "cirq-core", "torch", "numpy")
    }
    if installed != manifest["versions"]:
        parser.error("Installed numerical SDK versions differ from the manifest")
    sys.path.insert(0, str(harness / "src"))
    from sdkbench.candidates.adapters import load_adapter, workflow_configuration
    from sdkbench.candidates.applications import FAMILIES, TOPOLOGIES, WIDTHS, fixture
    from sdkbench.p0 import Bundle
    from sdkbench.p0.evaluators import routing

    bundle = Bundle(harness / "docs/implementation")
    seeds = list(dict.fromkeys(bundle.plan["execution_profile"]["compiler_seeds"]))
    configurations = {name: workflow_configuration(name) for name in WORKFLOWS}
    if (
        seeds != manifest["compiler_seeds"]
        or configurations != manifest["configurations"]
    ):
        parser.error("Declared seeds or participant workflow configurations changed")
    adapters = {
        name: load_adapter(
            SimpleNamespace(
                configuration={
                    "options": {
                        "sdk": config["sdk"],
                        "workflow": name,
                        "compiler_options": config,
                    }
                }
            )
        )
        for name, config in configurations.items()
    }
    report: dict[str, Any] = {
        "evidence_class": "development",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "publication_eligible": False,
        "formal_campaign": "user_deferred",
        "implementation_commit": manifest["implementation_commit"],
        "base_commit": RUNTIME,
        "source_commit": HARNESS,
        "manifest": manifest,
        "python": sys.version,
        "platform": platform.platform(),
        "flagquantum_version": flagquantum.__version__,
        "runner_source": Path(__file__).read_text(),
        "protocol_source": (root / PROTOCOL).read_text(),
        "limits": "Shared macOS development host, one numerical thread, sequential calls. Unchanged adapter deadlines, operation caps and independent evaluator. No cgroup/RSS or formal resource executor qualification, isolated process blocks, latency ranking, hardware or scalability claim. Larger widths have three probes only.",
        "completed": False,
        "inputs": [],
        "rows": [],
        "payloads": {},
    }
    _save(args.output, report)
    for width, family, topology in itertools.product(WIDTHS, FAMILIES, TOPOLOGIES):
        selected = fixture(width, family, topology, manifest["input_nonce"])
        inspected = fixture(width, family, topology, "01" * 32)
        fingerprint = _digest(
            {key: selected[key] for key in ("operations", "target_edges")}
        )
        prior = _digest({key: inspected[key] for key in ("operations", "target_edges")})
        identity = f"{width:02d}/{family}/{topology}"
        report["inputs"].append(
            {
                "case": identity,
                "fixture_sha256": _digest(selected),
                "source_target_sha256": fingerprint,
                "inspected_source_target_sha256": prior,
                "source_target_is_new": fingerprint != prior,
            }
        )
        acceptance = bundle.cases[
            "ROUTE-GRID-08" if width == 8 else f"ROUTE-LINE-{width:02d}"
        ]
        for seed in seeds:
            # Rotate the same complete workflow list to avoid fixed participant order.
            offset = len(report["rows"]) // len(WORKFLOWS) % len(WORKFLOWS)
            order = WORKFLOWS[offset:] + WORKFLOWS[:offset]
            for name in order:
                row: dict[str, Any] = {
                    "case": identity,
                    "seed": seed,
                    "workflow": name,
                    "status": "failed",
                }
                try:
                    case_fixture = deepcopy(selected)
                    payload = adapters[name].execute("route", case_fixture, seed)
                    if _digest(case_fixture) != _digest(selected):
                        raise AssertionError("Participant mutated the declared fixture")
                    normalized = _canonical(
                        {key: payload[key] for key in ("operations", "final_layout")}
                    )
                    digest = hashlib.sha256(normalized.encode()).hexdigest()
                    report["payloads"][digest] = normalized
                    row.update(
                        payload_sha256=digest,
                        compile_wall_seconds=payload["compile_wall_seconds"],
                        audit=dict(adapters[name].audit),
                    )
                    row["metrics"] = routing(payload, selected, None, acceptance, 0)
                    if payload["final_layout"] != list(range(width)):
                        raise AssertionError("Delivered final layout must be identity")
                    row["status"] = "accepted"
                except Exception as error:
                    row.update(error_type=type(error).__name__, error=str(error))
                report["rows"].append(row)
            _save(args.output, report)
            print(
                json.dumps(
                    {
                        "phase": "acceptance",
                        "case": identity,
                        "seed": seed,
                        "attempted": len(report["rows"]),
                        "failures": sum(
                            row["status"] != "accepted" for row in report["rows"]
                        ),
                    }
                ),
                flush=True,
            )
    report["completed"] = True
    _save(args.output, report)


if __name__ == "__main__":
    main()
