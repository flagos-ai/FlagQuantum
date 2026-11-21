"""Check private matrix products against an exact Git baseline on inspected inputs.

The numerical check uses only the standard library. Optional workflows reuse the
existing unchanged SDKBench acceptance helper and do not qualify performance.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
import platform
import random
import struct
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast
from unittest.mock import patch

Matrix = list[list[complex]]
Product = Callable[[Matrix, Matrix], Matrix]
_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = "flagquantum/compiler/optimization/two_qubit_optimization.py"
_BASE = "b3ed982a0d90a05ec69b093723aa1b6e83630463"
_HARNESS = "9aa9260920984e2ca05bcb293abb885fccd9c144"


class DevelopmentReport(TypedDict, total=False):
    scope: str
    base_commit: str
    implementation_commit: str
    candidate_source_sha256: str
    baseline_source_sha256: str
    runner_sha256: str
    python: str
    platform: str
    numerical_cases: int
    complete: bool
    source_commit: str
    workflows: list[dict[str, object]]


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, timeout=20
    ).strip()


def _load(source: str) -> Product:
    node = next(
        item
        for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef) and item.name == "_matmul"
    )
    namespace: dict[str, object] = {}
    exec("from __future__ import annotations\n" + ast.unparse(node), namespace)
    return cast(Product, namespace["_matmul"])


def _bits(matrix: Matrix) -> bytes:
    return b"".join(
        struct.pack("!dd", complex(value).real, complex(value).imag)
        for row in matrix
        for value in row
    )


def _numerical(baseline: Product, candidate: Product) -> int:
    rng = random.Random(982137)
    edges = (0.0, -0.0, 1.0, -1.0, math.inf, -math.inf, math.nan, 1e-300, 1e300)
    for index in range(2400):
        matrices: list[Matrix] = []
        for _ in range(2):
            values: list[complex] = []
            for _ in range(16):
                real = rng.choice(edges) if index >= 1200 else rng.uniform(-2, 2)
                imag = rng.choice(edges) if index >= 1200 else rng.uniform(-2, 2)
                # Exercise real-only and mixed runtime values as well as complex.
                value = real if index % 3 == 0 else complex(real, imag)
                values.append(value)
            matrices.append([values[start : start + 4] for start in range(0, 16, 4)])
        left, right = matrices
        if _bits(baseline(left, right)) != _bits(candidate(left, right)):
            raise AssertionError(f"Matrix result differs at deterministic case {index}")
    return 2400


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sdkbench-root", type=Path)
    args = parser.parse_args()
    baseline_source = _git(_ROOT, "show", f"{_BASE}:{_SOURCE}")
    candidate_source = (_ROOT / _SOURCE).read_text()
    baseline, candidate = _load(baseline_source), _load(candidate_source)
    report: DevelopmentReport = {
        "scope": "single_device_fast_path; inspected development controls; no ranking, qualified performance or scalability claim",
        "base_commit": _BASE,
        "implementation_commit": _git(_ROOT, "rev-parse", "HEAD"),
        "candidate_source_sha256": hashlib.sha256(
            candidate_source.encode()
        ).hexdigest(),
        "baseline_source_sha256": hashlib.sha256(baseline_source.encode()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numerical_cases": _numerical(baseline, candidate),
        "complete": False,
    }
    rows: list[dict[str, object]] = []
    report["workflows"] = rows
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.sdkbench_root is not None:
        harness = args.sdkbench_root.resolve()
        if _git(harness, "rev-parse", "HEAD") != _HARNESS or _git(
            harness, "status", "--porcelain"
        ):
            parser.error("The pinned SDKBench checkout must be unchanged")
        sys.path.insert(0, str(harness / "src"))
        import flagquantum.compiler.optimization._cz_blocks as blocks
        import flagquantum.compiler.optimization.two_qubit_optimization as matrices
        from benchmarks.compiler_cz_identity_prefix import _observe

        report["source_commit"] = _HARNESS
        original = matrices._matmul
        for case in itertools.product(
            (6, 8, 12),
            ("qaoa", "ising", "hopping", "qft", "repetition"),
            ("line", "ring", "grid", "sparse"),
        ):
            observations = {}
            for name, function in (("baseline", baseline), ("candidate", original)):
                with (
                    patch.object(matrices, "_matmul", function),
                    patch.object(blocks, "_matmul", function),
                ):
                    observations[name] = _observe(case, harness)
            comparable = (
                "metrics",
                "selected_strategy",
                "restored_swap_count",
                "rewrite_count",
                "maximum_rewrite_inputs",
                "normalized_output_sha256",
            )
            if any(
                observations["baseline"][key] != observations["candidate"][key]
                for key in comparable
            ):
                raise AssertionError(f"Complete compilation differs: {case}")
            rows.append({"case": case, "observations": observations})
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(len(rows), case, "accepted with identical outputs", flush=True)
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
