#!/usr/bin/env python3
"""Run the CPU-safe checks that should block a FlagQuantum push.

The hook reuses checked-in pre-commit hooks and CI tiers. It intentionally
leaves clean-environment version matrices, package installation, supply-chain
audits, and accelerator jobs to GitHub Actions.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

Command = tuple[str, ...]


@dataclass(frozen=True)
class Check:
    name: str
    command: Command


def environment_executable(python_executable: str, name: str) -> str:
    python_path = Path(python_executable)
    if python_path.is_absolute():
        candidate = python_path.with_name(name)
        if candidate.is_file():
            return str(candidate)
    return name


def checks(python_executable: str) -> tuple[Check, ...]:
    pre_commit = environment_executable(python_executable, "pre-commit")
    mypy = environment_executable(python_executable, "mypy")
    return (
        Check(
            "pre-commit",
            (
                pre_commit,
                "run",
                "--all-files",
                "--hook-stage",
                "pre-commit",
                "--show-diff-on-failure",
            ),
        ),
        Check(
            "strict typed trainable module",
            (
                mypy,
                "--strict",
                "--no-site-packages",
                "--ignore-missing-imports",
                "--follow-imports",
                "skip",
                "flagquantum/runtime/module.py",
                "flagquantum/runtime/training.py",
            ),
        ),
        Check(
            "strict typed execution mainline",
            (
                mypy,
                "--strict",
                "--no-site-packages",
                "--ignore-missing-imports",
                "--follow-imports",
                "skip",
                "flagquantum/compilation",
                "flagquantum/runtime/execution.py",
                "flagquantum/runtime/distributed/protocols.py",
                "flagquantum/runtime/backend_registry.py",
                "flagquantum/deployment",
            ),
        ),
        Check(
            "strict typed interoperability contract",
            (
                mypy,
                "--strict",
                "--no-site-packages",
                "--ignore-missing-imports",
                "--follow-imports",
                "skip",
                "flagquantum/interop/contracts.py",
                "flagquantum/interop/conformance.py",
                "flagquantum/interop/registry.py",
                "flagquantum/interop/pennylane/adapter.py",
                "flagquantum/interop/pennylane/conformance.py",
                "flagquantum/interop/pennylane/models.py",
                "flagquantum/interop/qiskit/adapter.py",
                "flagquantum/interop/qiskit/conformance.py",
                "flagquantum/interop/qiskit/models.py",
            ),
        ),
        Check(
            "capability maturity",
            (python_executable, "tools/check_capability_maturity.py"),
        ),
        Check(
            "dependency policy",
            (python_executable, "tools/check_dependency_policy.py"),
        ),
        Check(
            "Qiskit interoperability contract",
            (python_executable, "tools/check_qiskit_interop_contract.py"),
        ),
        Check(
            "PennyLane interoperability contract",
            (python_executable, "tools/check_pennylane_interop_contract.py"),
        ),
        Check(
            "required-check contract",
            (python_executable, "tools/validate_required_checks.py"),
        ),
        Check(
            "lazy import budget",
            (python_executable, "tools/check_import_time.py"),
        ),
        Check(
            "smoke and unit tier",
            (python_executable, "tools/ci_tier.py", "pr-default"),
        ),
        Check(
            "runtime integration tier",
            (python_executable, "tools/ci_tier.py", "pr-runtime"),
        ),
        Check(
            "distributed CPU and release-contract tier",
            (python_executable, "tools/ci_tier.py", "pr-distributed"),
        ),
    )


def run_checks(
    selected: Sequence[Check],
    *,
    root: Path,
    dry_run: bool,
    python_executable: str | None = None,
) -> int:
    environment = os.environ.copy()
    environment.setdefault(
        "BLACK_CACHE_DIR",
        str(Path(tempfile.gettempdir()) / "flagquantum-black-cache"),
    )
    if python_executable:
        python_bin = str(Path(python_executable).resolve().parent)
        environment["PATH"] = os.pathsep.join((python_bin, environment.get("PATH", "")))
    for index, check in enumerate(selected, start=1):
        rendered = " ".join(check.command)
        print(
            f"\n[{index}/{len(selected)}] {check.name}\n+ {rendered}",
            flush=True,
        )
        if dry_run:
            continue
        executable = check.command[0]
        if shutil.which(executable, path=environment["PATH"]) is None:
            print(
                f"error: required executable is not installed: {executable}",
                file=sys.stderr,
            )
            return 127
        result = subprocess.run(
            check.command,
            cwd=root,
            check=False,
            env=environment,
        )
        if result.returncode != 0:
            print(
                f"\npre-push blocked by: {check.name}",
                file=sys.stderr,
            )
            return result.returncode
    print("\nFlagQuantum pre-push gate passed.", flush=True)
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the gate without executing commands.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used by checked-in CI tools.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parents[1]
    return run_checks(
        checks(args.python),
        root=root,
        dry_run=args.dry_run,
        python_executable=args.python,
    )


if __name__ == "__main__":
    raise SystemExit(main())
